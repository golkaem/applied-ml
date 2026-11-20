"""Baseline experiment orchestration and comparison helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data import build_loaders, build_train_transform, clone_with_transform
from src.models import (
    assert_groups_cover_model,
    build_model,
    build_optimizer,
    freeze_all_parameters,
    get_unfreeze_groups,
    unfreeze_group,
)
from src.training import (
    evaluate_model,
    load_best_model,
    train_model,
)
from src.utils import set_seed
from src.visualization import plot_history


@dataclass(frozen=True)
class UnfreezeStage:
    """One stage of iterative unfreezing."""

    group: str  # "head", "late", "mid", "early" (in this order)
    epochs: int


@dataclass(frozen=True)
class AugmentationStage:
    """One stage of iterative augmentation."""

    level: int  # 0, 1 or 2
    epochs: int


@dataclass(frozen=True)
class ModelSettings:
    """Per-model baseline training configuration."""

    batch_size: int
    channels_last: bool
    epochs: int
    early_stopping_patience: Optional[int] = None


@dataclass(frozen=True)
class BaselineConfig:
    """Shared data, runtime, and optimization settings for the baseline."""

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    criterion: nn.Module
    checkpoint_dir: Path
    device: torch.device
    seed: int
    num_classes: int
    num_workers: int
    backbone_lr: float
    classifier_lr: float
    weight_decay: float
    baseline_prefix: str


@dataclass
class ExperimentResult:
    """Results of one baseline experiment."""

    model_name: str
    model: nn.Module
    history: dict[str, list[float]]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_minutes: float
    best_checkpoint: Path


def _finalize_experiment(
    model: nn.Module,
    model_name: str,
    settings: ModelSettings,
    config: BaselineConfig,
    val_loader: DataLoader,
    test_loader: DataLoader,
    history: dict[str, list[float]],
    experiment_name: str,
    train_minutes: float,
    title: str,
) -> ExperimentResult:
    """Load the best checkpoint, evaluate, plot, and pack the result."""
    best_checkpoint = config.checkpoint_dir / f"{experiment_name}_best.pt"

    # Restore the checkpoint with the best validation macro F1
    model = load_best_model(
        model=model,
        checkpoint_path=best_checkpoint,
        device=config.device,
        use_channels_last=settings.channels_last,
    )
    # Evaluate the best checkpoint on val and test
    eval_kwargs = dict(
        model=model,
        criterion=config.criterion,
        num_classes=config.num_classes,
        device=config.device,
        use_channels_last=settings.channels_last,
    )
    val_metrics = evaluate_model(loader=val_loader, **eval_kwargs)
    test_metrics = evaluate_model(loader=test_loader, **eval_kwargs)

    print("\nBest validation metrics:")
    print(val_metrics)
    print("\nTest metrics:")
    print(test_metrics)

    # Plot the history
    plot_history(history, title_prefix=title)

    # Return results
    return ExperimentResult(
        model_name=model_name,
        model=model,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_minutes=train_minutes,
        best_checkpoint=best_checkpoint,
    )


def run_baseline_experiment(
    model_name: str,
    settings: ModelSettings,
    config: BaselineConfig,
    aug_stages: Optional[Sequence[AugmentationStage]] = None,
    experiment_prefix: Optional[str] = None,
    title_prefix: str = "Baseline",
) -> ExperimentResult:
    """Train and evaluate one full fine-tuning baseline."""
    # Reseed before every experiment
    # No dependance on previously executed code
    set_seed(config.seed)

    train_loader, val_loader, test_loader = build_loaders(
        train_dataset=config.train_dataset,
        val_dataset=config.val_dataset,
        test_dataset=config.test_dataset,
        batch_size=settings.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    print(
        f"\n{model_name}: batch_size={settings.batch_size}, "
        f"channels_last={settings.channels_last}, "
        f"epochs={settings.epochs}, "
        f"train batches={len(train_loader)}"
    )

    # Create a new pretrained model for each experiment
    model, classifier_prefix = build_model(
        model_name,
        config.num_classes,
    )
    # Move the model before creating the optimizer so the optimizer always
    # references the parameter tensors that will actually be trained
    if settings.channels_last:
        model = model.to(
            device=config.device,
            memory_format=torch.channels_last,
        )
    else:
        model = model.to(config.device)

    optimizer = build_optimizer(
        model=model,
        classifier_prefix=classifier_prefix,
        backbone_lr=config.backbone_lr,
        classifier_lr=config.classifier_lr,
        weight_decay=config.weight_decay,
    )

    prefix = experiment_prefix or config.baseline_prefix
    experiment_name = f"{prefix}_{model_name}"
    started = time.perf_counter()

    # Train
    train_kwargs = dict(
        model=model,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=config.criterion,
        num_classes=config.num_classes,
        checkpoint_dir=config.checkpoint_dir,
        experiment_name=experiment_name,
        scheduler=None,
        early_stopping_patience=settings.early_stopping_patience,
        device=config.device,
        use_channels_last=settings.channels_last,
    )

    if aug_stages is None:
        # Baseline behaviour
        history = train_model(
            train_loader=train_loader,
            epochs=settings.epochs,
            history=None,
            **train_kwargs,
        )
    else:
        # Staged schedules must use the same maximum epoch budget as the
        # baseline so that the experiments remain comparable
        staged_epochs = sum(stage.epochs for stage in aug_stages)
        if staged_epochs != settings.epochs:
            raise ValueError(
                f"Augmentation stages sum to {staged_epochs} epochs, "
                f"expected {settings.epochs}"
            )

        history = None

        for stage_idx, stage in enumerate(aug_stages):
            print(
                f"\nAugmentation stage {stage_idx + 1}/{len(aug_stages)}: "
                f"level={stage.level}, epochs={stage.epochs}"
            )
            stage_dataset = clone_with_transform(
                config.train_dataset,
                build_train_transform(
                    stage.level,
                    image_size=config.train_dataset.images.shape[1],
                ),
            )
            # New train loader per stage; seed shifted so the shuffle order
            # differs between stages
            # Val/test loaders are reused
            stage_train_loader, _, _ = build_loaders(
                train_dataset=stage_dataset,
                val_dataset=config.val_dataset,
                test_dataset=config.test_dataset,
                batch_size=settings.batch_size,
                num_workers=config.num_workers,
                seed=config.seed + stage_idx,
            )
            epochs_before = len(history["epoch"]) if history is not None else 0
            history = train_model(
                train_loader=stage_train_loader,
                epochs=stage.epochs,
                history=history,  # continues epochs and best F1
                reset_early_stopping=True,  # patience restarts per stage
                **train_kwargs,
            )
            epochs_run = len(history["epoch"]) - epochs_before
            if epochs_run < stage.epochs:
                # A new augmentation level is a separate training regime.
                # Stop only this stage early and continue with the next level.
                print(
                    f"Stage stopped early after {epochs_run} of "
                    f"{stage.epochs} epochs"
                )

    train_minutes = (time.perf_counter() - started) / 60.0

    return _finalize_experiment(
        model=model,
        model_name=model_name,
        settings=settings,
        config=config,
        val_loader=val_loader,
        test_loader=test_loader,
        history=history,
        experiment_name=experiment_name,
        train_minutes=train_minutes,
        title=f"{title_prefix}: {model_name}",
    )


def load_experiment_result(
    model_name: str,
    settings: ModelSettings,
    config: BaselineConfig,
    experiment_name: str,
    train_minutes: Optional[float] = None,
) -> ExperimentResult:
    """Rebuild an experiment result from saved artifacts without training."""
    best_checkpoint = config.checkpoint_dir / f"{experiment_name}_best.pt"
    history_path = config.checkpoint_dir / f"{experiment_name}_history.json"

    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_checkpoint}")

    if not history_path.exists():
        raise FileNotFoundError(f"History not found: {history_path}")

    with history_path.open(encoding="utf-8") as file:
        history = json.load(file)

    if train_minutes is None:
        # Older runs store the training time only in the comparison table
        comparison_path = config.checkpoint_dir / "baseline_comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path, index_col="Model")
            if model_name in comparison.index:
                train_minutes = float(
                    comparison.loc[model_name, "train_minutes"]
                )

    if train_minutes is None:
        train_minutes = 0.0

    _, val_loader, test_loader = build_loaders(
        train_dataset=config.train_dataset,
        val_dataset=config.val_dataset,
        test_dataset=config.test_dataset,
        batch_size=settings.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    model, _ = build_model(model_name, config.num_classes)
    model = load_best_model(
        model=model,
        checkpoint_path=best_checkpoint,
        device=config.device,
        use_channels_last=settings.channels_last,
    )

    eval_kwargs = dict(
        model=model,
        criterion=config.criterion,
        num_classes=config.num_classes,
        device=config.device,
        use_channels_last=settings.channels_last,
    )
    val_metrics = evaluate_model(loader=val_loader, **eval_kwargs)
    test_metrics = evaluate_model(loader=test_loader, **eval_kwargs)

    return ExperimentResult(
        model_name=model_name,
        model=model,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_minutes=train_minutes,
        best_checkpoint=best_checkpoint,
    )


def build_comparison_table(
    results: Mapping[str, ExperimentResult],
) -> pd.DataFrame:
    """Flatten baseline results into a comparable DataFrame."""
    comparison = {}

    for model_name, result in results.items():
        comparison[model_name] = {
            **{
                f"val_{metric_name}": value
                for metric_name, value in result.val_metrics.items()
            },
            **{
                f"test_{metric_name}": value
                for metric_name, value in result.test_metrics.items()
            },
            "train_minutes": result.train_minutes,
            "epochs_run": len(result.history["epoch"]),
        }

    dataframe = pd.DataFrame(comparison).T
    dataframe.index.name = "Model"

    return dataframe


def _train_with_unfreezing(
    model: nn.Module,
    model_name: str,
    classifier_prefix: str,
    stages: Sequence[UnfreezeStage],
    lr_factor: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: BaselineConfig,
    settings: ModelSettings,
    experiment_name: str,
    early_stopping_patience: Optional[int],
) -> dict[str, list[float]]:
    """Train stage by stage, unfreezing one more part before each stage."""
    groups = get_unfreeze_groups(model_name)
    order = [name for name, _ in groups]
    prefixes_by_name = dict(groups)
    names = [stage.group for stage in stages]

    if names != order[: len(names)] or not names:
        raise ValueError(f"Stages must follow the order {order}, got {names}")

    assert_groups_cover_model(model, model_name)
    freeze_all_parameters(model)

    optimizer = None
    history = None

    for index, stage in enumerate(stages):
        parameters = unfreeze_group(model, prefixes_by_name[stage.group])
        # Each next part gets an lr lr_factor times lower than the previous one
        stage_lr = config.classifier_lr / (lr_factor**index)

        if optimizer is None:
            # Only the head is trainable here, so a single "head" group
            optimizer = build_optimizer(
                model=model,
                classifier_prefix=classifier_prefix,
                backbone_lr=config.backbone_lr,
                classifier_lr=stage_lr,
                weight_decay=config.weight_decay,
            )
        else:
            # Existing groups and their Adam state stay untouched
            optimizer.add_param_group(
                {"params": parameters, "lr": stage_lr, "name": stage.group}
            )

        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            f"\nUnfreezing stage {index + 1}/{len(stages)}: "
            f"+{stage.group} (lr={stage_lr:.0e}), epochs={stage.epochs}, "
            f"trainable params={trainable:,}"
        )

        epochs_before = len(history["epoch"]) if history is not None else 0
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=config.criterion,
            num_classes=config.num_classes,
            epochs=stage.epochs,
            checkpoint_dir=config.checkpoint_dir,
            experiment_name=experiment_name,
            scheduler=None,
            history=history,
            early_stopping_patience=early_stopping_patience,
            reset_early_stopping=True,  # patience restarts per stage
            device=config.device,
            use_channels_last=settings.channels_last,
        )

        epochs_run = len(history["epoch"]) - epochs_before
        if epochs_run < stage.epochs:
            # A newly unfrozen group is a separate training regime.
            # Stop only this stage early and continue with the next group.
            print(
                f"Stage stopped early after {epochs_run} of "
                f"{stage.epochs} epochs"
            )

    return history


def run_unfreeze_experiment(
    model_name: str,
    settings: ModelSettings,
    config: BaselineConfig,
    stages: Sequence[UnfreezeStage],
    lr_factor: float = 10.0,
    experiment_prefix: str = "unfreeze_local",
    use_early_stopping: bool = True,
    title_prefix: str = "Iterative unfreezing",
) -> ExperimentResult:
    """Train a new pretrained model with iterative unfreezing."""
    if lr_factor < 10:
        raise ValueError("lr_factor must be >= 10")

    # The staged schedule must use the same maximum epoch budget as the
    # baseline so that the experiments remain comparable
    total_epochs = sum(stage.epochs for stage in stages)
    if total_epochs != settings.epochs:
        raise ValueError(
            f"Unfreezing stages sum to {total_epochs} epochs, "
            f"expected {settings.epochs}"
        )

    set_seed(config.seed)

    train_loader, val_loader, test_loader = build_loaders(
        train_dataset=config.train_dataset,
        val_dataset=config.val_dataset,
        test_dataset=config.test_dataset,
        batch_size=settings.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    print(
        f"\n{model_name}: batch_size={settings.batch_size}, "
        f"channels_last={settings.channels_last}, "
        f"stages={[(s.group, s.epochs) for s in stages]}, "
        f"total epochs={total_epochs} (baseline: {settings.epochs})"
    )

    # A new pretrained model for every experiment
    model, classifier_prefix = build_model(model_name, config.num_classes)
    if settings.channels_last:
        model = model.to(
            device=config.device, memory_format=torch.channels_last
        )
    else:
        model = model.to(config.device)

    experiment_name = f"{experiment_prefix}_{model_name}"
    started = time.perf_counter()

    history = _train_with_unfreezing(
        model=model,
        model_name=model_name,
        classifier_prefix=classifier_prefix,
        stages=stages,
        lr_factor=lr_factor,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        settings=settings,
        experiment_name=experiment_name,
        early_stopping_patience=(
            settings.early_stopping_patience if use_early_stopping else None
        ),
    )

    train_minutes = (time.perf_counter() - started) / 60.0

    return _finalize_experiment(
        model=model,
        model_name=model_name,
        settings=settings,
        config=config,
        val_loader=val_loader,
        test_loader=test_loader,
        history=history,
        experiment_name=experiment_name,
        train_minutes=train_minutes,
        title=f"{title_prefix}: {model_name}",
    )
