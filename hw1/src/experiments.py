"""Baseline experiment orchestration and comparison helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.data import build_loaders, build_train_transform, clone_with_transform
from src.models import build_model, build_optimizer
from src.training import (
    evaluate_model,
    load_best_model,
    train_model,
)
from src.utils import set_seed
from src.visualization import plot_history


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
        history = None
        planned_epochs = 0
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
            history = train_model(
                train_loader=stage_train_loader,
                epochs=stage.epochs,
                history=history,  # continues epochs / best F1 / patience
                **train_kwargs,
            )
            planned_epochs += stage.epochs
            if len(history["epoch"]) < planned_epochs:
                print("Early stopping triggered, remaining stages skipped")
                break

    train_minutes = (time.perf_counter() - started) / 60.0
    best_checkpoint = config.checkpoint_dir / f"{experiment_name}_best.pt"

    # Restore the checkpoint with the best validation macro F1
    model = load_best_model(
        model=model,
        checkpoint_path=best_checkpoint,
        device=config.device,
        use_channels_last=settings.channels_last,
    )
    # Evaluate the best checkpoint on val and test
    val_metrics = evaluate_model(
        model=model,
        loader=val_loader,
        criterion=config.criterion,
        num_classes=config.num_classes,
        device=config.device,
        use_channels_last=settings.channels_last,
    )
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=config.criterion,
        num_classes=config.num_classes,
        device=config.device,
        use_channels_last=settings.channels_last,
    )

    print("\nBest validation metrics:")
    print(val_metrics)
    print("\nTest metrics:")
    print(test_metrics)

    # Plot the history
    plot_history(
        history,
        title_prefix=f"{title_prefix}: {model_name}",
    )

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
