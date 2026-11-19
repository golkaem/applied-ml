"""Training, evaluation, metrics, and checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> dict[str, float]:
    """Calculate accuracy and macro/weighted F1 over all classes."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    use_channels_last: bool = False,
) -> dict[str, float]:
    """Run one training or evaluation epoch and return loss/metrics."""
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images_batch, labels_batch in loader:
            if use_channels_last:
                images_batch = images_batch.to(
                    device=device,
                    memory_format=torch.channels_last,
                )
            else:
                images_batch = images_batch.to(device)

            labels_batch = labels_batch.to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_targets.extend(labels_batch.detach().cpu().tolist())

    epoch_loss = total_loss / total_samples
    metrics = calculate_metrics(all_targets, all_predictions, num_classes)

    return {
        "loss": float(epoch_loss),
        **metrics,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_classes: int,
    epochs: int,
    checkpoint_dir: Path,
    experiment_name: str,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    history: Optional[dict[str, list[float]]] = None,
    early_stopping_patience: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    use_channels_last: bool = False,
) -> dict[str, list[float]]:
    """Train a model, save checkpoints/history, and return the history."""
    if use_channels_last:
        model = model.to(
            device=device,
            memory_format=torch.channels_last,
        )
    else:
        model = model.to(device)

    # Initialize history
    if history is None:
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "train_weighted_f1": [],
            "val_weighted_f1": [],
        }

    # Continue from the number of epochs already in history
    start_epoch = len(history["epoch"]) + 1
    best_val_f1 = max(
        history["val_macro_f1"],
        default=-float("inf"),
    )
    epochs_without_improvement = 0

    # Count epochs_without_improvement so early stopping also accounts for
    # epochs completed
    for value in reversed(history["val_macro_f1"]):
        if value >= best_val_f1:
            break
        epochs_without_improvement += 1

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # _best.pt keeps weights only for evaluation; _last.pt also keeps training
    # state so a run can be resumed
    best_path = checkpoint_dir / f"{experiment_name}_best.pt"
    last_path = checkpoint_dir / f"{experiment_name}_last.pt"
    history_path = checkpoint_dir / f"{experiment_name}_history.json"

    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch + local_epoch - 1

        # Train
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            num_classes=num_classes,
            optimizer=optimizer,
            device=device,
            use_channels_last=use_channels_last,
        )
        # Val
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            num_classes=num_classes,
            optimizer=None,
            device=device,
            use_channels_last=use_channels_last,
        )

        # Scheduler step
        if scheduler is not None:
            if isinstance(
                scheduler,
                optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        history["epoch"].append(epoch)

        # Append train/validation metrics for this epoch
        for metric_name in [
            "loss",
            "accuracy",
            "macro_f1",
            "weighted_f1",
        ]:
            history[f"train_{metric_name}"].append(
                float(train_metrics[metric_name])
            )
            history[f"val_{metric_name}"].append(
                float(val_metrics[metric_name])
            )

        for index, group in enumerate(optimizer.param_groups):
            group_name = group.get("name", f"group_{index}")
            # Record the learning rate of every parameter group
            # On resume, pad the learning rate list to the current epoch
            history.setdefault(
                f"lr_{group_name}",
                [0.0] * (len(history["epoch"]) - 1),
            ).append(float(group["lr"]))

        lr_info = " | ".join(
            f"{group.get('name', index)} LR: {group['lr']:.2e}"
            for index, group in enumerate(optimizer.param_groups)
        )

        print(
            f"\nEpoch [{local_epoch}/{epochs}] (global epoch {epoch})\n"
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"Accuracy: {train_metrics['accuracy']:.4f} | "
            f"Macro F1: {train_metrics['macro_f1']:.4f} | "
            f"Weighted F1: {train_metrics['weighted_f1']:.4f}\n"
            f"Val   | Loss: {val_metrics['loss']:.4f} | "
            f"Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Weighted F1: {val_metrics['weighted_f1']:.4f}\n"
            f"{lr_info}"
        )

        # Convert metrics to built-in floats
        val_metrics_to_save = {
            key: float(value) for key, value in val_metrics.items()
        }

        # Keep optimizer and scheduler state only in the last checkpoint
        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "val_metrics": val_metrics_to_save,
        }
        torch.save(last_checkpoint, last_path)

        # The best checkpoint stores weights only and is used for evaluation
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0

            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics_to_save,
            }
            torch.save(best_checkpoint, best_path)
            print(f"Best model updated. " f"Val Macro F1: {best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1

        # Save history after every epoch
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)

        # Early stopping
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print("\nEarly stopping")
            break

    print("\nTraining finished")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"History: {history_path}")

    return history


def load_best_model(
    model: nn.Module,
    checkpoint_path: Path,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
    use_channels_last: bool = False,
) -> nn.Module:
    """Load model weights and optional optimizer/scheduler state."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if use_channels_last:
        model = model.to(
            device=device,
            memory_format=torch.channels_last,
        )
    else:
        model = model.to(device)

    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    return model


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    device: torch.device = torch.device("cpu"),
    use_channels_last: bool = False,
) -> dict[str, float]:
    """Evaluate a model on a loader without updating weights."""
    if use_channels_last:
        model = model.to(
            device=device,
            memory_format=torch.channels_last,
        )
    else:
        model = model.to(device)

    return run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        num_classes=num_classes,
        optimizer=None,
        device=device,
        use_channels_last=use_channels_last,
    )
