"""Training-history visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_history(
    history: dict[str, list[float]],
    title_prefix: str,
) -> None:
    """Plot train/validation loss, accuracy, and macro F1 by epoch."""
    epochs = history["epoch"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()
    axes[0].grid()

    # Accuracy
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid()

    # Macro F1
    axes[2].plot(epochs, history["train_macro_f1"], label="Train")
    axes[2].plot(epochs, history["val_macro_f1"], label="Validation")
    axes[2].set_title("Macro F1-score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1-score")
    axes[2].legend()
    axes[2].grid()

    figure.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()
