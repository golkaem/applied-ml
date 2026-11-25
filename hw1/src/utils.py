"""Reproducibility and runtime helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(num_threads: int) -> None:
    """Configure the number of CPU threads used by PyTorch."""
    torch.set_num_threads(num_threads)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python for a DataLoader worker process."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
