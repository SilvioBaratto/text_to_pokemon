"""
Training utilities for Pokemon CVAE.

Provides KL annealing schedules and checkpoint management.
"""

import os
import sys
from typing import Any, Dict

import torch

sys.path.append('.')
import config


def get_kl_annealing_factor(epoch: int, warmup_epochs: int = config.KL_WARMUP_EPOCHS) -> float:
    """
    Compute KL annealing factor for current epoch.

    Uses cyclical annealing schedule with minimum beta to prevent posterior collapse.

    Args:
        epoch: Current epoch (1-indexed)
        warmup_epochs: Warmup epochs for linear mode

    Returns:
        Beta weight factor (0.5 to 1.0)
    """
    if config.KL_ANNEALING_TYPE == "cyclical":
        cycle_epochs = config.KL_CYCLE_EPOCHS
        cycle_position = (epoch - 1) % cycle_epochs
        ramp_epochs = cycle_epochs // 2

        if cycle_position < ramp_epochs:
            min_beta = 0.5
            progress = cycle_position / ramp_epochs
            return min_beta + (1.0 - min_beta) * progress
        else:
            return 1.0
    else:
        return 1.0 if epoch >= warmup_epochs else epoch / warmup_epochs


def get_base_dataset(dataset: Any) -> Any:
    """Unwrap Subset to get base PokemonCIFARDataset."""
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filepath)

    from tqdm import tqdm
    tqdm.write(f"Saved checkpoint: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> int:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: PyTorch optimizer
        device: Device to load on

    Returns:
        Next epoch to resume from
    """
    if not os.path.exists(filepath):
        print(f"ERROR: Checkpoint not found: {filepath}")
        return 1

    print(f"\nLoading checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    print("Checkpoint loaded successfully")
    print(f"Previous epoch: {checkpoint['epoch']}")
    print(f"Previous loss: {checkpoint['loss']:.2f}")
    print(f"Resuming from epoch: {start_epoch}\n")

    return start_epoch
