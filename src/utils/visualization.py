"""
Visualization utilities for Pokemon CVAE training.

Generates sample images and training curves for model evaluation.
"""

import os
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
import config

from src.utils.training_utils import get_base_dataset
from src.utils.conditioning import extract_conditioning_from_metadata


def visualize_samples(
    model: torch.nn.Module,
    device: torch.device,
    epoch: int,
    test_loader: Any,
    output_dir: str = "outputs"
) -> None:
    """
    Generate and save sample Pokemon images.

    Args:
        model: Pokemon CVAE model
        device: Device to run on
        epoch: Current epoch number
        test_loader: Test data loader
        output_dir: Output directory
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    base_dataset = get_base_dataset(test_loader.dataset)

    dataset_size = len(base_dataset)
    stride = max(1, dataset_size // 8)

    diverse_indices = [i * stride for i in range(8)]
    diverse_indices = [idx for idx in diverse_indices if idx < dataset_size]

    test_images = []
    test_prompt_indices = []
    metadata_list = []

    for idx in diverse_indices[:8]:
        img, label, _, prompt_idx = base_dataset[idx]
        test_images.append(img)
        test_prompt_indices.append(prompt_idx)
        metadata_list.append(base_dataset.get_metadata_by_label(label))

    test_images = torch.stack(test_images).to(device)

    (text_emb, type1, type2, primary_color, secondary_color,
     shape, size, evolution_stage, habitat,
     legendary, mythical) = extract_conditioning_from_metadata(
        metadata_list, test_prompt_indices, device
    )

    condition = model.prepare_condition(
        text_emb, type1, type2, primary_color, secondary_color,
        shape, size, evolution_stage, habitat,
        legendary, mythical
    )

    with torch.no_grad():
        generated_flat = model.generate(condition, num_samples=1)
        generated = generated_flat.view(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle(f'Pokemon Generation - Epoch {epoch}', fontsize=14)

    for i in range(min(8, test_images.size(0))):
        orig = test_images[i].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1.0) / 2.0
        orig = np.clip(orig, 0, 1)
        axes[0, i].imshow(orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        gen = generated[i].cpu().permute(1, 2, 0).numpy()
        gen = (gen + 1.0) / 2.0
        gen = np.clip(gen, 0, 1)
        axes[1, i].imshow(gen)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Generated', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'pokemon_samples_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    tqdm.write(f"Generated samples: {save_path}")


def plot_training_curves(
    train_losses: Dict[str, List[float]],
    test_losses: Dict[str, List[float]],
    save_path: str = "training_curves.png"
) -> None:
    """
    Plot and save training curves.

    Args:
        train_losses: Dict with keys ['total', 'l1', 'kl', 'perceptual']
        test_losses: Dict with keys ['total', 'l1', 'kl', 'perceptual']
        save_path: Path to save plot
    """
    epochs = range(1, len(train_losses['total']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(epochs, train_losses['total'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, test_losses['total'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_losses['l1'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, test_losses['l1'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('L1 Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, train_losses['kl'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, test_losses['kl'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('KL Divergence', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, train_losses['perceptual'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, test_losses['perceptual'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Perceptual Loss', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    tqdm.write(f"Training curves saved: {save_path}")
