"""Quick test to verify loss function returns reasonable values."""

import sys
import torch
sys.path.append('.')
import config
from src.model.vae import loss_function

# Simulate a batch
batch_size = 128
latent_dim = config.LATENT_DIM  # 64
input_dim = config.INPUT_DIM    # 12288

# Create fake data
recon_x = torch.randn(batch_size, input_dim) * 0.5  # Reconstructed images
x = torch.randn(batch_size, input_dim) * 0.5        # Original images
mu = torch.randn(batch_size, latent_dim) * 0.5      # Mean of latent distribution
logvar = torch.randn(batch_size, latent_dim) * 0.5  # Log variance

# Compute losses
beta = 0.02  # First epoch with warmup
total_loss, l1_loss, kl_loss, perceptual_loss = loss_function(
    recon_x, x, mu, logvar, beta=beta, perceptual_loss_model=None
)

print("=" * 70)
print("LOSS FUNCTION TEST (with mean reduction)")
print("=" * 70)
print(f"Batch size: {batch_size}")
print(f"Latent dim: {latent_dim}")
print(f"Input dim:  {input_dim}")
print(f"Beta:       {beta}")
print()
print("Loss Values (per-sample averages):")
print("-" * 70)
print(f"L1 Loss:         {l1_loss.item():.4f}")
print(f"KL Loss:         {kl_loss.item():.4f}  (sum over {latent_dim} latent dims, mean over batch)")
print(f"Perceptual Loss: {perceptual_loss.item():.4f}")
print(f"Total Loss:      {total_loss.item():.4f}")
print("=" * 70)
print()
print("Expected ranges with mean reduction:")
print("  L1 loss:         0.2 - 0.5 (mean absolute difference per pixel)")
print("  KL loss:         5 - 50 (sum of ~64 latent dims, averaged over batch)")
print("  Total loss:      0.3 - 1.0 in first epoch")
print()
print("✓ If KL loss is in range 5-50, the fix is working correctly!")
print("✗ If KL loss is > 1000, there's still a problem.")
print("=" * 70)
