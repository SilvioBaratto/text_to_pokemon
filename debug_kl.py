"""
Debug script to understand KL divergence values.
Run this to diagnose the KL=5000 issue.
"""

import sys
import torch
sys.path.append('.')
import config
from src.model.vae import loss_function

print("=" * 80)
print("KL DIVERGENCE DIAGNOSTIC SCRIPT")
print("=" * 80)
print()

# Configuration
batch_size = 128
latent_dim = config.LATENT_DIM  # Should be 64
input_dim = config.INPUT_DIM    # Should be 12288

print(f"Configuration:")
print(f"  LATENT_DIM: {config.LATENT_DIM}")
print(f"  INPUT_DIM: {config.INPUT_DIM}")
print(f"  KL_BASE_WEIGHT: {config.KL_BASE_WEIGHT}")
print(f"  USE_NORMALIZED_KL: {config.USE_NORMALIZED_KL}")
print(f"  Batch size: {batch_size}")
print()

# Simulate realistic VAE outputs
# In early training, mu and logvar should be close to 0
torch.manual_seed(42)
mu = torch.randn(batch_size, latent_dim) * 0.1      # Small values
logvar = torch.randn(batch_size, latent_dim) * 0.1  # Small values

print(f"Input statistics:")
print(f"  mu mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
print(f"  logvar mean: {logvar.mean().item():.4f}, std: {logvar.std().item():.4f}")
print()

# Manual KL calculation - step by step
print("=" * 80)
print("STEP-BY-STEP KL CALCULATION")
print("=" * 80)

# Step 1: Calculate KL terms
kl_term = 1 + logvar - mu.pow(2) - logvar.exp()
print(f"Step 1 - KL term (1 + logvar - mu^2 - exp(logvar)):")
print(f"  Shape: {kl_term.shape}")
print(f"  Mean: {kl_term.mean().item():.4f}")
print(f"  Min: {kl_term.min().item():.4f}, Max: {kl_term.max().item():.4f}")
print()

# Step 2: Sum over latent dimensions
kl_per_sample = -0.5 * torch.sum(kl_term, dim=1)
print(f"Step 2 - KL per sample (sum over latent dims):")
print(f"  Shape: {kl_per_sample.shape}")
print(f"  Mean: {kl_per_sample.mean().item():.4f}")
print(f"  Min: {kl_per_sample.min().item():.4f}, Max: {kl_per_sample.max().item():.4f}")
print(f"  First 5 samples: {kl_per_sample[:5].tolist()}")
print()

# Step 3: Mean over batch
kl_loss = torch.mean(kl_per_sample)
print(f"Step 3 - KL loss (mean over batch):")
print(f"  Value: {kl_loss.item():.4f}")
print()

# Step 4: Apply weight
if config.USE_NORMALIZED_KL:
    kl_weight = config.KL_BASE_WEIGHT * (config.LATENT_DIM / config.INPUT_DIM)
else:
    kl_weight = config.KL_BASE_WEIGHT

print(f"Step 4 - Apply KL weight:")
print(f"  KL_BASE_WEIGHT: {config.KL_BASE_WEIGHT}")
print(f"  Normalized factor: {config.LATENT_DIM / config.INPUT_DIM:.6f}")
print(f"  Final kl_weight: {kl_weight:.6f}")
print()

beta = 0.08  # Epoch 8 with warmup=100
weighted_kl = beta * kl_weight * kl_loss
print(f"Step 5 - Apply beta annealing (β={beta}):")
print(f"  Weighted KL contribution to loss: {weighted_kl.item():.6f}")
print()

print("=" * 80)
print("USING ACTUAL LOSS FUNCTION")
print("=" * 80)

# Create fake reconstruction
recon_x = torch.randn(batch_size, input_dim) * 0.5
x = torch.randn(batch_size, input_dim) * 0.5

# Call actual loss function
total_loss, l1_loss, kl_loss_fn, perceptual_loss = loss_function(
    recon_x, x, mu, logvar, beta=beta, perceptual_loss_model=None
)

print(f"Loss function returns:")
print(f"  L1 loss: {l1_loss.item():.4f}")
print(f"  KL loss: {kl_loss_fn.item():.4f}")
print(f"  Perceptual loss: {perceptual_loss.item():.4f}")
print(f"  Total loss: {total_loss.item():.4f}")
print()

print("=" * 80)
print("EXPECTED vs ACTUAL")
print("=" * 80)
print(f"Expected KL (manual): {kl_loss.item():.4f}")
print(f"Actual KL (function): {kl_loss_fn.item():.4f}")
print(f"Match: {'✓ YES' if abs(kl_loss.item() - kl_loss_fn.item()) < 0.01 else '✗ NO'}")
print()

if kl_loss_fn.item() > 100:
    print("⚠️  WARNING: KL loss is > 100, which is very high!")
    print("   This indicates the loss function is NOT using mean reduction properly.")
elif kl_loss_fn.item() > 50:
    print("⚠️  WARNING: KL loss is > 50, which is higher than expected.")
    print("   Expected range: 0.5-10 for early training")
elif kl_loss_fn.item() < 0.01:
    print("⚠️  WARNING: KL loss is < 0.01, which is too low.")
    print("   The KL weight might be too small.")
else:
    print("✓ KL loss is in reasonable range (0.01-50)")

print()
print("=" * 80)
print("If you're seeing KL=5000 in training, possible causes:")
print("  1. Python module caching - restart Python/Jupyter kernel")
print("  2. Loading old checkpoint with old loss calculation")
print("  3. Looking at wrong metric (accumulated loss vs per-batch loss)")
print("  4. Training script not using updated vae.py")
print("=" * 80)
