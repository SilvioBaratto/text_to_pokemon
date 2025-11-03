"""
Pokemon CVAE Training Script

Trains a Conditional Variational Autoencoder for text-to-image Pokemon generation
with dual conditioning (CLIP text embeddings + categorical attributes).
"""

import csv
import os
import sys
import warnings
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.append('.')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')

import config
from src.model.vae import CVAE, PerceptualLoss, loss_function
from src.utils.cleanup import register_cleanup_handlers, set_cleanup_references
from src.utils.conditioning import extract_conditioning_from_metadata
from src.utils.data_loader import get_pokemon_dataloaders
from src.utils.debug_utils import save_dataset_images_for_debugging
from src.utils.training_utils import (
    get_base_dataset,
    get_kl_annealing_factor,
    load_checkpoint,
    save_checkpoint
)
from src.utils.visualization import plot_training_curves, visualize_samples


def train_epoch(
    model: Union[CVAE, Any],
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float,
    scaler: Optional[GradScaler] = None,
    perceptual_loss_model: Optional[PerceptualLoss] = None,
    amp_device: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """
    Execute one training epoch with mixed precision support.

    Args:
        model: Conditional VAE model (may be compiled)
        train_loader: Training data loader
        optimizer: Parameter optimizer
        device: Compute device
        beta: KL divergence annealing weight
        scaler: Gradient scaler for mixed precision training
        perceptual_loss_model: LPIPS perceptual loss model
        amp_device: Device type for automatic mixed precision

    Returns:
        Tuple of average losses: (total, L1, KL, perceptual)
    """
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    total_perceptual = 0.0

    base_dataset = get_base_dataset(train_loader.dataset)
    batch_pbar = tqdm(train_loader, desc="  Training", leave=False, unit="batch")

    for _, (images, labels, _, prompt_indices) in enumerate(batch_pbar):
        non_blocking = config.NON_BLOCKING_TRANSFER if device.type == 'cuda' else False
        images = images.to(device, non_blocking=non_blocking)
        prompt_indices = prompt_indices.tolist()

        metadata_list = [base_dataset.get_metadata_by_label(label.item()) for label in labels]

        (text_emb, type1, type2, primary_color, secondary_color,
         shape, size, evolution_stage, habitat,
         legendary, mythical) = extract_conditioning_from_metadata(
            metadata_list, prompt_indices, device
        )

        condition = model.prepare_condition(
            text_emb, type1, type2, primary_color, secondary_color,
            shape, size, evolution_stage, habitat,
            legendary, mythical
        )

        x = images.view(images.size(0), -1)

        if scaler is not None and amp_device is not None:
            with autocast(amp_device):
                recon, mu, logvar = model(x, condition)
                loss, l1_loss, kl_loss, perceptual_loss = loss_function(
                    recon, x, mu, logvar, beta=beta,
                    perceptual_loss_model=perceptual_loss_model
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            recon, mu, logvar = model(x, condition)
            loss, l1_loss, kl_loss, perceptual_loss = loss_function(
                recon, x, mu, logvar, beta=beta,
                perceptual_loss_model=perceptual_loss_model
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_kl += kl_loss.item()
        total_perceptual += perceptual_loss.item()

        batch_pbar.set_postfix({
            'loss': f'{loss.item():.2f}',
            'kl': f'{kl_loss.item():.2f}'
        })

    batch_pbar.close()

    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_l1 = total_l1 / n_batches
    avg_kl = total_kl / n_batches
    avg_perceptual = total_perceptual / n_batches

    # Debug: print to verify averaging
    if not hasattr(train_epoch, '_debug_printed'):
        print(f"\n[DEBUG train_epoch] n_batches={n_batches}, total_kl={total_kl:.2f}, avg_kl={avg_kl:.4f}")
        train_epoch._debug_printed = True

    return (avg_loss, avg_l1, avg_kl, avg_perceptual)


def evaluate(
    model: Union[CVAE, Any],
    test_loader,
    device: torch.device,
    perceptual_loss_model: Optional[PerceptualLoss] = None
) -> Tuple[float, float, float, float]:
    """
    Evaluate model on validation or test set.

    Args:
        model: Conditional VAE model (may be compiled)
        test_loader: Evaluation data loader
        device: Compute device
        perceptual_loss_model: LPIPS perceptual loss model

    Returns:
        Tuple of average losses: (total, L1, KL, perceptual)
    """
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    total_perceptual = 0.0

    base_dataset = get_base_dataset(test_loader.dataset)
    batch_pbar = tqdm(test_loader, desc="  Validating", leave=False, unit="batch")

    with torch.no_grad():
        for _, (images, labels, _, prompt_indices) in enumerate(batch_pbar):
            non_blocking = config.NON_BLOCKING_TRANSFER if device.type == 'cuda' else False
            images = images.to(device, non_blocking=non_blocking)
            prompt_indices = prompt_indices.tolist()

            metadata_list = [base_dataset.get_metadata_by_label(label.item()) for label in labels]

            (text_emb, type1, type2, primary_color, secondary_color,
             shape, size, evolution_stage, habitat,
             legendary, mythical) = extract_conditioning_from_metadata(
                metadata_list, prompt_indices, device
            )

            condition = model.prepare_condition(
                text_emb, type1, type2, primary_color, secondary_color,
                shape, size, evolution_stage, habitat,
                legendary, mythical
            )

            x = images.view(images.size(0), -1)

            recon, mu, logvar = model(x, condition)
            loss, l1_loss, kl_loss, perceptual_loss = loss_function(
                recon, x, mu, logvar, beta=1.0,
                perceptual_loss_model=perceptual_loss_model
            )

            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_kl += kl_loss.item()
            total_perceptual += perceptual_loss.item()

            batch_pbar.set_postfix({
                'loss': f'{loss.item():.2f}',
                'kl': f'{kl_loss.item():.2f}'
            })

    batch_pbar.close()

    n_batches = len(test_loader)
    avg_loss = total_loss / n_batches
    avg_l1 = total_l1 / n_batches
    avg_kl = total_kl / n_batches
    avg_perceptual = total_perceptual / n_batches

    # Debug: print to verify averaging
    if not hasattr(evaluate, '_debug_printed'):
        print(f"\n[DEBUG evaluate] n_batches={n_batches}, total_kl={total_kl:.2f}, avg_kl={avg_kl:.4f}")
        evaluate._debug_printed = True

    return (avg_loss, avg_l1, avg_kl, avg_perceptual)


def main() -> None:
    """Main training loop with CLI argument support."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Pokemon CVAE with dual conditioning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset-dir', type=str, default='pokemon-cifar',
                        help='Path to Pokemon dataset directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume training')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory for saving checkpoints')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for saving outputs')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--debug-samples', type=int, default=None,
                        help='Limit total samples for debugging')
    parser.add_argument('--save-debug-images', action='store_true',
                        help='Save dataset images with metadata')
    parser.add_argument('--debug-images-dir', type=str, default='debug_images',
                        help='Debug images output directory')
    parser.add_argument('--use-preprocessing', action='store_true',
                        help='Enable image preprocessing pipeline')

    args = parser.parse_args()

    print("="*80)
    print("COMMAND LINE ARGUMENTS")
    print("="*80)
    for arg, value in vars(args).items():
        print(f"  {arg:20s} = {value}")
    print("="*80)
    print()

    checkpoint_dir = args.checkpoint_dir or config.CHECKPOINT_DIR
    output_dir = args.output_dir or config.OUTPUT_DIR
    num_epochs = args.epochs or config.NUM_EPOCHS
    batch_size = args.batch_size or config.BATCH_SIZE

    print("="*80)
    print("POKEMON CVAE TRAINING")
    print("="*80)
    print("Automatic cleanup enabled (Ctrl+C safe)")
    print("="*80)

    register_cleanup_handlers()

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")

    print(f"Dataset:         {args.dataset_dir}")
    print(f"Checkpoint dir:  {checkpoint_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Epochs:          {num_epochs}")

    test_interval = 1 if args.debug_samples else config.TEST_INTERVAL
    save_interval = 5 if args.debug_samples else config.SAVE_INTERVAL

    if args.debug_samples:
        print(f"Debug mode:      {args.debug_samples} samples")
        print(f"                 Test interval: {test_interval}, Save interval: {save_interval}")
    print("="*80)

    device = config.DEVICE
    print(f"\nDevice: {device}")

    print(f"\nLoading {args.dataset_dir} dataset...")

    train_loader, val_loader, test_loader = get_pokemon_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=batch_size,
        val_split=0.2,
        num_workers=0,
        debug_samples=args.debug_samples,
        use_preprocessing=args.use_preprocessing
    )

    if args.use_preprocessing:
        print("✓ Image preprocessing enabled (background removal + contrast + bilateral filter)")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")  # type: ignore

    # Save debug images if requested
    if args.save_debug_images:
        print(f"\n{'='*80}")
        print("DEBUG MODE: Saving all dataset images with metadata")
        print(f"Output directory: {args.debug_images_dir}")
        print(f"{'='*80}")

        save_dataset_images_for_debugging(train_loader, args.debug_images_dir, 'train', device)
        save_dataset_images_for_debugging(val_loader, args.debug_images_dir, 'val', device)
        save_dataset_images_for_debugging(test_loader, args.debug_images_dir, 'test', device)

        print(f"\n{'='*80}")
        print("Dataset images saved successfully")
        print(f"{'='*80}")
        print("\nInspect images before continuing.")
        print("Press Enter to start training, or Ctrl+C to exit...")
        input()

    print("\n" + "=" * 80)
    print("CUDA OPTIMIZATIONS")
    print("=" * 80)

    if config.ENABLE_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled (A100 optimization)")

    print("\nInitializing CVAE model...")
    model: Union[CVAE, Any] = CVAE(
        input_dim=config.INPUT_DIM,
        condition_dim=config.TOTAL_CONDITION_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM
    ).to(device)

    if config.USE_CHANNELS_LAST and device.type == 'cuda':
        model = model.to(device, memory_format=torch.channels_last)  # type: ignore[call-overload]
        print("Using channels_last memory format")

    if config.USE_TORCH_COMPILE and hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model (1-2 minutes)...")
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled successfully")
    elif config.USE_TORCH_COMPILE and device.type != 'cuda':
        print("Warning: torch.compile() disabled on MPS/CPU")
    elif config.USE_TORCH_COMPILE:
        print("Warning: torch.compile() requires PyTorch 2.0+")

    print("=" * 80)

    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    embedder_params = sum(p.numel() for p in model.category_embedder.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nEncoder:          {encoder_params:>12,} parameters")
    print(f"Decoder:          {decoder_params:>12,} parameters")
    print(f"Embedder:         {embedder_params:>12,} parameters")
    print("-" * 70)
    print(f"Total:            {total_params:>12,} parameters")

    print(f"\nInput: {config.IMAGE_SIZE}×{config.IMAGE_SIZE} RGB → Latent: {config.LATENT_DIM} dims")
    print(f"Conditioning: {config.TOTAL_CONDITION_DIM} dims (CLIP + categorical)")
    print("=" * 70)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    set_cleanup_references(model, optimizer, device)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=config.MIN_LEARNING_RATE
    )

    start_epoch = 1

    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)
        if start_epoch == 1:
            return

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config.MIN_LEARNING_RATE,
            last_epoch=start_epoch - 1
        )

    perceptual_loss_model = None
    if config.USE_PERCEPTUAL_LOSS:
        print("\nInitializing perceptual loss (LPIPS)...")
        perceptual_loss_model = PerceptualLoss(net='alex').to(device)
        print(f"Perceptual loss weight: {config.PERCEPTUAL_LOSS_WEIGHT}")

    if config.USE_MIXED_PRECISION and torch.cuda.is_available():
        scaler = GradScaler('cuda')
        amp_device = 'cuda'
        print("Mixed precision training enabled (CUDA)")
    elif config.USE_MIXED_PRECISION and device.type == 'mps':
        print("Warning: AMP not supported on MPS")
        scaler = None
        amp_device = None
    else:
        scaler = None
        amp_device = None

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs} (start: {start_epoch})")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  LR scheduler: Cosine annealing")
    print(f"  KL annealing: {config.KL_ANNEALING_TYPE}")
    if config.KL_ANNEALING_TYPE == 'cyclical':
        print(f"  KL cycle: {config.KL_CYCLE_EPOCHS} epochs")
    print(f"  Latent dims: {config.LATENT_DIM}")
    print(f"  Conditioning dims: {config.TOTAL_CONDITION_DIM}")

    best_val_loss = float('inf')

    train_losses = {'total': [], 'l1': [], 'kl': [], 'perceptual': []}
    val_losses = {'total': [], 'l1': [], 'kl': [], 'perceptual': []}
    test_losses = {'total': [], 'l1': [], 'kl': [], 'perceptual': []}

    csv_path = "training_losses.csv"

    # If resuming, append to existing CSV; otherwise create new one
    if args.resume and os.path.exists(csv_path):
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        print(f"Appending to existing CSV: {csv_path}")
    else:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Train_Loss', 'Train_L1', 'Train_KL', 'Train_Perceptual',
                             'Val_Loss', 'Val_L1', 'Val_KL', 'Val_Perceptual',
                             'Test_Loss', 'Test_L1', 'Test_KL', 'Test_Perceptual', 'Beta', 'LR'])
        print(f"Creating new CSV: {csv_path}")
    print("")

    # Epoch-level progress bar
    epoch_pbar = tqdm(range(start_epoch, num_epochs + 1), desc="Epochs", unit="epoch", position=0)

    for epoch in epoch_pbar:
        beta = get_kl_annealing_factor(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # Update epoch progress bar
        epoch_pbar.set_postfix({'β': f'{beta:.3f}', 'LR': f'{current_lr:.6f}'})

        train_loss, train_l1, train_kl, train_perceptual = train_epoch(
            model, train_loader, optimizer, device, beta, scaler, perceptual_loss_model, amp_device
        )

        val_loss, val_l1, val_kl, val_perceptual = evaluate(model, val_loader, device, perceptual_loss_model)

        # Step scheduler
        scheduler.step()

        train_losses['total'].append(train_loss)
        train_losses['l1'].append(train_l1)
        train_losses['kl'].append(train_kl)
        train_losses['perceptual'].append(train_perceptual)
        val_losses['total'].append(val_loss)
        val_losses['l1'].append(val_l1)
        val_losses['kl'].append(val_kl)
        val_losses['perceptual'].append(val_perceptual)

        # Print metrics cleanly below progress bar
        tqdm.write(f"\nTrain - Loss: {train_loss:.2f}, L1: {train_l1:.2f}, KL: {train_kl:.2f}, Perceptual: {train_perceptual:.2f}")
        tqdm.write(f"Val   - Loss: {val_loss:.2f}, L1: {val_l1:.2f}, KL: {val_kl:.2f}, Perceptual: {val_perceptual:.2f}")

        if epoch % test_interval == 0 or epoch == num_epochs:
            test_loss, test_l1, test_kl, test_perceptual = evaluate(model, test_loader, device, perceptual_loss_model)
            tqdm.write(f"Test  - Loss: {test_loss:.2f}, L1: {test_l1:.2f}, KL: {test_kl:.2f}, Perceptual: {test_perceptual:.2f}")
        else:
            test_loss, test_l1, test_kl, test_perceptual = None, None, None, None

        test_losses['total'].append(test_loss if test_loss is not None else float('nan'))
        test_losses['l1'].append(test_l1 if test_l1 is not None else float('nan'))
        test_losses['kl'].append(test_kl if test_kl is not None else float('nan'))
        test_losses['perceptual'].append(test_perceptual if test_perceptual is not None else float('nan'))

        csv_writer.writerow([epoch, train_loss, train_l1, train_kl, train_perceptual,
                            val_loss, val_l1, val_kl, val_perceptual,
                            test_loss if test_loss is not None else '',
                            test_l1 if test_l1 is not None else '',
                            test_kl if test_kl is not None else '',
                            test_perceptual if test_perceptual is not None else '',
                            beta, current_lr])
        csv_file.flush()

        if epoch % save_interval == 0:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'model_pokemon_epoch_{epoch}.pt'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
            visualize_samples(model, device, epoch, test_loader)
            plot_training_curves(train_losses, val_losses, save_path="training_curves.png")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model_pokemon.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)

    epoch_pbar.close()
    csv_file.close()
    print(f"\nLosses saved to: {csv_path}")

    plot_training_curves(train_losses, val_losses, save_path="training_curves.png")

    print(f"\n{'='*80}")
    print("Final Evaluation")
    print(f"{'='*80}")
    final_test_loss, final_test_l1, final_test_kl, final_test_perceptual = evaluate(model, test_loader, device, perceptual_loss_model)
    print(f"Test - Loss: {final_test_loss:.2f}, L1: {final_test_l1:.2f}, KL: {final_test_kl:.2f}, Perceptual: {final_test_perceptual:.2f}")

    print(f"\n{'='*80}")
    print("Training complete")
    print(f"Best val loss: {best_val_loss:.2f}")
    print(f"Final test loss: {final_test_loss:.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
