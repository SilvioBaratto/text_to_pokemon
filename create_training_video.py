"""
Create a video showing the progression of generated images across training epochs.
"""

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import numpy as np
import re
import io
import csv


def extract_epoch_number(filename):
    """Extract epoch number from filename like 'pokemon_samples_epoch_5.png'"""
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def read_csv_data(csv_path):
    """Read CSV file and return data as dict of lists."""
    data = {
        'Epoch': [],
        'Train_Loss': [],
        'Train_L1': [],
        'Train_KL': [],
        'Train_Perceptual': [],
        'Val_Loss': [],
        'Val_L1': [],
        'Val_KL': [],
        'Val_Perceptual': []
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['Epoch'].append(int(row['Epoch']))
            data['Train_Loss'].append(float(row['Train_Loss']))
            data['Train_L1'].append(float(row['Train_L1']))
            data['Train_KL'].append(float(row['Train_KL']))
            data['Train_Perceptual'].append(float(row['Train_Perceptual']))
            data['Val_Loss'].append(float(row['Val_Loss']))
            data['Val_L1'].append(float(row['Val_L1']))
            data['Val_KL'].append(float(row['Val_KL']))
            data['Val_Perceptual'].append(float(row['Val_Perceptual']))

    return data


def create_loss_plot(csv_path, current_epoch, max_epoch):
    """Create a plot showing training progress up to current epoch."""
    # Read CSV
    data = read_csv_data(csv_path)

    # Filter data up to current epoch
    indices = [i for i, e in enumerate(data['Epoch']) if e <= current_epoch]

    epochs = [data['Epoch'][i] for i in indices]
    train_loss = [data['Train_Loss'][i] for i in indices]
    val_loss = [data['Val_Loss'][i] for i in indices]
    train_l1 = [data['Train_L1'][i] for i in indices]
    val_l1 = [data['Val_L1'][i] for i in indices]
    train_kl = [data['Train_KL'][i] for i in indices]
    val_kl = [data['Val_KL'][i] for i in indices]
    train_perceptual = [data['Train_Perceptual'][i] for i in indices]
    val_perceptual = [data['Val_Perceptual'][i] for i in indices]

    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f'Training Progress (Epoch {current_epoch})', fontsize=14, fontweight='bold')

    # Plot 1: Total Loss (top-left)
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val', linewidth=2)
    axes[0, 0].set_ylabel('Total Loss', fontsize=10)
    axes[0, 0].legend(loc='upper right', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, max_epoch)

    # Plot 2: L1 Loss (top-right)
    axes[0, 1].plot(epochs, train_l1, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, val_l1, 'r-', label='Val', linewidth=2)
    axes[0, 1].set_ylabel('L1 Loss (MAE)', fontsize=10)
    axes[0, 1].legend(loc='upper right', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, max_epoch)

    # Plot 3: KL Divergence (bottom-left)
    axes[1, 0].plot(epochs, train_kl, 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, val_kl, 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=10)
    axes[1, 0].set_ylabel('KL Divergence', fontsize=10)
    axes[1, 0].legend(loc='upper right', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, max_epoch)

    # Plot 4: Perceptual Loss (bottom-right)
    axes[1, 1].plot(epochs, train_perceptual, 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, val_perceptual, 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=10)
    axes[1, 1].set_ylabel('Perceptual Loss', fontsize=10)
    axes[1, 1].legend(loc='upper right', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, max_epoch)

    plt.tight_layout()

    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close(fig)

    return plot_img


def create_training_video(
    images_dir="outputs",
    csv_path="training_losses.csv",
    output_path="training_progression.gif",
    fps=1,
    loop=0
):
    """
    Create an animated GIF showing training progression with loss curves.

    Args:
        images_dir: Directory containing epoch sample images
        csv_path: Path to training losses CSV file
        output_path: Path to save the output GIF
        fps: Frames per second (1 means each frame shows for 1 second)
        loop: Number of times to loop (0 = infinite)
    """
    # Find all sample images
    pattern = os.path.join(images_dir, "pokemon_samples_epoch_*.png")
    image_files = glob.glob(pattern)

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # Sort by epoch number
    image_files.sort(key=extract_epoch_number)

    print(f"Found {len(image_files)} epoch images")
    epochs = [extract_epoch_number(f) for f in image_files]
    print(f"Epochs: {epochs}")

    max_epoch = max(epochs)

    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        print("Creating video without loss plots")
        csv_path = None

    # Create animated GIF
    print(f"\nCreating animated GIF with loss curves: {output_path}")
    print(f"FPS: {fps}, Duration per frame: {1000/fps:.0f}ms")

    # Create combined frames
    combined_images = []
    for img_file, epoch in zip(image_files, epochs):
        # Load generated samples image
        samples_img = Image.open(img_file)
        # Create loss plot if CSV available
        if csv_path:
            plot_img = create_loss_plot(csv_path, epoch, max_epoch)
        else:
            # Create blank plot placeholder (matching 2x2 grid dimensions)
            plot_img = Image.new('RGB', (800, 800), color='white')

        # Resize images to same height
        target_height = 800
        samples_ratio = samples_img.width / samples_img.height
        samples_new_width = int(target_height * samples_ratio)
        samples_resized = samples_img.resize((samples_new_width, target_height), Image.Resampling.LANCZOS)

        # Combine images side by side
        total_width = samples_resized.width + plot_img.width
        combined = Image.new('RGB', (total_width, target_height), color='white')
        combined.paste(samples_resized, (0, 0))
        combined.paste(plot_img, (samples_resized.width, 0))

        # Add epoch text at the bottom of samples
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        text = f"Epoch {epoch}"
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (samples_resized.width - text_width) // 2
        y = target_height - text_height - 30

        padding = 10
        draw.rectangle(
            [(x - padding, y - padding), (x + text_width + padding, y + text_height + padding)],
            fill='black'
        )
        draw.text((x, y), text, fill='white', font=font)

        combined_images.append(combined)

    # Save as GIF
    duration_ms = int(1000 / fps)
    combined_images[0].save(
        output_path,
        save_all=True,
        append_images=combined_images[1:],
        duration=duration_ms,
        loop=loop
    )

    print(f"✓ Saved animated GIF: {output_path}")
    print(f"  Total frames: {len(combined_images)}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def create_static_comparison(
    images_dir="outputs",
    output_path="training_comparison.png"
):
    """
    Create a static side-by-side comparison of all epochs.

    Args:
        images_dir: Directory containing epoch sample images
        output_path: Path to save the comparison image
    """
    # Find all sample images
    pattern = os.path.join(images_dir, "pokemon_samples_epoch_*.png")
    image_files = glob.glob(pattern)

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # Sort by epoch number
    image_files.sort(key=extract_epoch_number)
    epochs = [extract_epoch_number(os.path.basename(f)) for f in image_files]

    print(f"\nCreating static comparison: {output_path}")

    # Load images
    images = [Image.open(f) for f in image_files]

    # Create grid layout
    n_images = len(images)
    cols = min(3, n_images)  # Max 3 columns
    rows = (n_images + cols - 1) // cols  # Ceiling division

    fig = plt.figure(figsize=(6 * cols, 6 * rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.3, wspace=0.1)

    for idx, (img, epoch) in enumerate(zip(images, epochs)):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(f'Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison image: {output_path}")
    print(f"  Grid: {rows}x{cols}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    print("="*80)
    print("Training Progression Visualization")
    print("="*80)

    # Create animated GIF with loss curves (1 fps = 1 second per frame)
    create_training_video(
        images_dir="outputs",
        csv_path="training_losses.csv",
        output_path="training_progression.gif",
        fps=1,
        loop=0  # Infinite loop
    )

    # Create static comparison
    create_static_comparison(
        images_dir="outputs",
        output_path="training_comparison.png"
    )

    print("\n" + "="*80)
    print("Done!")
    print("="*80)
