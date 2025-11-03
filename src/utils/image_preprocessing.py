"""
Image preprocessing utilities for Pokemon dataset.

Provides background removal, contrast enhancement, and edge-preserving filters.
"""

from typing import Dict, Tuple

import cv2
import numpy as np


def remove_white_background(
    image: np.ndarray,
    threshold: int = 240,
    replace_with: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Remove white background from images.

    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        threshold: Pixel threshold for white detection
        replace_with: RGB tuple for replacement

    Returns:
        Image with background removed
    """
    mask = np.all(image >= threshold, axis=-1)
    result = image.copy()
    result[mask] = replace_with
    return result


def enhance_edges_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: int = 75,
    sigma_space: int = 75
) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving smoothing.

    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Filtered image
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement.

    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        clip_limit: Contrast limiting threshold
        tile_size: Grid size for histogram equalization

    Returns:
        Contrast-enhanced image
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced


def sharpen_image(
    image: np.ndarray,
    amount: float = 1.0
) -> np.ndarray:
    """
    Apply unsharp masking for edge sharpening.

    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        amount: Sharpening strength

    Returns:
        Sharpened image
    """
    image_float = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image_float, (0, 0), 3)
    sharpened = cv2.addWeighted(image_float, 1.0 + amount, blurred, -amount, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


def preprocess_pokemon_image(
    image: np.ndarray,
    remove_background: bool = True,
    enhance_contrast: bool = True,
    sharpen: bool = False,
    bilateral_filter: bool = False
) -> np.ndarray:
    """
    Full preprocessing pipeline for images.

    Args:
        image: RGB image (H, W, 3) in range [0, 255] or [0, 1]
        remove_background: Remove white background
        enhance_contrast: Apply CLAHE enhancement
        sharpen: Apply unsharp masking
        bilateral_filter: Apply edge-preserving filter

    Returns:
        Preprocessed image in same range as input
    """
    is_normalized = image.max() <= 1.0

    if is_normalized:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    result = image_uint8.copy()

    if remove_background:
        result = remove_white_background(result, threshold=240, replace_with=(0, 0, 0))

    if bilateral_filter:
        result = enhance_edges_bilateral(result, d=9, sigma_color=75, sigma_space=75)

    if enhance_contrast:
        result = enhance_contrast_clahe(result, clip_limit=2.0, tile_size=(8, 8))

    if sharpen:
        result = sharpen_image(result, amount=0.5)

    if is_normalized:
        result = result.astype(np.float32) / 255.0

    return result


def visualize_preprocessing_comparison(
    image: np.ndarray,
    save_path: str = "preprocessing_comparison.png"
) -> None:
    """
    Visualize different preprocessing options.

    Args:
        image: RGB image (H, W, 3)
        save_path: Path to save comparison plot
    """
    import matplotlib.pyplot as plt

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    variants: Dict[str, np.ndarray] = {
        "Original": image,
        "Background Removed": remove_white_background(image),
        "+ Contrast": enhance_contrast_clahe(remove_white_background(image)),
        "+ Bilateral Filter": enhance_edges_bilateral(
            enhance_contrast_clahe(remove_white_background(image))
        ),
        "+ Sharpened": sharpen_image(
            enhance_contrast_clahe(remove_white_background(image))
        ),
        "All": preprocess_pokemon_image(
            image,
            remove_background=True,
            enhance_contrast=True,
            sharpen=False,
            bilateral_filter=True
        )
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (title, img) in enumerate(variants.items()):
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved preprocessing comparison: {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.utils.data_loader import get_pokemon_dataloaders

    print("Loading sample image...")
    train_loader, _, _ = get_pokemon_dataloaders(
        dataset_dir='pokemon-cifar',
        batch_size=1,
        debug_samples=30
    )

    images, _, _, _ = next(iter(train_loader))
    sample_image = images[0].permute(1, 2, 0).numpy()

    print("Generating preprocessing comparison...")
    visualize_preprocessing_comparison(sample_image)

    print("\nPreprocessing options:")
    print("1. Background removal: Eliminates white pixels")
    print("2. CLAHE: Enhances contrast adaptively")
    print("3. Bilateral filter: Preserves edges while smoothing")
    print("4. Sharpening: Emphasizes edges")
