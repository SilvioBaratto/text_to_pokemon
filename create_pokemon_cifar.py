"""
Pokemon CIFAR Dataset Builder

This module constructs a balanced Pokemon dataset in CIFAR format using stratified
sampling techniques. It ensures consistent distributions of Pokemon attributes
(type, color, body shape) across training, validation, and test splits.

The balancing approach prevents distribution shifts between splits, which is
critical for reliable model evaluation.
"""

import json
import multiprocessing
import pickle
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def _load_single_image(
    img_path: Path,
    metadata_path: Path,
    label_idx: int,
    pokemon_name: str,
    image_size: int = 32
) -> Dict:
    """
    Load and process a single Pokemon image with its metadata.

    This function is designed for parallel processing and handles image loading,
    validation, and conversion to CIFAR format.

    Args:
        img_path: Path to the image file
        metadata_path: Path to the JSON metadata file
        label_idx: Numeric label index for this Pokemon species
        pokemon_name: Name of the Pokemon species
        image_size: Expected image dimensions (32 or 64)

    Returns:
        Dictionary containing processed image data and metadata, or error information
    """
    try:
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)

            if img_array.shape == (image_size, image_size, 3):
                img_flat = img_array.transpose(2, 0, 1).reshape(-1)
                stratify_key = _create_stratification_key(metadata)

                return {
                    'data': img_flat,
                    'label': label_idx,
                    'filename': f"{pokemon_name}/{img_path.name}",
                    'metadata': metadata,
                    'stratify_key': stratify_key,
                    'success': True
                }
            else:
                return {
                    'success': False,
                    'error': f'invalid_shape: expected ({image_size}, {image_size}, 3), got {img_array.shape}'
                }
    except Exception as e:
        return {'success': False, 'error': str(e), 'path': str(img_path)}


def load_pokemon_dataset(
    images_dir: str,
    require_metadata: bool = True,
    max_workers: Optional[int] = None,
    image_size: int = 32
) -> Dict:
    """
    Load Pokemon image dataset with metadata using parallel processing.

    This function discovers all Pokemon species folders, loads images with their
    associated metadata files, and prepares them for CIFAR-format conversion.

    Args:
        images_dir: Path to directory containing Pokemon species subdirectories
        require_metadata: Skip images without accompanying JSON metadata
        max_workers: Number of parallel worker processes (defaults to CPU count)
        image_size: Expected image dimensions in pixels (32 or 64)

    Returns:
        Dictionary containing:
            - data: numpy array of flattened images
            - labels: list of numeric species labels
            - filenames: list of image file paths
            - metadata: list of metadata dictionaries
            - stratify_keys: list of stratification keys
            - label_names: list of Pokemon species names
    """
    images_path = Path(images_dir)
    pokemon_folders = sorted([f for f in images_path.iterdir() if f.is_dir()])

    label_names = [folder.name for folder in pokemon_folders]
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    print(f"Found {len(label_names)} Pokemon species")

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print("Discovering image files...", flush=True)
    tasks = []
    for pokemon_folder in pokemon_folders:
        pokemon_name = pokemon_folder.name
        pokemon_name_lower = pokemon_name.lower()
        label_idx = label_to_idx[pokemon_name]

        png_files = list(pokemon_folder.glob('*.png'))
        jpg_files = list(pokemon_folder.glob('*.jpg'))
        image_files = sorted(png_files + jpg_files)

        for img_file in image_files:
            image_stem = img_file.stem
            metadata_file = img_file.parent / f"{pokemon_name_lower}_{image_stem}.json"

            if require_metadata and not metadata_file.exists():
                continue

            tasks.append((img_file, metadata_file, label_idx, pokemon_name))

    print(f"Loading {len(tasks)} images with {max_workers} workers...", flush=True)
    print(f"Target resolution: {image_size}×{image_size}", flush=True)

    data_size = image_size * image_size * 3
    all_data = np.zeros((len(tasks), data_size), dtype=np.uint8)
    all_labels = np.zeros(len(tasks), dtype=np.int32)
    all_filenames = [None] * len(tasks)
    all_metadata = [None] * len(tasks)
    all_stratify_keys = [None] * len(tasks)

    skipped_invalid_shape = 0
    skipped_error = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_load_single_image, img_path, meta_path, label, pname, image_size): i
            for i, (img_path, meta_path, label, pname) in enumerate(tasks)
        }

        for completed, future in enumerate(as_completed(future_to_idx), 1):
            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(tasks)} ({completed*100//len(tasks)}%)", flush=True)

            result = future.result()

            if result['success']:
                idx = future_to_idx[future]
                all_data[idx] = result['data']
                all_labels[idx] = result['label']
                all_filenames[idx] = result['filename']
                all_metadata[idx] = result['metadata']
                all_stratify_keys[idx] = result['stratify_key']
            else:
                if 'invalid_shape' in result.get('error', ''):
                    skipped_invalid_shape += 1
                else:
                    skipped_error += 1
                    if 'path' in result:
                        print(f"Error loading {result['path']}: {result['error']}")

    valid_indices = [i for i in range(len(tasks)) if all_filenames[i] is not None]

    if len(valid_indices) < len(tasks):
        print(f"Filtering: {len(valid_indices)} valid images from {len(tasks)} total", flush=True)
        all_data = all_data[valid_indices]
        all_labels = all_labels[valid_indices]
        all_filenames = [all_filenames[i] for i in valid_indices]
        all_metadata = [all_metadata[i] for i in valid_indices]
        all_stratify_keys = [all_stratify_keys[i] for i in valid_indices]

    if skipped_invalid_shape > 0:
        print(f"Skipped {skipped_invalid_shape} images (invalid dimensions)")
    if skipped_error > 0:
        print(f"Skipped {skipped_error} images (load errors)")

    print(f"Loaded {len(all_data)} images successfully")

    return {
        'data': all_data,
        'labels': all_labels,
        'filenames': all_filenames,
        'metadata': all_metadata,
        'stratify_keys': all_stratify_keys,
        'label_names': label_names
    }


def _create_stratification_key(metadata: Optional[Dict]) -> str:
    """
    Generate a stratification key from Pokemon metadata attributes.

    Combines type, color, and body shape into a composite key used for
    balanced sampling across train/val/test splits.

    Args:
        metadata: Dictionary containing Pokemon attributes, or None

    Returns:
        Stratification key string in format "Type_Color_BodyShape"
    """
    if metadata is None:
        return "unknown_unknown_unknown"

    cat_attrs = metadata.get('categorical_attributes', {})
    vis_attrs = metadata.get('visual_attributes', {})

    type1 = cat_attrs.get('type1', 'Normal')
    color_dist = vis_attrs.get('color_distribution', {})
    primary_color = color_dist.get('primary', 'Blue')
    body_shape = vis_attrs.get('body_shape', 'Bipedal')

    return f"{type1}_{primary_color}_{body_shape}"


def balance_dataset_by_attributes(
    data_dict: Dict,
    target_per_group: int = 50,
    min_per_group: int = 10
) -> Dict:
    """
    Balance dataset by equalizing samples across attribute combinations.

    Uses oversampling and undersampling to ensure each Type×Color×BodyShape
    combination has equal representation. This prevents model bias toward
    common attribute combinations.

    Args:
        data_dict: Dataset dictionary from load_pokemon_dataset
        target_per_group: Target sample count per stratification group
        min_per_group: Minimum samples required to retain a group

    Returns:
        Balanced dataset dictionary with equalized groups
    """
    print(f"\n{'='*80}")
    print("BALANCING DATASET BY ATTRIBUTE COMBINATIONS")
    print(f"{'='*80}")
    print(f"Target samples per group: {target_per_group}")
    print(f"Minimum samples to retain: {min_per_group}\n")

    groups = defaultdict(list)
    for i, key in enumerate(data_dict['stratify_keys']):
        groups[key].append(i)

    group_sizes = [len(indices) for indices in groups.values()]
    print(f"Before balancing:")
    print(f"  Unique groups: {len(groups)}")
    print(f"  Group size range: {min(group_sizes)}-{max(group_sizes)}")
    print(f"  Average group size: {np.mean(group_sizes):.1f}")
    print(f"  Total samples: {sum(group_sizes)}\n")

    balanced_indices = []
    stats = {
        'removed': 0,
        'oversampled': 0,
        'undersampled': 0,
        'unchanged': 0
    }

    for key, indices in sorted(groups.items()):
        current_count = len(indices)

        if current_count < min_per_group:
            stats['removed'] += 1
            continue

        if current_count > target_per_group:
            selected = random.sample(indices, target_per_group)
            stats['undersampled'] += 1
        elif current_count < target_per_group:
            selected = indices.copy()
            needed = target_per_group - current_count
            selected.extend(random.choices(indices, k=needed))
            stats['oversampled'] += 1
        else:
            selected = indices
            stats['unchanged'] += 1

        balanced_indices.extend(selected)

    random.seed(42)
    random.shuffle(balanced_indices)

    balanced_data = {
        'data': data_dict['data'][balanced_indices],
        'labels': [data_dict['labels'][i] for i in balanced_indices],
        'filenames': [data_dict['filenames'][i] for i in balanced_indices],
        'metadata': [data_dict['metadata'][i] for i in balanced_indices],
        'stratify_keys': [data_dict['stratify_keys'][i] for i in balanced_indices],
        'label_names': data_dict['label_names']
    }

    print(f"After balancing:")
    print(f"  Groups retained: {len(groups) - stats['removed']}")
    print(f"  Groups removed (< {min_per_group}): {stats['removed']}")
    print(f"  Oversampled: {stats['oversampled']}")
    print(f"  Undersampled: {stats['undersampled']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Total samples: {len(balanced_indices)}")

    balanced_groups = defaultdict(int)
    for key in balanced_data['stratify_keys']:
        balanced_groups[key] += 1

    group_sizes_after = list(balanced_groups.values())
    print(f"  Final group size: {min(group_sizes_after)}-{max(group_sizes_after)} "
          f"(avg: {np.mean(group_sizes_after):.1f})")
    print(f"{'='*80}\n")

    return balanced_data


def create_stratified_split(
    data_dict: Dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Dict, Dict, Dict]:
    """
    Create stratified train/validation/test splits with balanced distributions.

    Maintains consistent attribute distributions across all splits to prevent
    distribution shift during evaluation.

    Args:
        data_dict: Dataset dictionary from balance_dataset_by_attributes
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion

    Returns:
        Tuple of (train_dict, val_dict, test_dict)

    Raises:
        AssertionError: If ratios don't sum to 1.0
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    print(f"\nCreating stratified splits:")
    print(f"  Train: {train_ratio*100:.1f}%")
    print(f"  Validation: {val_ratio*100:.1f}%")
    print(f"  Test: {test_ratio*100:.1f}%")
    print(f"  Stratification: Type × Color × Body Shape")

    stratify_counter = Counter(data_dict['stratify_keys'])
    print(f"\nDiscovered {len(stratify_counter)} unique groups")

    min_samples_for_stratify = 5
    rare_groups = {key for key, count in stratify_counter.items() if count < min_samples_for_stratify}
    if rare_groups:
        print(f"Warning: {len(rare_groups)} groups have <{min_samples_for_stratify} samples and will be handled specially")

    # Separate rare and common samples
    rare_indices = [i for i, key in enumerate(data_dict['stratify_keys']) if key in rare_groups]
    common_indices = [i for i, key in enumerate(data_dict['stratify_keys']) if key not in rare_groups]

    print(f"  Rare samples: {len(rare_indices)}")
    print(f"  Common samples: {len(common_indices)}")

    # For common samples, use stratified split
    if common_indices:
        common_data = data_dict['data'][common_indices]
        common_labels = [data_dict['labels'][i] for i in common_indices]
        common_filenames = [data_dict['filenames'][i] for i in common_indices]
        common_metadata = [data_dict['metadata'][i] for i in common_indices]
        common_stratify = [data_dict['stratify_keys'][i] for i in common_indices]

        # First split: train vs (val + test)
        train_data, temp_data, train_labels, temp_labels, \
        train_filenames, temp_filenames, train_metadata, temp_metadata = train_test_split(
            common_data, common_labels, common_filenames, common_metadata,
            test_size=(val_ratio + test_ratio),
            stratify=common_stratify,
            random_state=42
        )

        # Second split: val vs test (from temp)
        # Check which groups have enough samples for stratification
        temp_stratify = [_create_stratification_key(m) for m in temp_metadata]
        temp_stratify_counter = Counter(temp_stratify)
        temp_rare_groups = {key for key, count in temp_stratify_counter.items() if count < 2}

        if temp_rare_groups:
            # Some groups too small, filter them out for stratified split
            temp_common_indices = [i for i, key in enumerate(temp_stratify) if key not in temp_rare_groups]
            temp_rare_indices = [i for i, key in enumerate(temp_stratify) if key in temp_rare_groups]

            # Calculate adjusted ratio for val/test split
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)

            if temp_common_indices:
                # Stratify common samples
                temp_common_data = temp_data[temp_common_indices]
                temp_common_labels = [temp_labels[i] for i in temp_common_indices]
                temp_common_filenames = [temp_filenames[i] for i in temp_common_indices]
                temp_common_metadata = [temp_metadata[i] for i in temp_common_indices]
                temp_common_stratify = [temp_stratify[i] for i in temp_common_indices]

                val_data_common, test_data_common, val_labels_common, test_labels_common, \
                val_filenames_common, test_filenames_common, val_metadata_common, test_metadata_common = train_test_split(
                    temp_common_data, temp_common_labels, temp_common_filenames, temp_common_metadata,
                    test_size=(1 - val_ratio_adjusted),
                    stratify=temp_common_stratify,
                    random_state=42
                )
            else:
                val_data_common, test_data_common = np.array([]), np.array([])
                val_labels_common, test_labels_common = [], []
                val_filenames_common, test_filenames_common = [], []
                val_metadata_common, test_metadata_common = [], []

            # Distribute rare samples randomly
            if temp_rare_indices:
                n_temp_rare = len(temp_rare_indices)
                n_val_rare = int(n_temp_rare * val_ratio_adjusted)

                temp_rare_val_idx = temp_rare_indices[:n_val_rare]
                temp_rare_test_idx = temp_rare_indices[n_val_rare:]

                val_data_rare = temp_data[temp_rare_val_idx]
                val_labels_rare = [temp_labels[i] for i in temp_rare_val_idx]
                val_filenames_rare = [temp_filenames[i] for i in temp_rare_val_idx]
                val_metadata_rare = [temp_metadata[i] for i in temp_rare_val_idx]

                test_data_rare = temp_data[temp_rare_test_idx]
                test_labels_rare = [temp_labels[i] for i in temp_rare_test_idx]
                test_filenames_rare = [temp_filenames[i] for i in temp_rare_test_idx]
                test_metadata_rare = [temp_metadata[i] for i in temp_rare_test_idx]

                # Combine common and rare
                val_data = np.concatenate([val_data_common, val_data_rare]) if len(val_data_common) > 0 else val_data_rare
                val_labels = val_labels_common + val_labels_rare
                val_filenames = val_filenames_common + val_filenames_rare
                val_metadata = val_metadata_common + val_metadata_rare

                test_data = np.concatenate([test_data_common, test_data_rare]) if len(test_data_common) > 0 else test_data_rare
                test_labels = test_labels_common + test_labels_rare
                test_filenames = test_filenames_common + test_filenames_rare
                test_metadata = test_metadata_common + test_metadata_rare
            else:
                val_data, val_labels = val_data_common, val_labels_common
                val_filenames, val_metadata = val_filenames_common, val_metadata_common
                test_data, test_labels = test_data_common, test_labels_common
                test_filenames, test_metadata = test_filenames_common, test_metadata_common
        else:
            # All groups have enough samples, use pure stratification
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)

            val_data, test_data, val_labels, test_labels, \
            val_filenames, test_filenames, val_metadata, test_metadata = train_test_split(
                temp_data, temp_labels, temp_filenames, temp_metadata,
                test_size=(1 - val_ratio_adjusted),
                stratify=temp_stratify,
                random_state=42
            )
    else:
        # No common samples
        train_data, train_labels, train_filenames, train_metadata = [], [], [], []
        val_data, val_labels, val_filenames, val_metadata = [], [], [], []
        test_data, test_labels, test_filenames, test_metadata = [], [], [], []

    # For rare samples, distribute them randomly but proportionally
    if rare_indices:
        np.random.seed(42)
        np.random.shuffle(rare_indices)

        n_rare = len(rare_indices)
        n_train_rare = int(n_rare * train_ratio)
        n_val_rare = int(n_rare * val_ratio)

        rare_train_idx = rare_indices[:n_train_rare]
        rare_val_idx = rare_indices[n_train_rare:n_train_rare + n_val_rare]
        rare_test_idx = rare_indices[n_train_rare + n_val_rare:]

        # Add rare samples to splits
        train_data = np.concatenate([train_data, data_dict['data'][rare_train_idx]]) if len(train_data) > 0 else data_dict['data'][rare_train_idx]
        train_labels.extend([data_dict['labels'][i] for i in rare_train_idx])
        train_filenames.extend([data_dict['filenames'][i] for i in rare_train_idx])
        train_metadata.extend([data_dict['metadata'][i] for i in rare_train_idx])

        val_data = np.concatenate([val_data, data_dict['data'][rare_val_idx]]) if len(val_data) > 0 else data_dict['data'][rare_val_idx]
        val_labels.extend([data_dict['labels'][i] for i in rare_val_idx])
        val_filenames.extend([data_dict['filenames'][i] for i in rare_val_idx])
        val_metadata.extend([data_dict['metadata'][i] for i in rare_val_idx])

        test_data = np.concatenate([test_data, data_dict['data'][rare_test_idx]]) if len(test_data) > 0 else data_dict['data'][rare_test_idx]
        test_labels.extend([data_dict['labels'][i] for i in rare_test_idx])
        test_filenames.extend([data_dict['filenames'][i] for i in rare_test_idx])
        test_metadata.extend([data_dict['metadata'][i] for i in rare_test_idx])

    # Create dictionaries
    train_dict = {
        'data': train_data,
        'labels': train_labels.tolist() if isinstance(train_labels, np.ndarray) else train_labels,
        'filenames': train_filenames,
        'metadata': train_metadata
    }

    val_dict = {
        'data': val_data,
        'labels': val_labels.tolist() if isinstance(val_labels, np.ndarray) else val_labels,
        'filenames': val_filenames,
        'metadata': val_metadata
    }

    test_dict = {
        'data': test_data,
        'labels': test_labels.tolist() if isinstance(test_labels, np.ndarray) else test_labels,
        'filenames': test_filenames,
        'metadata': test_metadata
    }

    print(f"\nStratified split complete:")
    print(f"  Training:   {len(train_dict['data'])} samples")
    print(f"  Validation: {len(val_dict['data'])} samples")
    print(f"  Test:       {len(test_dict['data'])} samples")

    return train_dict, val_dict, test_dict


def save_cifar_batches(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    label_names: List[str],
    output_dir: str,
    num_train_batches: int = 5,
    image_size: int = 32
) -> None:
    """
    Save dataset in CIFAR-10 compatible pickle format.

    Combines training and validation data into multiple batch files,
    saves test data separately, and creates metadata files.

    Args:
        train_data: Training set dictionary
        val_data: Validation set dictionary
        test_data: Test set dictionary
        label_names: List of Pokemon species names
        output_dir: Output directory path
        num_train_batches: Number of training batch files to create
        image_size: Image dimensions (for metadata)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    combined_train_data = np.concatenate([train_data['data'], val_data['data']])
    combined_train_labels = train_data['labels'] + val_data['labels']
    combined_train_filenames = train_data['filenames'] + val_data['filenames']
    combined_train_metadata = train_data['metadata'] + val_data['metadata']

    np.random.seed(42)
    indices = np.random.permutation(len(combined_train_data))
    combined_train_data = combined_train_data[indices]
    combined_train_labels = [combined_train_labels[i] for i in indices]
    combined_train_filenames = [combined_train_filenames[i] for i in indices]
    combined_train_metadata = [combined_train_metadata[i] for i in indices]

    train_size = len(combined_train_data)
    batch_size = train_size // num_train_batches

    for i in range(num_train_batches):
        start_idx = i * batch_size
        if i == num_train_batches - 1:
            end_idx = train_size
        else:
            end_idx = (i + 1) * batch_size

        batch_dict = {
            'data': combined_train_data[start_idx:end_idx],
            'labels': combined_train_labels[start_idx:end_idx],
            'filenames': combined_train_filenames[start_idx:end_idx],
            'metadata': combined_train_metadata[start_idx:end_idx],
            'batch_label': f'training batch {i + 1} of {num_train_batches}'
        }

        batch_file = output_path / f'data_batch_{i + 1}'
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_dict, f)

        print(f"Saved {batch_file} with {len(batch_dict['data'])} images")

    # Save test batch
    test_batch = {
        'data': test_data['data'],
        'labels': test_data['labels'],
        'filenames': test_data['filenames'],
        'metadata': test_data['metadata'],
        'batch_label': 'testing batch 1 of 1 (BALANCED)'
    }

    test_file = output_path / 'test_batch'
    with open(test_file, 'wb') as f:
        pickle.dump(test_batch, f)

    print(f"Saved {test_file} with {len(test_batch['data'])} images")

    meta = {
        'label_names': label_names,
        'num_cases_per_batch': batch_size,
        'num_vis': image_size * image_size * 3,
        'balanced': True,
        'split_method': 'stratified'
    }

    meta_file = output_path / 'batches.meta'
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved {meta_file}")

    readme_content = f"""Pokemon CIFAR Dataset
===============================================================================

Dataset Format: CIFAR-10 compatible
Pokemon Species: {len(label_names)}
Image Resolution: {image_size}×{image_size} RGB
Balancing Method: Stratified sampling with attribute-based oversampling

Files
-----
- data_batch_1 through data_batch_{num_train_batches}: Training data
- test_batch: Test data
- batches.meta: Dataset metadata
- README.txt: This file

Dataset Statistics
------------------
Training + Validation: {len(combined_train_data)} images
Test: {len(test_data['data'])} images
Total: {len(combined_train_data) + len(test_data['data'])} images

Balancing Strategy
------------------
This dataset uses multi-attribute stratified sampling to ensure balanced
representation across Pokemon type, color, and body shape combinations.

- Rare attribute combinations are oversampled
- Common combinations are undersampled
- Train/val/test splits maintain consistent distributions
- Prevents distribution shift during model evaluation

Expected Model Behavior
-----------------------
Due to balanced sampling, validation and test losses should be similar.
Typical reconstruction loss: 80-100 (L1) for 32×32 images.
"""

    readme_file = output_path / 'README.txt'
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"Saved {readme_file}")


def main():
    """Main execution function for dataset creation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build balanced Pokemon dataset in CIFAR-10 format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images_32x32",
        help="Path to Pokemon images directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pokemon-cifar",
        help="Output directory for CIFAR batches"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set proportion"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set proportion"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set proportion"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        choices=[32, 64],
        help="Image dimensions in pixels"
    )
    parser.add_argument(
        "--target-per-group",
        type=int,
        default=50,
        help="Target samples per attribute group"
    )
    parser.add_argument(
        "--min-per-group",
        type=int,
        default=10,
        help="Minimum samples to retain a group"
    )

    args = parser.parse_args()

    print("="*80)
    print("POKEMON CIFAR DATASET BUILDER")
    print("="*80)
    print(f"Source directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image resolution: {args.image_size}×{args.image_size}")
    print(f"Split ratios: {args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%}")
    print("="*80)

    print("\n[1/4] Loading Pokemon images and metadata...")
    dataset = load_pokemon_dataset(
        args.images_dir,
        require_metadata=True,
        image_size=args.image_size
    )

    print("\n[2/4] Balancing dataset by attributes...")
    dataset = balance_dataset_by_attributes(
        dataset,
        target_per_group=args.target_per_group,
        min_per_group=args.min_per_group
    )

    print("\n[3/4] Creating stratified splits...")
    train_data, val_data, test_data = create_stratified_split(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    print("\n[4/4] Saving CIFAR-format batches...")
    save_cifar_batches(
        train_data, val_data, test_data,
        dataset['label_names'],
        args.output_dir,
        image_size=args.image_size
    )

    print("\n" + "="*80)
    print("DATASET CREATION COMPLETE")
    print("="*80)
    print(f"Output: {args.output_dir}")
    print(f"Species: {len(dataset['label_names'])}")
    print(f"Training: {len(train_data['data']) + len(val_data['data'])} images")
    print(f"Test: {len(test_data['data'])} images")
    print("="*80)


if __name__ == "__main__":
    main()
