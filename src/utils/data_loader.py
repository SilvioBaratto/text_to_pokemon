"""
Pokemon-CIFAR dataset loading and PyTorch data utilities.

Provides Dataset class and DataLoader factory for Pokemon images in CIFAR format.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from src.utils.image_preprocessing import preprocess_pokemon_image
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    print("Warning: Image preprocessing not available")


class PokemonCIFARDataset(Dataset):
    """
    Pokemon dataset in CIFAR format.

    Loads Pokemon images with metadata and supports prompt-based dataset expansion.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = 'train',
        use_preprocessing: bool = False
    ):
        """
        Initialize dataset.

        Args:
            dataset_dir: Path to pokemon-cifar directory
            split: 'train' or 'test'
            use_preprocessing: Apply image preprocessing
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.use_preprocessing = use_preprocessing and PREPROCESSING_AVAILABLE

        if use_preprocessing and not PREPROCESSING_AVAILABLE:
            print("Warning: Preprocessing requested but dependencies not installed")

        self.images = []
        self.labels = []
        self.filenames = []
        self.pokemon_names = []
        self.all_metadata = []
        self.prompt_indices = []

        meta_path = self.dataset_dir / 'batches.meta'
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            self.label_names = meta['label_names']
            num_vis = meta.get('num_vis', 3072)
            self.image_size = int((num_vis // 3) ** 0.5)
            print(f"Detected image size: {self.image_size}×{self.image_size}")

        self.metadata_by_label = {}

        if split == 'train':
            batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        elif split == 'test':
            batch_files = ['test_batch']
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        for batch_file in batch_files:
            batch_path = self.dataset_dir / batch_file
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')

            batch_images = batch['data']
            batch_labels = batch['labels']
            batch_filenames = batch['filenames']
            batch_metadata = batch.get('metadata', [None] * len(batch_labels))

            batch_images = batch_images.reshape(-1, 3, self.image_size, self.image_size)

            self.images.append(batch_images)
            self.labels.extend(batch_labels)
            self.filenames.extend(batch_filenames)
            self.all_metadata.extend(batch_metadata)

        base_images = np.concatenate(self.images, axis=0)

        for idx in range(len(self.labels)):
            label_id = self.labels[idx]
            metadata = self.all_metadata[idx]
            if metadata is not None:
                if label_id not in self.metadata_by_label:
                    self.metadata_by_label[label_id] = metadata

        print(f"Built metadata lookup for {len(self.metadata_by_label)} unique Pokemon")

        expanded_images = []
        expanded_labels = []
        expanded_filenames = []
        expanded_metadata = []
        expanded_prompt_indices = []

        original_count = len(base_images)

        for idx in range(original_count):
            metadata = self.all_metadata[idx]

            if metadata is not None and 'prompts' in metadata:
                num_prompts = len(metadata['prompts'])
            else:
                num_prompts = 1

            for prompt_idx in range(num_prompts):
                expanded_images.append(base_images[idx])
                expanded_labels.append(self.labels[idx])
                expanded_filenames.append(self.filenames[idx])
                expanded_metadata.append(metadata)
                expanded_prompt_indices.append(prompt_idx)

        self.images = np.stack(expanded_images, axis=0)
        self.labels = expanded_labels
        self.filenames = expanded_filenames
        self.all_metadata = expanded_metadata
        self.prompt_indices = expanded_prompt_indices

        print(f"Dataset expanded: {original_count} images -> {len(self.images)} samples")

        num_samples = len(self.images)
        seed = 42 if split == 'train' else 123
        shuffle_indices = np.random.RandomState(seed).permutation(num_samples)

        self.images = self.images[shuffle_indices]
        self.labels = [self.labels[i] for i in shuffle_indices]
        self.filenames = [self.filenames[i] for i in shuffle_indices]
        self.all_metadata = [self.all_metadata[i] for i in shuffle_indices]
        self.prompt_indices = [self.prompt_indices[i] for i in shuffle_indices]
        print(f"Dataset shuffled (seed={seed})")

        self.pokemon_names = []
        for filename in self.filenames:
            pokemon_name = filename.split('/')[0]
            self.pokemon_names.append(pokemon_name)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        """
        Get single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label, pokemon_name, prompt_idx)
        """
        image = self.images[idx]

        if self.use_preprocessing:
            image_hwc = image.transpose(1, 2, 0)

            image_hwc = preprocess_pokemon_image(
                image_hwc,
                remove_background=True,
                enhance_contrast=True,
                sharpen=False,
                bilateral_filter=True
            )

            image = image_hwc.transpose(2, 0, 1)

        image = (image.astype(np.float32) / 255.0) * 2.0 - 1.0

        image = torch.from_numpy(image)

        label = self.labels[idx]
        pokemon_name = self.pokemon_names[idx]
        prompt_idx = self.prompt_indices[idx]

        return image, label, pokemon_name, prompt_idx

    def get_metadata(self, idx: int) -> Optional[dict]:
        """
        Get metadata by sample index.

        Args:
            idx: Sample index

        Returns:
            Metadata dict or None
        """
        if idx < len(self.all_metadata):
            return self.all_metadata[idx]
        return None

    def get_metadata_by_label(self, label_id: int) -> Optional[dict]:
        """
        Get metadata by Pokemon label ID.

        Args:
            label_id: Pokemon label ID

        Returns:
            Metadata dict or None
        """
        return self.metadata_by_label.get(label_id, None)

    def get_label_name(self, label_id: int) -> str:
        """Get Pokemon name from label ID."""
        return self.label_names[label_id]

    def get_num_classes(self) -> int:
        """Get number of Pokemon classes."""
        return len(self.label_names)


def get_pokemon_dataloaders(
    dataset_dir: str = 'pokemon-cifar',
    batch_size: int = 64,
    val_split: float = 0.2,
    num_workers: int = 0,
    debug_samples: Optional[int] = None,
    use_preprocessing: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for Pokemon dataset.

    Args:
        dataset_dir: Path to pokemon-cifar directory
        batch_size: Batch size
        val_split: Validation split fraction
        num_workers: Number of DataLoader workers
        debug_samples: Limit dataset size for debugging
        use_preprocessing: Apply image preprocessing

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    full_train_dataset = PokemonCIFARDataset(
        dataset_dir=dataset_dir,
        split='train',
        use_preprocessing=use_preprocessing
    )

    test_dataset = PokemonCIFARDataset(
        dataset_dir=dataset_dir,
        split='test',
        use_preprocessing=use_preprocessing
    )

    if debug_samples is not None:
        print(f"\n{'='*80}")
        print(f"DEBUG MODE: Limiting to {debug_samples} total samples")
        print(f"{'='*80}")

        train_samples = int(debug_samples * 0.71)
        val_samples = int(debug_samples * 0.14)
        test_samples = debug_samples - train_samples - val_samples

        from torch.utils.data import Subset
        full_train_dataset = Subset(full_train_dataset, range(min(train_samples + val_samples, len(full_train_dataset))))
        test_dataset = Subset(test_dataset, range(min(test_samples, len(test_dataset))))

        print(f"Train+Val: {len(full_train_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples")
        print(f"{'='*80}\n")

    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    use_pin_memory = torch.cuda.is_available()
    persistent_workers = use_pin_memory and num_workers > 0

    if num_workers == 0 and torch.cuda.is_available():
        sys.path.append('.')
        import config as cfg
        num_workers = cfg.NUM_WORKERS
        print(f"Using {num_workers} DataLoader workers")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader, test_loader
