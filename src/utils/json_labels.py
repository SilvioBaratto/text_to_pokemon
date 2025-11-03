"""
Utility functions for working with pokemon-cifar JSON labels.

This module provides convenient access to the JSON metadata file
that supplements the original pickled CIFAR format.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class PokemonLabels:
    """Interface for accessing pokemon-cifar JSON labels."""

    def __init__(self, json_path: str):
        """
        Initialize with path to pokemon_labels.json file.

        Args:
            json_path: Path to the JSON labels file
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.samples = self.data['samples']
        self.classes = self.data['classes']

        # Build lookup indices
        self._build_indices()

    def _build_indices(self) -> None:
        """Build indices for fast lookup."""
        # Index by sample ID
        self.samples_by_id = {s['id']: s for s in self.samples}

        # Index by Pokemon name
        self.samples_by_pokemon = {}
        for sample in self.samples:
            name = sample['pokemon_name']
            if name not in self.samples_by_pokemon:
                self.samples_by_pokemon[name] = []
            self.samples_by_pokemon[name].append(sample)

        # Index by split
        self.train_samples = [s for s in self.samples if s['split'] == 'train']
        self.test_samples = [s for s in self.samples if s['split'] == 'test']

    def get_sample(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """
        Get sample metadata by ID.

        Args:
            sample_id: Sample ID

        Returns:
            Sample metadata dictionary or None if not found
        """
        return self.samples_by_id.get(sample_id)

    def get_samples_by_pokemon(self, pokemon_name: str) -> List[Dict[str, Any]]:
        """
        Get all samples for a specific Pokemon.

        Args:
            pokemon_name: Name of the Pokemon

        Returns:
            List of sample metadata dictionaries
        """
        return self.samples_by_pokemon.get(pokemon_name, [])

    def get_train_samples(self) -> List[Dict[str, Any]]:
        """Get all training samples."""
        return self.train_samples

    def get_test_samples(self) -> List[Dict[str, Any]]:
        """Get all test samples."""
        return self.test_samples

    def get_pokemon_names(self) -> List[str]:
        """Get list of all Pokemon class names."""
        return self.classes

    def get_num_samples(self, split: Optional[str] = None) -> int:
        """
        Get number of samples.

        Args:
            split: Optional split ('train' or 'test'). If None, returns total.

        Returns:
            Number of samples
        """
        if split == 'train':
            return len(self.train_samples)
        elif split == 'test':
            return len(self.test_samples)
        else:
            return len(self.samples)

    def get_num_classes(self) -> int:
        """Get number of Pokemon classes."""
        return len(self.classes)

    def search_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        """
        Search samples by filename pattern.

        Args:
            filename: Filename or pattern to search for

        Returns:
            List of matching samples
        """
        return [s for s in self.samples if filename in s['filename']]

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset summary information.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            'dataset': self.data['dataset'],
            'description': self.data['description'],
            'num_samples': self.data['num_samples'],
            'num_train': self.data['num_train'],
            'num_test': self.data['num_test'],
            'num_classes': self.data['num_classes'],
            'image_shape': self.data['image_shape']
        }


def load_pokemon_labels(dataset_dir: str = 'pokemon-cifar') -> PokemonLabels:
    """
    Load Pokemon labels from default location.

    Args:
        dataset_dir: Path to pokemon-cifar directory

    Returns:
        PokemonLabels instance
    """
    json_path = Path(dataset_dir) / 'pokemon_labels.json'
    return PokemonLabels(str(json_path))


# Example usage
if __name__ == '__main__':
    # Load labels
    labels = load_pokemon_labels('/Users/silviobaratto/Desktop/pokemon/pokemon-cifar')

    # Get dataset info
    info = labels.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Get samples for a specific Pokemon
    print("\nPikachu samples:")
    pikachu_samples = labels.get_samples_by_pokemon('Pikachu')
    for sample in pikachu_samples[:3]:
        print(f"  ID {sample['id']}: {sample['filename']} (batch: {sample['batch']})")

    # Get sample by ID
    print("\nSample ID 0:")
    sample = labels.get_sample(0)
    if sample:
        print(f"  Pokemon: {sample['pokemon_name']}")
        print(f"  Filename: {sample['filename']}")
        print(f"  Split: {sample['split']}")
