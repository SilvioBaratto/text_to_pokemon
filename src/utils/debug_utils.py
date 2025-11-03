"""
Debug utilities for Pokemon CVAE training.

Saves dataset images and metadata for inspection and validation.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.training_utils import get_base_dataset


def save_dataset_images_for_debugging(
    data_loader: Any,
    output_dir: str,
    split_name: str,
    device: torch.device
) -> None:
    """
    Save images and metadata from dataloader for debugging.

    Args:
        data_loader: PyTorch DataLoader
        output_dir: Root output directory
        split_name: 'train', 'val', or 'test'
        device: Device to use
    """
    print(f"\n{'='*80}")
    print(f"SAVING {split_name.upper()} IMAGES FOR DEBUGGING")
    print(f"{'='*80}")

    split_dir = os.path.join(output_dir, f"images_{split_name}")
    os.makedirs(split_dir, exist_ok=True)

    base_dataset = get_base_dataset(data_loader.dataset)

    total_images = 0

    for batch_idx, (images, labels, pokemon_names, prompt_indices) in enumerate(tqdm(data_loader, desc=f"Saving {split_name} batches")):
        batch_dir = os.path.join(split_dir, f"batch_{batch_idx + 1}")
        os.makedirs(batch_dir, exist_ok=True)

        metadata_list = [base_dataset.get_metadata_by_label(label.item()) for label in labels]
        prompt_indices_list = prompt_indices.tolist()

        for img_idx, (image, label, pokemon_name, prompt_idx, metadata) in enumerate(
            zip(images, labels, pokemon_names, prompt_indices_list, metadata_list)
        ):
            img_num = img_idx + 1
            total_images += 1

            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            img_filename = f"image_{img_num}_{pokemon_name}.png"
            img_path = os.path.join(batch_dir, img_filename)
            pil_image.save(img_path)

            if metadata is not None:
                cat_attrs = metadata.get('categorical_attributes', {})
                vis_attrs = metadata.get('visual_attributes', {})
                sem_attrs = metadata.get('semantic_attributes', {})
                prompts = metadata.get('prompts', [])

                selected_prompt = prompts[prompt_idx] if prompts and prompt_idx < len(prompts) else "A Pokemon"

                metadata_dict = {
                    "image_number": img_num,
                    "batch_number": batch_idx + 1,
                    "pokemon_name": pokemon_name,
                    "pokemon_id": label.item(),
                    "prompt_index": prompt_idx,
                    "selected_prompt": selected_prompt,
                    "all_prompts": prompts,
                    "categorical_attributes": {
                        "type1": cat_attrs.get('type1', 'Unknown'),
                        "type2": cat_attrs.get('type2', None),
                        "is_legendary": cat_attrs.get('is_legendary', False),
                        "is_mythical": cat_attrs.get('is_mythical', False),
                        "evolution_stage": cat_attrs.get('evolution_stage', 'Unknown')
                    },
                    "visual_attributes": {
                        "primary_color": vis_attrs.get('color_distribution', {}).get('primary', 'Unknown'),
                        "secondary_color": vis_attrs.get('color_distribution', {}).get('secondary', None),
                        "body_shape": vis_attrs.get('body_shape', 'Unknown'),
                        "size_class": vis_attrs.get('size_class', 'Unknown')
                    },
                    "semantic_attributes": {
                        "habitat": sem_attrs.get('habitat', 'Unknown')
                    }
                }
            else:
                metadata_dict = {
                    "image_number": img_num,
                    "batch_number": batch_idx + 1,
                    "pokemon_name": pokemon_name,
                    "pokemon_id": label.item(),
                    "prompt_index": prompt_idx,
                    "metadata": "Not available"
                }

            json_filename = f"image_{img_num}_{pokemon_name}_metadata.json"
            json_path = os.path.join(batch_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)

    print(f"Saved {total_images} images to: {split_dir}")

    summary_path = os.path.join(split_dir, f"_SUMMARY_{split_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Dataset Split: {split_name.upper()}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Total Batches: {batch_idx + 1}\n")
        f.write(f"Batch Size: {len(images)}\n\n")
        f.write(f"Directory Structure:\n")
        f.write(f"  {split_dir}/\n")
        f.write(f"    ├── batch_1/\n")
        f.write(f"    │   ├── image_1_<pokemon_name>.png\n")
        f.write(f"    │   ├── image_1_<pokemon_name>_metadata.json\n")
        f.write(f"    │   ├── image_2_<pokemon_name>.png\n")
        f.write(f"    │   ├── image_2_<pokemon_name>_metadata.json\n")
        f.write(f"    │   └── ...\n")
        f.write(f"    ├── batch_2/\n")
        f.write(f"    │   └── ...\n")
        f.write(f"    └── ...\n\n")
        f.write(f"Metadata includes:\n")
        f.write(f"  - Pokemon name and ID\n")
        f.write(f"  - Selected text prompt (and all available prompts)\n")
        f.write(f"  - Types (type1, type2)\n")
        f.write(f"  - Colors (primary, secondary)\n")
        f.write(f"  - Body shape\n")
        f.write(f"  - Size class\n")
        f.write(f"  - Evolution stage\n")
        f.write(f"  - Habitat\n")
        f.write(f"  - Legendary/Mythical status\n")

    print(f"Created summary file: {summary_path}")
    print(f"{'='*80}\n")
