"""
Conditioning extraction from Pokemon metadata for CVAE training.

Converts metadata dictionaries into text embeddings and categorical attribute tensors.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.append('.')
import config
from src.model.text_encoder import encode_texts_batch
from src.utils.attribute_mappings import (
    get_type_idx, get_color_idx, get_shape_idx, get_size_idx,
    get_evolution_stage_idx, get_habitat_idx
)


def extract_conditioning_from_metadata(
    metadata_list: List[Optional[Dict[str, Any]]],
    prompt_indices: List[int],
    device: torch.device
) -> Tuple[torch.Tensor, ...]:
    """
    Extract conditioning vectors from metadata batch.

    Args:
        metadata_list: List of metadata dicts per image
        prompt_indices: Prompt index for each sample
        device: Device to place tensors on

    Returns:
        Tuple of (text_embeddings, type1, type2, primary_color, secondary_color,
                  shape, size, evolution_stage, habitat, legendary, mythical)
    """
    text_prompts = []
    type1_ids = []
    type2_ids = []
    primary_color_ids = []
    secondary_color_ids = []
    shape_ids = []
    size_ids = []
    evolution_stage_ids = []
    habitat_ids = []
    legendary_flags = []
    mythical_flags = []

    for metadata, prompt_idx in zip(metadata_list, prompt_indices):
        if metadata is None:
            text_prompts.append("A Pokemon")
            type1_ids.append(0)
            type2_ids.append(config.NUM_TYPES)
            primary_color_ids.append(0)
            secondary_color_ids.append(config.NUM_COLORS)
            shape_ids.append(0)
            size_ids.append(1)
            evolution_stage_ids.append(0)
            habitat_ids.append(0)
            legendary_flags.append(0)
            mythical_flags.append(0)
            continue

        prompts = metadata.get('prompts', [])
        if prompts and prompt_idx < len(prompts):
            text_prompts.append(prompts[prompt_idx])
        else:
            text_prompts.append("A Pokemon")

        cat_attrs = metadata.get('categorical_attributes', {})
        vis_attrs = metadata.get('visual_attributes', {})
        sem_attrs = metadata.get('semantic_attributes', {})

        type1 = cat_attrs.get('type1', 'Normal')
        type1_ids.append(get_type_idx(type1))

        type2 = cat_attrs.get('type2')
        if type2:
            type2_ids.append(get_type_idx(type2))
        else:
            type2_ids.append(config.NUM_TYPES)

        color_dist = vis_attrs.get('color_distribution', {})
        primary_color = color_dist.get('primary', 'Red')
        primary_color_ids.append(get_color_idx(primary_color))

        secondary_color = color_dist.get('secondary')
        if secondary_color:
            secondary_color_ids.append(get_color_idx(secondary_color))
        else:
            secondary_color_ids.append(config.NUM_COLORS)

        body_shape = vis_attrs.get('body_shape', 'Bipedal')
        shape_ids.append(get_shape_idx(body_shape))

        size_class = vis_attrs.get('size_class', 'Small')
        size_ids.append(get_size_idx(size_class))

        evolution_stage = cat_attrs.get('evolution_stage', 'Basic')
        evolution_stage_ids.append(get_evolution_stage_idx(evolution_stage))

        habitat = sem_attrs.get('habitat', 'Forest')
        habitat_ids.append(get_habitat_idx(habitat))

        legendary_flags.append(1 if cat_attrs.get('is_legendary', False) else 0)
        mythical_flags.append(1 if cat_attrs.get('is_mythical', False) else 0)

    text_embeddings = encode_texts_batch(text_prompts, device=str(device))

    type1_ids = torch.tensor(type1_ids, dtype=torch.long, device=device)
    type2_ids = torch.tensor(type2_ids, dtype=torch.long, device=device)
    primary_color_ids = torch.tensor(primary_color_ids, dtype=torch.long, device=device)
    secondary_color_ids = torch.tensor(secondary_color_ids, dtype=torch.long, device=device)
    shape_ids = torch.tensor(shape_ids, dtype=torch.long, device=device)
    size_ids = torch.tensor(size_ids, dtype=torch.long, device=device)
    evolution_stage_ids = torch.tensor(evolution_stage_ids, dtype=torch.long, device=device)
    habitat_ids = torch.tensor(habitat_ids, dtype=torch.long, device=device)
    legendary_flags = torch.tensor(legendary_flags, dtype=torch.long, device=device)
    mythical_flags = torch.tensor(mythical_flags, dtype=torch.long, device=device)

    return (text_embeddings, type1_ids, type2_ids, primary_color_ids, secondary_color_ids,
            shape_ids, size_ids, evolution_stage_ids, habitat_ids,
            legendary_flags, mythical_flags)
