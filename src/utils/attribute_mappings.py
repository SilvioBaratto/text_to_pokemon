"""
Attribute mappings for converting Pokemon metadata to tensor indices.

Maps categorical attributes (types, colors, shapes) to integer indices for
embedding layers in the CVAE model.
"""

from typing import Dict
POKEMON_TYPE_TO_IDX: Dict[str, int] = {
    "Normal": 0,
    "Fire": 1,
    "Water": 2,
    "Electric": 3,
    "Grass": 4,
    "Ice": 5,
    "Fighting": 6,
    "Poison": 7,
    "Ground": 8,
    "Flying": 9,
    "Psychic": 10,
    "Bug": 11,
    "Rock": 12,
    "Ghost": 13,
    "Dragon": 14,
    "Dark": 15,
    "Steel": 16,
    "Fairy": 17
}

POKEMON_COLOR_TO_IDX: Dict[str, int] = {
    "Red": 0,
    "Blue": 1,
    "Yellow": 2,
    "Green": 3,
    "Purple": 4,
    "Orange": 5,
    "Pink": 6,
    "Brown": 7,
    "Black": 8,
    "White": 9,
    "Gray": 10
}

BODY_SHAPE_TO_IDX: Dict[str, int] = {
    "Bipedal": 0,
    "Quadruped": 1,
    "Serpentine": 2,
    "Winged": 3,
    "Amorphous": 4,
    "Insectoid": 5,
    "Aquatic": 6,
    "Humanoid": 7
}

SIZE_CLASS_TO_IDX: Dict[str, int] = {
    "Tiny": 0,
    "Small": 1,
    "Medium": 2,
    "Large": 3,
    "Huge": 4
}

EVOLUTION_STAGE_TO_IDX: Dict[str, int] = {
    "Basic": 0,
    "Middle": 1,
    "Final": 2,
    "Single": 3,
    "Mega": 4,
    "Gigantamax": 5
}

HABITAT_TO_IDX: Dict[str, int] = {
    "Forest": 0,
    "Grassland": 1,
    "Mountain": 2,
    "Cave": 3,
    "Sea": 4,
    "Urban": 5,
    "Rare": 6,
    "Desert": 7,
    "Tundra": 8,
    "Wetland": 9
}


def get_type_idx(type_str: str, default: int = 0) -> int:
    """Convert Pokemon type string to index."""
    return POKEMON_TYPE_TO_IDX.get(type_str, default)


def get_color_idx(color_str: str, default: int = 0) -> int:
    """Convert Pokemon color string to index."""
    return POKEMON_COLOR_TO_IDX.get(color_str, default)


def get_shape_idx(shape_str: str, default: int = 0) -> int:
    """Convert body shape string to index."""
    return BODY_SHAPE_TO_IDX.get(shape_str, default)


def get_size_idx(size_str: str, default: int = 1) -> int:
    """Convert size class string to index."""
    return SIZE_CLASS_TO_IDX.get(size_str, default)


def get_evolution_stage_idx(stage_str: str, default: int = 0) -> int:
    """Convert evolution stage string to index."""
    return EVOLUTION_STAGE_TO_IDX.get(stage_str, default)


def get_habitat_idx(habitat_str: str, default: int = 0) -> int:
    """Convert habitat string to index."""
    return HABITAT_TO_IDX.get(habitat_str, default)
