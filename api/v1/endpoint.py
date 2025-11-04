"""
FastAPI endpoints for Pokemon image generation.

Provides REST API for text-to-image generation using trained CVAE model.
"""

import asyncio
import base64
import io
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import config
from src.model.vae import CVAE
from src.model.text_encoder import encode_text_to_embedding
from src.utils.attribute_mappings import (
    get_type_idx, get_color_idx, get_shape_idx, get_size_idx,
    get_evolution_stage_idx, get_habitat_idx
)

model: Optional[CVAE] = None
device: Optional[torch.device] = None

router = APIRouter()


class GenerateRequest(BaseModel):
    """Request model for Pokemon generation."""
    prompt: str = Field(
        description="Text description of the Pokemon to generate",
        examples=["A fire-type Pokemon with red and orange flames"]
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of Pokemon variations to generate (1-4)"
    )


class GenerateResponse(BaseModel):
    """Response model with generated Pokemon image."""
    success: bool
    image: str = Field(description="Base64 encoded PNG image")
    prompt: str
    message: Optional[str] = None


def load_model() -> None:
    """Load the trained Pokemon CVAE model."""
    global model, device

    if model is not None:
        return

    device = config.get_device()
    print(f"Loading model on device: {device}")

    model = CVAE(
        input_dim=config.INPUT_DIM,
        condition_dim=config.TOTAL_CONDITION_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM
    ).to(device)

    checkpoint_path = os.path.join(
        os.path.dirname(__file__), '../../checkpoints/best_model_pokemon.pt'
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please train the model first using: python src/train.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'unknown'):.2f}")


def parse_prompt_attributes(prompt: str) -> Dict[str, int]:
    """
    Parse text prompt to extract Pokemon attributes.

    Args:
        prompt: Text description

    Returns:
        Dictionary with attribute indices
    """
    prompt_lower = prompt.lower()

    type1_idx = get_type_idx("Normal")
    type2_idx = config.NUM_TYPES

    types_list = ["fire", "water", "grass", "electric", "psychic", "dragon",
                  "flying", "bug", "rock", "fighting", "poison", "ground",
                  "ice", "ghost", "steel", "dark", "fairy", "normal"]

    found_types = []
    for type_name in types_list:
        if type_name in prompt_lower:
            type_idx = get_type_idx(type_name.capitalize())
            if type_idx not in found_types:
                found_types.append(type_idx)

    if len(found_types) >= 1:
        type1_idx = found_types[0]
    if len(found_types) >= 2:
        type2_idx = found_types[1]

    primary_color_idx = get_color_idx("Brown")
    secondary_color_idx = config.NUM_COLORS

    color_names = ["red", "blue", "yellow", "green", "black", "white",
                   "brown", "purple", "pink", "gray", "grey", "orange"]
    color_synonyms = {
        "crimson": "Red", "scarlet": "Red", "ruby": "Red",
        "sapphire": "Blue", "azure": "Blue", "cyan": "Blue", "aqua": "Blue",
        "golden": "Yellow", "gold": "Yellow", "amber": "Yellow",
        "emerald": "Green", "jade": "Green", "lime": "Green",
        "ebony": "Black", "obsidian": "Black", "charcoal": "Black",
        "ivory": "White", "pearl": "White", "silver": "White",
        "violet": "Purple", "lavender": "Purple", "magenta": "Purple",
        "rose": "Pink", "salmon": "Pink"
    }

    found_colors = []
    words = prompt_lower.split()

    for word in words:
        for color_name in color_names:
            if color_name in word:
                color_idx = get_color_idx(color_name.capitalize())
                if color_idx not in found_colors:
                    found_colors.append(color_idx)
                break

        if not found_colors or len(found_colors) < 2:
            for synonym, canonical_color in color_synonyms.items():
                if synonym in word:
                    color_idx = get_color_idx(canonical_color)
                    if color_idx not in found_colors:
                        found_colors.append(color_idx)
                    break

    if len(found_colors) >= 1:
        primary_color_idx = found_colors[0]
    if len(found_colors) >= 2:
        secondary_color_idx = found_colors[1]

    shape_idx = get_shape_idx("Quadruped")
    shape_keywords = {
        "bipedal": "Bipedal", "two-legged": "Bipedal", "two legs": "Bipedal",
        "standing upright": "Bipedal", "walks upright": "Bipedal",

        "quadruped": "Quadruped", "four-legged": "Quadruped", "four legs": "Quadruped",

        "serpentine": "Serpentine", "snake-like": "Serpentine", "snake": "Serpentine",
        "slithering": "Serpentine", "eel-like": "Serpentine",

        "amorphous": "Amorphous", "blob": "Amorphous", "gaseous": "Amorphous",
        "wispy": "Amorphous", "cloud-like": "Amorphous", "shapeless": "Amorphous",

        "winged": "Winged", "wings": "Winged",
        "airborne": "Winged", "bird-like": "Winged",

        "insectoid": "Insectoid", "insect-like": "Insectoid", "bug-like": "Insectoid",

        "aquatic": "Aquatic", "fish-like": "Aquatic", "swimming": "Aquatic",
        "fish": "Aquatic", "marine": "Aquatic",

        "humanoid": "Humanoid", "human-like": "Humanoid", "anthropomorphic": "Humanoid"
    }

    for keyword, shape_name in shape_keywords.items():
        if keyword in prompt_lower:
            shape_idx = get_shape_idx(shape_name)
            break

    if type2_idx == get_type_idx("Flying") and shape_idx == get_shape_idx("Quadruped"):
        shape_idx = get_shape_idx("Winged")

    size_idx = 2
    if any(word in prompt_lower for word in ["small", "tiny", "little", "compact", "miniature"]):
        size_idx = get_size_idx("Small")
    elif any(word in prompt_lower for word in ["large", "big", "huge", "massive", "giant", "enormous"]):
        size_idx = get_size_idx("Large")

    evolution_idx = get_evolution_stage_idx("Basic")
    evolution_keywords = {
        "basic": "Basic", "unevolved": "Basic", "baby": "Basic", "first stage": "Basic",

        "stage 1": "Middle", "evolved": "Middle", "middle": "Middle",
        "second stage": "Middle", "mid-evolution": "Middle",

        "final": "Final", "fully evolved": "Final", "final form": "Final",
        "stage 2": "Final", "third stage": "Final", "ultimate": "Final"
    }

    for keyword, stage_name in evolution_keywords.items():
        if keyword in prompt_lower:
            evolution_idx = get_evolution_stage_idx(stage_name)
            break

    habitat_idx = get_habitat_idx("Forest")
    habitat_keywords = {
        "forest": "Forest", "woods": "Forest", "woodland": "Forest", "jungle": "Forest",

        "mountain": "Mountain", "mountains": "Mountain", "peak": "Mountain",
        "rocky": "Mountain", "highland": "Mountain",

        "cave": "Cave", "cavern": "Cave", "underground": "Cave",

        "ocean": "Sea", "sea": "Sea", "marine": "Sea",
        "lake": "Wetland", "river": "Wetland", "pond": "Wetland", "wetland": "Wetland",

        "urban": "Urban", "city": "Urban", "town": "Urban", "building": "Urban",

        "grassland": "Grassland", "plains": "Grassland", "meadow": "Grassland",
        "prairie": "Grassland", "field": "Grassland",

        "desert": "Desert", "arid": "Desert", "sand": "Desert", "dunes": "Desert",

        "tundra": "Tundra", "ice": "Tundra", "frozen": "Tundra", "arctic": "Tundra"
    }

    for keyword, habitat_name in habitat_keywords.items():
        if keyword in prompt_lower:
            habitat_idx = get_habitat_idx(habitat_name)
            break

    legendary = 1 if "legendary" in prompt_lower else 0
    mythical = 1 if "mythical" in prompt_lower else 0

    return {
        "type1": type1_idx,
        "type2": type2_idx,
        "primary_color": primary_color_idx,
        "secondary_color": secondary_color_idx,
        "shape": shape_idx,
        "size": size_idx,
        "evolution_stage": evolution_idx,
        "habitat": habitat_idx,
        "legendary": legendary,
        "mythical": mythical
    }


def generate_pokemon_image(prompt: str, num_samples: int = 1) -> np.ndarray:
    """
    Generate Pokemon image from text prompt.

    Args:
        prompt: Text description
        num_samples: Number of variations to generate

    Returns:
        RGB image array in [0, 255] with white background
    """
    global model, device

    if model is None:
        load_model()

    assert model is not None, "Model failed to load"
    assert device is not None, "Device not initialized"

    text_embedding = encode_text_to_embedding(prompt, device=str(device))
    text_embedding = text_embedding.unsqueeze(0)

    attrs = parse_prompt_attributes(prompt)

    type1 = torch.tensor([attrs["type1"]], dtype=torch.long, device=device)
    type2 = torch.tensor([attrs["type2"]], dtype=torch.long, device=device)
    primary_color = torch.tensor([attrs["primary_color"]], dtype=torch.long, device=device)
    secondary_color = torch.tensor([attrs["secondary_color"]], dtype=torch.long, device=device)
    shape = torch.tensor([attrs["shape"]], dtype=torch.long, device=device)
    size = torch.tensor([attrs["size"]], dtype=torch.long, device=device)
    evolution_stage = torch.tensor([attrs["evolution_stage"]], dtype=torch.long, device=device)
    habitat = torch.tensor([attrs["habitat"]], dtype=torch.long, device=device)
    legendary = torch.tensor([attrs["legendary"]], dtype=torch.long, device=device)
    mythical = torch.tensor([attrs["mythical"]], dtype=torch.long, device=device)

    condition = model.prepare_condition(
        text_embedding, type1, type2, primary_color, secondary_color,
        shape, size, evolution_stage, habitat, legendary, mythical
    )

    with torch.no_grad():
        generated = model.generate(condition, num_samples=num_samples)

    generated_images = generated.view(num_samples, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    generated_images = generated_images.cpu().numpy()

    generated_images = (generated_images + 1.0) / 2.0
    generated_images = np.clip(generated_images, 0, 1)

    for i in range(num_samples):
        mask = np.all(generated_images[i] < 0.1, axis=0)
        for c in range(3):
            generated_images[i, c][mask] = 1.0

    generated_images = (generated_images * 255).astype(np.uint8)
    generated_images = np.transpose(generated_images, (0, 2, 3, 1))

    if num_samples == 1:
        return generated_images[0]
    else:
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        img_size = config.IMAGE_SIZE
        grid = np.zeros((grid_size * img_size, grid_size * img_size, 3), dtype=np.uint8) + 255

        for idx in range(num_samples):
            row = idx // grid_size
            col = idx % grid_size
            grid[row*img_size:(row+1)*img_size, col*img_size:(col+1)*img_size] = generated_images[idx]

        return grid


def numpy_to_base64_png(image: np.ndarray, scale_factor: int = 8) -> str:
    """
    Convert numpy array to base64 encoded PNG.

    Args:
        image: RGB image array
        scale_factor: Upscaling factor

    Returns:
        Base64 encoded PNG string
    """
    pil_image = Image.fromarray(image, mode='RGB')

    new_size = (pil_image.width * scale_factor, pil_image.height * scale_factor)
    pil_image = pil_image.resize(new_size, Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return f"data:image/png;base64,{image_base64}"


async def generate_progressive_images(prompt: str, num_steps: int = 5):
    """
    Generate Pokemon progressively by sampling at different noise levels.

    Yields:
        JSON strings with base64 encoded images
    """
    global model, device

    if model is None:
        load_model()

    assert model is not None, "Model failed to load"
    assert device is not None, "Device not initialized"

    text_embedding = encode_text_to_embedding(prompt, device=str(device))
    text_embedding = text_embedding.unsqueeze(0)

    attrs = parse_prompt_attributes(prompt)

    type1 = torch.tensor([attrs["type1"]], dtype=torch.long, device=device)
    type2 = torch.tensor([attrs["type2"]], dtype=torch.long, device=device)
    primary_color = torch.tensor([attrs["primary_color"]], dtype=torch.long, device=device)
    secondary_color = torch.tensor([attrs["secondary_color"]], dtype=torch.long, device=device)
    shape = torch.tensor([attrs["shape"]], dtype=torch.long, device=device)
    size = torch.tensor([attrs["size"]], dtype=torch.long, device=device)
    evolution_stage = torch.tensor([attrs["evolution_stage"]], dtype=torch.long, device=device)
    habitat = torch.tensor([attrs["habitat"]], dtype=torch.long, device=device)
    legendary = torch.tensor([attrs["legendary"]], dtype=torch.long, device=device)
    mythical = torch.tensor([attrs["mythical"]], dtype=torch.long, device=device)

    condition = model.prepare_condition(
        text_embedding, type1, type2, primary_color, secondary_color,
        shape, size, evolution_stage, habitat, legendary, mythical
    )

    with torch.no_grad():
        z = torch.randn(1, config.LATENT_DIM, device=device)

        noise_scales = np.linspace(0.8, 0.0, num_steps)

        for step, noise_scale in enumerate(noise_scales):
            if noise_scale > 0:
                noise = torch.randn_like(z) * noise_scale
                z_noisy = z + noise
            else:
                z_noisy = z

            generated = model.decoder(z_noisy, condition)

            generated_image = generated.view(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            generated_image = generated_image.cpu().numpy()

            generated_image = (generated_image + 1.0) / 2.0
            generated_image = np.clip(generated_image, 0, 1)

            mask = np.all(generated_image[0] < 0.1, axis=0)
            for c in range(3):
                generated_image[0, c][mask] = 1.0

            generated_image = (generated_image * 255).astype(np.uint8)
            generated_image = np.transpose(generated_image[0], (1, 2, 0))

            image_base64 = numpy_to_base64_png(generated_image, scale_factor=8)

            data = {
                "step": step + 1,
                "total_steps": num_steps,
                "image": image_base64,
                "is_final": step == num_steps - 1
            }

            yield f"data: {json.dumps(data)}\n\n"

            await asyncio.sleep(0.3)


@router.get("/generate/stream")
async def generate_stream(prompt: str, num_steps: int = 5):
    """
    Stream progressive Pokemon generation.

    Args:
        prompt: Text description
        num_steps: Number of evolution steps

    Returns:
        Server-Sent Events stream with progressive images
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if num_steps < 2 or num_steps > 10:
        raise HTTPException(status_code=400, detail="num_steps must be between 2 and 10")

    return StreamingResponse(
        generate_progressive_images(prompt, num_steps),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a Pokemon image from a text prompt."""
    try:
        if model is None:
            load_model()

        pokemon_image = generate_pokemon_image(request.prompt, request.num_samples)

        image_base64 = numpy_to_base64_png(pokemon_image, scale_factor=8)

        return GenerateResponse(
            success=True,
            image=image_base64,
            prompt=request.prompt,
            message=f"Generated {request.num_samples} Pokemon variation(s)"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Pokemon: {str(e)}"
        )


@router.get("/download")
async def download_pokemon(prompt: str, num_samples: int = 1):
    """
    Generate and download a Pokemon image as PNG file.

    Args:
        prompt: Text description
        num_samples: Number of variations

    Returns:
        PNG file download
    """
    try:
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        if num_samples < 1 or num_samples > 4:
            raise HTTPException(status_code=400, detail="num_samples must be between 1 and 4")

        if model is None:
            load_model()

        pokemon_image = generate_pokemon_image(prompt, num_samples)

        pil_image = Image.fromarray(pokemon_image, mode='RGB')

        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)

        filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
        filename = filename.replace(' ', '_')[:50]
        filename = f"pokemon_{filename}.png"

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Pokemon: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Check if the model is loaded and ready."""
    try:
        if model is None:
            load_model()

        return {
            "status": "ready",
            "model_loaded": True,
            "device": str(device),
            "message": "Pokemon generator is ready to create images!"
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "message": str(e)
        }
