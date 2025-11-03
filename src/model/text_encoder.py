"""
CLIP text encoder for converting natural language prompts to embeddings.

Provides caching and batch processing for efficient text encoding.
"""

import sys
from typing import List, Optional, Tuple

import torch
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append('.')
import config


_clip_model: Optional[CLIPTextModel] = None
_clip_tokenizer: Optional[CLIPTokenizer] = None


def get_clip_model(device: str = 'cpu') -> Tuple[CLIPTextModel, CLIPTokenizer]:
    """
    Load and cache CLIP text encoder.

    Args:
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    global _clip_model, _clip_tokenizer

    if _clip_model is None:
        print(f"Loading CLIP model ({config.CLIP_MODEL_NAME}) on {device}")
        _clip_tokenizer = CLIPTokenizer.from_pretrained(config.CLIP_MODEL_NAME)
        _clip_model = CLIPTextModel.from_pretrained(config.CLIP_MODEL_NAME)
        _clip_model.eval()

        if device != 'cpu':
            _clip_model = _clip_model.to(device) # type: ignore[assignment]
            print(f"CLIP text encoder loaded on {device}")
        else:
            print("CLIP text encoder loaded on CPU")

    assert _clip_model is not None and _clip_tokenizer is not None
    return _clip_model, _clip_tokenizer


def encode_text_to_embedding(text: str, device: str = 'cpu') -> torch.Tensor:
    """
    Encode single text prompt to CLIP embedding.

    Args:
        text: Natural language prompt
        device: Device to place tensor on

    Returns:
        768-dimensional embedding vector
    """
    model, tokenizer = get_clip_model(device=device)

    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output.squeeze(0)

    return embedding


def encode_texts_batch(texts: List[str], device: str = 'cpu') -> torch.Tensor:
    """
    Encode batch of text prompts to CLIP embeddings.

    Args:
        texts: List of text prompts
        device: Device to place tensor on

    Returns:
        Embedding matrix (batch_size, 768)
    """
    model, tokenizer = get_clip_model(device=device)

    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output

    return embeddings
