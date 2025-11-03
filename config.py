"""
Configuration for Pokemon Conditional VAE.

This module defines model architecture, training hyperparameters, and system
settings for the text-to-image Pokemon generation system.
"""

import torch
from typing import Literal


# Model Architecture
LATENT_DIM = 256
HIDDEN_DIM = 512
INPUT_DIM = 12288
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
DROPOUT_RATE = 0.0


# Text and Attribute Conditioning
TEXT_EMBEDDING_DIM = 768
CATEGORICAL_EMBEDDING_DIM = 64
TOTAL_CONDITION_DIM = TEXT_EMBEDDING_DIM + CATEGORICAL_EMBEDDING_DIM
LABEL_DIM = TOTAL_CONDITION_DIM

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

NUM_TYPES = 18
NUM_COLORS = 11
NUM_SHAPES = 8
NUM_SIZES = 5
NUM_EVOLUTION_STAGES = 6
NUM_HABITATS = 10


# Training Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_VALUE = 1.0

MIN_LEARNING_RATE = 1e-6
LR_SCHEDULE: Literal["cosine_with_warmup"] = "cosine_with_warmup"
WARMUP_EPOCHS = 10


# KL Divergence Annealing
KL_ANNEALING_TYPE: Literal["cyclical", "linear", "monotonic"] = "cyclical"
KL_CYCLE_EPOCHS = 50
KL_WARMUP_EPOCHS = 0
KL_WEIGHT = 8.0


# Perceptual Loss
USE_PERCEPTUAL_LOSS = True
PERCEPTUAL_LOSS_WEIGHT = 0.15


# Performance Optimization
USE_MIXED_PRECISION = False
USE_TORCH_COMPILE = False
USE_CHANNELS_LAST = False
ENABLE_TF32 = False
PERSISTENT_WORKERS = True
NUM_WORKERS = 2
NON_BLOCKING_TRANSFER = True


# Directory Paths
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"


# Logging and Checkpointing
SAVE_INTERVAL = 5
TEST_INTERVAL = 5
LOG_INTERVAL = 100
EARLY_STOP_PATIENCE = 20


# Device Selection
def get_device() -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device configured for available hardware
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


DEVICE = get_device()
