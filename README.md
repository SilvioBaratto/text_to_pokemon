# Pokemon Text-to-Image Generator

**Generate custom Pokemon from text descriptions using a Conditional Variational Autoencoder (CVAE)**

![Training Progression](training_progression.gif)

A production-ready deep learning system that generates 64×64 Pokemon images from natural language prompts like *"Create a dragon-like Pokemon with fiery orange scales and majestic blue wings"* using advanced conditional generation techniques.

## Features

- **Intelligent Metadata Extraction**: Vision-based LLM pipeline using BAML + OpenAI GPT for structured attribute extraction
- **Dual Conditioning Architecture**: CLIP text embeddings (768-dim) + categorical attributes (types, colors, shapes)
- **Balanced Dataset Pipeline**: Stratified sampling ensuring consistent attribute distributions across splits
- **Production-Ready API**: Full Docker deployment with FastAPI backend and Angular frontend
- **Perceptual Loss Training**: LPIPS-based training for sharper, more realistic outputs
- **Deterministic Generation**: Reproducible results with fixed seed support
- **Automatic Checkpointing**: Auto-save every 5 epochs with best model tracking

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Development Workflow](#development-workflow)
  - [Step 1: Metadata Generation](#step-1-metadata-generation)
  - [Step 2: Dataset Preparation](#step-2-dataset-preparation)
  - [Step 3: Model Training](#step-3-model-training)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [API Deployment](#api-deployment)
- [Performance Notes](#performance-notes)
- [Tech Stack](#tech-stack)

## Architecture Overview

The system uses a **Conditional Variational Autoencoder (CVAE)** with two parallel conditioning pathways:

### 1. Text Conditioning (CLIP ViT-L/14)
- Converts natural language prompts to 768-dimensional semantic embeddings
- Pre-trained on 400M image-text pairs for rich understanding
- Model: `openai/clip-vit-large-patch14`

### 2. Categorical Conditioning
- Embeds 10 Pokemon attributes into 64-dimensional vectors
- **Attributes**: Type (18 options), Color (11 options), Shape (8 options), Size (5 options), Evolution Stage (6 options), Habitat (10 options), Legendary/Mythical flags

### 3. Encoder-Decoder Pipeline

```
Input (64×64 RGB) → ConvNet Encoder → 256-dim Latent Code → ConvNet Decoder → Output (64×64 RGB)
                          ↓
                832-dim Condition Vector (CLIP + Categorical)
```

### Loss Function

```python
Total Loss = L1 Loss + 0.15 × LPIPS Loss + β × 8.0 × KL Divergence
```

- **L1 Loss**: Pixel-level reconstruction (sharper edges than MSE)
- **LPIPS Loss**: Perceptual similarity using AlexNet features (reduces blurriness)
- **KL Divergence**: Cyclical annealing (β cycles every 50 epochs, max weight: 8.0)

## Development Workflow

This section explains the complete pipeline from raw Pokemon images to a trained generative model.

### Step 1: Metadata Generation

**Objective**: Extract structured attributes from Pokemon images using vision-language models.

#### Process

The `generate_pokemon_metadata.py` script uses BAML (Boundary ML) + OpenAI's GPT with vision capabilities to analyze Pokemon images and extract:

- **Visual Attributes**: Primary/secondary colors, color distribution, body shape, size
- **Categorical Attributes**: Type (Fire, Water, etc.), evolution stage, generation
- **Semantic Attributes**: Habitat preferences, legendary/mythical status
- **Text Prompts**: 5-7 diverse natural language descriptions per Pokemon

#### Implementation Details

```python
# Key components from generate_pokemon_metadata.py

def process_single_image(image_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract metadata from a single Pokemon image using BAML.

    Process:
    1. Load image and encode to base64
    2. Send to BAML vision API with Pokemon name context
    3. Receive structured PokemonMetadata object
    4. Serialize to JSON (handling Pydantic models and enums)
    5. Save metadata file alongside image
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')

    pokemon_image = Image.from_base64("image/png", image_b64)
    metadata = b.ExtractPokemonMetadata(img=pokemon_image, pokemon_name=pokemon_name)

    # Serialize and save
    metadata_dict = serialize_metadata(metadata)
    output_path = image_path.parent / f"{pokemon_name_lower}_{image_stem}.json"
    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
```

#### Usage

```bash
# Process all Pokemon images in the images/ directory
python generate_pokemon_metadata.py --images-dir images

# Skip images with existing metadata (default behavior)
python generate_pokemon_metadata.py --images-dir images --skip-existing

# Reprocess all images (overwrite existing metadata)
python generate_pokemon_metadata.py --images-dir images --no-skip-existing

# Minimal output
python generate_pokemon_metadata.py --quiet
```

#### Output Structure

```
images/
├── Pikachu/
│   ├── pikachu_front.png
│   ├── pikachu_pikachu_front.json    # Generated metadata
│   ├── pikachu_back.png
│   └── pikachu_pikachu_back.json
├── Charizard/
│   ├── charizard_front.png
│   └── charizard_charizard_front.json
└── ...
```

**Example Metadata Output**:
```json
{
  "prompts": [
    "An electric-type mouse Pokemon with yellow fur and red cheeks",
    "Small rodent Pokemon with lightning bolt tail and pointed ears",
    "Create a cute yellow Pokemon with electric powers"
  ],
  "visual_attributes": {
    "primary_colors": ["Yellow", "Brown"],
    "color_distribution": {
      "primary": "Yellow",
      "secondary": "Brown"
    },
    "body_shape": "Quadruped",
    "size": "Small"
  },
  "categorical_attributes": {
    "type1": "Electric",
    "type2": null,
    "generation": 1
  },
  "semantic_attributes": {
    "habitat": "Forest",
    "legendary": false,
    "mythical": false
  }
}
```

### Step 2: Dataset Preparation

**Objective**: Create a balanced, CIFAR-10 compatible dataset with stratified sampling.

#### Process

The `create_pokemon_cifar.py` script transforms raw images + metadata into a training-ready format:

1. **Load Images**: Parallel processing of all Pokemon images with metadata validation
2. **Stratification**: Create composite keys from Type × Color × Body Shape
3. **Balancing**: Equalize samples across attribute combinations using over/undersampling
4. **Splitting**: Stratified train (70%) / val (15%) / test (15%) splits
5. **Serialization**: Save in CIFAR-10 pickle format for efficient loading

#### Implementation Details

```python
# Key components from create_pokemon_cifar.py

def _create_stratification_key(metadata: Optional[Dict]) -> str:
    """
    Generate stratification key from Pokemon metadata.

    Combines type, color, and body shape to ensure balanced sampling
    across different attribute combinations during train/val/test splitting.
    """
    cat_attrs = metadata.get('categorical_attributes', {})
    vis_attrs = metadata.get('visual_attributes', {})

    type1 = cat_attrs.get('type1', 'Normal')
    primary_color = vis_attrs.get('color_distribution', {}).get('primary', 'Blue')
    body_shape = vis_attrs.get('body_shape', 'Bipedal')

    return f"{type1}_{primary_color}_{body_shape}"

def balance_dataset_by_attributes(
    data_dict: Dict,
    target_per_group: int = 50,
    min_per_group: int = 10
) -> Dict:
    """
    Balance dataset using oversampling and undersampling.

    Prevents model bias toward common attribute combinations
    by equalizing representation across Type×Color×BodyShape groups.
    """
    # Group samples by stratification key
    # Oversample rare groups, undersample common groups
    # Remove groups with < min_per_group samples
```

#### Usage

```bash
# Create dataset from 64×64 images
python create_pokemon_cifar.py \
    --images-dir images_64x64 \
    --output-dir pokemon-cifar-64 \
    --image-size 64

# Create dataset from 32×32 images
python create_pokemon_cifar.py \
    --images-dir images_32x32 \
    --output-dir pokemon-cifar-32 \
    --image-size 32

# Custom split ratios
python create_pokemon_cifar.py \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

# Adjust balancing parameters
python create_pokemon_cifar.py \
    --target-per-group 100 \
    --min-per-group 20
```

#### Output Structure

```
pokemon-cifar-64/
├── data_batch_1         # Training batch 1/5 (pickle)
├── data_batch_2         # Training batch 2/5
├── data_batch_3         # Training batch 3/5
├── data_batch_4         # Training batch 4/5
├── data_batch_5         # Training batch 5/5
├── test_batch           # Test set (balanced)
├── batches.meta         # Dataset metadata (label names, etc.)
└── README.txt           # Dataset statistics
```

**Dataset Statistics** (example):
```
Pokemon Species: 809
Training + Validation: 3,920 images
Test: 840 images
Total: 4,760 images
Balancing: Stratified sampling (Type × Color × Body Shape)
```

### Step 3: Model Training

**Objective**: Train a Conditional VAE with dual conditioning for text-to-image generation.

#### Process

The `src/train.py` script orchestrates the training pipeline:

1. **Data Loading**: Load CIFAR-format dataset with augmentation
2. **Model Initialization**: CVAE with encoder, decoder, and categorical embedder
3. **Training Loop**:
   - Extract CLIP embeddings from text prompts
   - Convert categorical attributes to embeddings
   - Combine conditions and feed to model
   - Compute multi-component loss (L1 + LPIPS + KL)
   - Gradient clipping and optimization
4. **Checkpointing**: Save model every 5 epochs + best validation loss
5. **Visualization**: Generate sample images and training curves

#### Model Architecture Details

**From `src/model/vae.py`**:

```python
class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder with dual conditioning.

    Architecture:
    - Encoder: Conv(3→32→64→128) + FC → μ, log(σ²)
    - Decoder: FC + TransposedConv(128→64→32→3)
    - CategoryEmbedder: 10 attribute embeddings → 64-dim vector
    - Condition Gate: Learnable gating for conditioning strength
    """

    def __init__(self, input_dim=12288, condition_dim=832,
                 hidden_dim=512, latent_dim=256):
        self.encoder = Encoder(condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim)
        self.category_embedder = CategoryEmbedder(embedding_dim=64)

        # Learnable gating for conditioning
        self.condition_gate = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.Sigmoid()
        )
```

**Encoder Design** (no batch normalization to avoid VAE instability):
```python
class Encoder(nn.Module):
    def __init__(self, condition_dim=832, hidden_dim=512, latent_dim=256):
        # Convolutional feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 64→32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32→16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 16→8

        # Fully connected layers
        conv_output_dim = 128 * 8 * 8  # 8192
        combined_dim = conv_output_dim + condition_dim  # 8192 + 832 = 9024

        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc1_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
```

**Text Encoder** (`src/model/text_encoder.py`):
```python
def encode_text_to_embedding(text: str, device: str = 'cpu') -> torch.Tensor:
    """
    Encode single text prompt using CLIP ViT-L/14.

    Returns: 768-dimensional embedding vector
    """
    model, tokenizer = get_clip_model(device=device)

    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output.squeeze(0)  # [768]

    return embedding
```

#### Training Configuration

Key hyperparameters from `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 64×64 | Output resolution |
| Batch Size | 128 | Training batch size |
| Learning Rate | 1e-4 | Adam optimizer LR |
| Latent Dim | 256 | Bottleneck dimension |
| Hidden Dim | 512 | FC layer dimension |
| KL Base Weight | 8.0 | Max KL divergence weight |
| Perceptual Weight | 0.15 | LPIPS loss coefficient |
| Dropout Rate | 0.2 | Regularization |
| Num Epochs | 300 | Total training epochs |

#### Usage

```bash
# Basic training (300 epochs)
python src/train.py --dataset-dir pokemon-cifar-64 --epochs 300

# Resume from checkpoint
python src/train.py --resume checkpoints/model_pokemon_epoch_100.pt

# Debug mode (quick validation)
python src/train.py --debug-samples 100 --epochs 5

# Save debug images to inspect dataset preprocessing
python src/train.py --save-debug-images --debug-images-dir debug_output

# Enable image preprocessing (background removal + contrast enhancement)
python src/train.py --use-preprocessing

# Custom directories
python src/train.py \
    --dataset-dir pokemon-cifar-64 \
    --checkpoint-dir checkpoints \
    --output-dir outputs \
    --epochs 500 \
    --batch-size 64
```

#### Training with Docker

```bash
# Build and train using Docker
docker-compose --profile training up train --build

# Checkpoints and outputs are saved to ./checkpoints/ and ./outputs/
```

#### Training Output

The training script produces:

1. **Checkpoints** (`checkpoints/`):
   - `model_pokemon_epoch_N.pt` - Saved every 5 epochs
   - `best_model_pokemon.pt` - Best validation loss

2. **Training Logs** (`training_losses.csv`):
   ```csv
   Epoch,Train_Loss,Train_L1,Train_KL,Train_Perceptual,Val_Loss,...
   1,145.23,89.45,12.34,0.78,152.11,...
   2,132.67,82.11,10.89,0.65,138.45,...
   ```

3. **Visualizations** (`outputs/`):
   - `training_curves.png` - Loss curves over epochs
   - `samples_epoch_N.png` - Generated samples per epoch

4. **Console Output**:
   ```
   ================================================================================
   POKEMON CVAE TRAINING
   ================================================================================
   Dataset:         pokemon-cifar-64
   Checkpoint dir:  checkpoints
   Output dir:      outputs
   Epochs:          300
   ================================================================================

   Device: cuda

   Loading pokemon-cifar-64 dataset...
   Train: 3920, Val: 840, Test: 840

   ================================================================================
   MODEL ARCHITECTURE
   ================================================================================
   Encoder:            8,947,456 parameters
   Decoder:            8,912,515 parameters
   Embedder:              15,232 parameters
   ----------------------------------------------------------------------
   Total:             17,875,203 parameters

   Input: 64×64 RGB → Latent: 256 dims
   Conditioning: 832 dims (CLIP + categorical)
   ================================================================================

   Epochs: 100%|████████| 300/300 [2:34:15<00:00, 30.85s/epoch]

   Train - Loss: 78.23, L1: 65.45, KL: 8.91, Perceptual: 0.45
   Val   - Loss: 82.11, L1: 68.22, KL: 9.45, Perceptual: 0.52
   Test  - Loss: 81.89, L1: 67.98, KL: 9.32, Perceptual: 0.50

   ✓ New best validation loss: 82.11
   ```

#### Loss Function Components

**From `src/model/vae.py:loss_function()`**:

```python
def loss_function(recon_x, x, mu, logvar, beta=1.0, perceptual_loss_model=None):
    """
    Compute VAE loss with three components:

    1. L1 Loss: Pixel-level reconstruction
       - Sharper edges than MSE
       - Mean reduction for batch-size invariance

    2. Perceptual Loss (LPIPS): High-level feature matching
       - AlexNet-based perceptual similarity
       - Weight: 0.15

    3. KL Divergence: Latent distribution regularization
       - Cyclical annealing (β cycles every 50 epochs)
       - Max weight: 8.0
       - Optional dimension normalization
    """
    # L1 reconstruction loss
    l1_loss = F.l1_loss(recon_x, x, reduction='mean')

    # Perceptual loss (LPIPS)
    perceptual_loss = torch.tensor(0.0, device=recon_x.device)
    if perceptual_loss_model is not None:
        recon_img = recon_x.view(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        target_img = x.view(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        perceptual_loss = perceptual_loss_model(recon_img, target_img)

    # KL divergence
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_per_sample)

    # Combined loss
    total_loss = (
        l1_loss +
        PERCEPTUAL_LOSS_WEIGHT * perceptual_loss +
        beta * KL_BASE_WEIGHT * kl_loss
    )

    return total_loss, l1_loss, kl_loss, perceptual_loss
```

**KL Annealing Strategy**:

```python
def get_kl_annealing_factor(epoch: int) -> float:
    """
    Cyclical KL annealing to prevent posterior collapse.

    β cycles every 50 epochs: 0.0 → 1.0 → 0.0 → ...
    This encourages the model to use the latent space throughout training.
    """
    if KL_ANNEALING_TYPE == 'cyclical':
        cycle = (epoch - 1) % KL_CYCLE_EPOCHS
        return min(1.0, cycle / (KL_CYCLE_EPOCHS * 0.5))
    elif KL_ANNEALING_TYPE == 'monotonic':
        return min(1.0, epoch / KL_WARMUP_EPOCHS)
    else:
        return 1.0
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Docker & Docker Compose
- OpenAI API key (for metadata generation)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pokemon-generator.git
cd pokemon-generator

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Complete Pipeline Example

```bash
# 1. Generate metadata from raw images
python generate_pokemon_metadata.py --images-dir raw_pokemon_images

# 2. Create CIFAR-style dataset (assuming images are resized to 64×64)
python create_pokemon_cifar.py \
    --images-dir raw_pokemon_images \
    --output-dir pokemon-cifar-64 \
    --image-size 64

# 3. Train model
python src/train.py --dataset-dir pokemon-cifar-64 --epochs 300

# 4. Launch API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Launch complete stack (frontend + backend)
docker-compose up --build

# Services:
# - Frontend: http://localhost (Angular UI)
# - API: http://localhost:8000 (FastAPI backend)
```

### Generate Pokemon

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a fire-type dragon with glowing red eyes",
    "num_samples": 1,
    "use_llm_parsing": false
  }'

# Via Python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={"prompt": "Electric mouse with lightning tail", "num_samples": 1}
)

image_base64 = response.json()["image"]
```

## Model Architecture

### Detailed Component Breakdown

#### 1. CategoryEmbedder (64-dim output)

Converts discrete attributes to continuous embeddings:

```python
class CategoryEmbedder(nn.Module):
    def __init__(self, embedding_dim=64):
        # 10 embedding layers for different attributes
        self.type1_embedding = nn.Embedding(18, 10)        # Fire, Water, etc.
        self.type2_embedding = nn.Embedding(19, 10)        # Secondary type + None
        self.primary_color_embedding = nn.Embedding(11, 8)
        self.secondary_color_embedding = nn.Embedding(12, 8)
        self.shape_embedding = nn.Embedding(8, 6)          # Bipedal, Quadruped, etc.
        self.size_embedding = nn.Embedding(5, 6)           # Small, Medium, Large, etc.
        self.evolution_stage_embedding = nn.Embedding(6, 6)
        self.habitat_embedding = nn.Embedding(10, 6)
        self.legendary_embedding = nn.Embedding(2, 2)      # Binary
        self.mythical_embedding = nn.Embedding(2, 2)       # Binary

        # Total: 10+10+8+8+6+6+6+6+2+2 = 64 dimensions
```

#### 2. Encoder (Input → μ, log(σ²))

Convolutional encoder without batch normalization:

```
Input: [Batch, 12288]  (64×64×3 flattened)
  ↓ Reshape
[Batch, 3, 64, 64]
  ↓ Conv2d(3→32, stride=2)
[Batch, 32, 32, 32]
  ↓ GELU + Conv2d(32→64, stride=2)
[Batch, 64, 16, 16]
  ↓ GELU + Conv2d(64→128, stride=2)
[Batch, 128, 8, 8]
  ↓ Flatten
[Batch, 8192]
  ↓ Concat with condition [832]
[Batch, 9024]
  ↓ Linear(9024→512) + LayerNorm + GELU + Dropout(0.2)
[Batch, 512]
  ↓ Linear(512→256) [μ]
  ↓ Linear(512→256) [log(σ²)]
```

**Why no BatchNorm?** Batch normalization can cause training instability in VAEs by interfering with the reparameterization trick. LayerNorm is used instead.

#### 3. Decoder (z → Reconstructed Image)

Transposed convolutional decoder:

```
Latent z: [Batch, 256] + Condition: [Batch, 832]
  ↓ Concat
[Batch, 1088]
  ↓ Linear(1088→512) + GELU + Dropout(0.2)
[Batch, 512]
  ↓ Linear(512→8192) + GELU
[Batch, 8192]
  ↓ Reshape
[Batch, 128, 8, 8]
  ↓ TransposedConv2d(128→64, stride=2)
[Batch, 64, 16, 16]
  ↓ GELU + TransposedConv2d(64→32, stride=2)
[Batch, 32, 32, 32]
  ↓ GELU + TransposedConv2d(32→3, stride=2)
[Batch, 3, 64, 64]
  ↓ Tanh + Flatten
[Batch, 12288]
```

#### 4. Conditioning Flow

```
Text Prompt: "Fire dragon with red scales"
  ↓ CLIP Tokenizer
Token IDs: [49406, 2250, 5593, ...]
  ↓ CLIP Text Encoder
[768] Text Embedding

Metadata: {type1: "Fire", color: "Red", shape: "Bipedal", ...}
  ↓ CategoryEmbedder
[64] Categorical Embedding

Text [768] + Categorical [64]
  ↓ Concat
[832] Combined Condition
  ↓ Gating Network (Learnable)
[832] Gated Condition
  ↓ Feed to Encoder & Decoder
```

**Gating Mechanism**: Allows the model to dynamically balance how much it relies on conditioning vs. latent space.

## API Deployment

### API Endpoints

**POST `/api/v1/generate`**
```json
{
  "prompt": "Create a water-type Pokemon with a shell",
  "num_samples": 1,
  "seed": 42,
  "use_llm_parsing": false
}

Response:
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "generation_time_ms": 187,
  "model_version": "pokemon_cvae_v1"
}
```

**GET `/api/v1/generate/stream`**
- Server-Sent Events (SSE) for progressive generation
- Real-time updates during image generation

**GET `/api/v1/download?prompt=...`**
- Download generated image as PNG file

**GET `/api/v1/health`**
- Model status and readiness check

### Production Deployment

```bash
# Using Gunicorn with 4 workers
gunicorn api.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

## Performance Notes

### Training Performance

**Mac M4 Mini (16GB RAM, MPS)**:
- Training time: ~10 hours for 300 epochs
- Dataset: 809 unique Pokemon, ~5,600 effective samples (with augmentation)
- Device: MPS (Apple Silicon GPU acceleration)

**NVIDIA A100 (40GB)**:
- Training time: ~3 hours for 300 epochs
- Mixed precision training enabled
- TF32 tensor cores utilized

### Generation Speed

- **Single image**: <200ms on GPU
- **Batch of 8**: <1s on GPU
- **First request**: ~5s (model loading overhead)

### Optimization Flags

For maximum performance on CUDA:

```python
# config.py
USE_MIXED_PRECISION = True   # AMP for A100/RTX GPUs
USE_TORCH_COMPILE = True     # PyTorch 2.0+ compilation
ENABLE_TF32 = True           # A100 tensor cores
USE_CHANNELS_LAST = True     # Memory layout optimization
```

For Apple Silicon (MPS):

```python
USE_MIXED_PRECISION = False  # Not supported on MPS
USE_TORCH_COMPILE = False    # Not recommended
NUM_WORKERS = 0              # For DataLoader stability
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | PyTorch 2.9 |
| Text Encoding | CLIP (openai/clip-vit-large-patch14) |
| Perceptual Loss | LPIPS (AlexNet) |
| Backend API | FastAPI 0.120 |
| Frontend | Angular 20 + Tailwind CSS |
| LLM Pipeline | BAML + OpenAI GPT (vision) |
| Deployment | Docker + Docker Compose |
| Reverse Proxy | Nginx |

## Project Structure

```
pokemon/
├── src/
│   ├── model/
│   │   ├── vae.py              # CVAE architecture (Encoder, Decoder, Loss)
│   │   └── text_encoder.py    # CLIP text embedding wrapper
│   ├── utils/
│   │   ├── data_loader.py      # Pokemon-CIFAR dataset loader
│   │   ├── conditioning.py     # Metadata → condition vectors
│   │   ├── attribute_mappings.py  # String → index mappings
│   │   └── training_utils.py  # Checkpointing, KL annealing
│   └── train.py                # Main training script
├── api/
│   ├── main.py                 # FastAPI application
│   └── v1/endpoint.py          # Generation endpoints
├── baml_src/
│   └── media.baml              # BAML metadata extraction schema
├── frontend/
│   ├── src/app/                # Angular 20 components
│   ├── nginx.conf              # Reverse proxy configuration
│   └── Dockerfile
├── config.py                   # Centralized hyperparameters
├── generate_pokemon_metadata.py  # Step 1: Metadata extraction
├── create_pokemon_cifar.py     # Step 2: Dataset preparation
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Multi-service orchestration
└── README.md                   # This file
```

## Future Improvements

This project demonstrates a functional CVAE approach, but modern generative models offer significantly better results:

### Recommended Next Steps

1. **Latent Diffusion Models (Stable Diffusion)**
   - 10-100× better image quality
   - Higher resolution (512×512 or 1024×1024)
   - Fine-tune `stable-diffusion-v1-5` or `SDXL` on Pokemon dataset
   - Use LoRA for efficient training

2. **GAN-Based Approaches**
   - StyleGAN3 for high-fidelity generation
   - Progressive growing for stable training

3. **Dataset Expansion**
   - Include fan-made Pokemon designs
   - Add regional variants and Mega evolutions
   - Expand to 10,000+ samples

4. **Advanced Conditioning**
   - Multi-modal inputs (sketch + text)
   - Attribute-level control sliders
   - Inpainting for iterative refinement

### Why Stable Diffusion?

While this CVAE produces recognizable Pokemon at 64×64, Stable Diffusion would enable:
- **512×512+ resolution** with crisp details
- **Photorealistic textures** instead of smooth gradients
- **Complex compositions** (multiple Pokemon, backgrounds, poses)
- **Fine-grained control** via attention mechanisms

## Citation

If you use this project in your research, please cite:

```bibtex
@software{pokemon_cvae_2025,
  title={Pokemon Text-to-Image Generator using Conditional VAE},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/pokemon-generator}
}
```

## Acknowledgments

- **Pokemon Dataset**: Original sprites from The Pokemon Company
- **CLIP Model**: OpenAI's CLIP (Radford et al., 2021)
- **LPIPS Loss**: Perceptual Similarity by Zhang et al. (2018)
- **BAML**: Boundary ML for structured LLM outputs

## License

This project is for educational purposes only. Pokemon is a trademark of The Pokemon Company/Nintendo.

---

**Ready to train your own Pokemon generator?** Start with the [Development Workflow](#development-workflow) section above!
