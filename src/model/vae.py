"""
Conditional Variational Autoencoder for Pokemon Image Generation

Implements a CVAE with dual conditioning (CLIP text embeddings + categorical
attributes) for generating Pokemon images from text descriptions.
"""

import sys
from typing import Optional, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')
import config


class CategoryEmbedder(nn.Module):
    """
    Embeds categorical Pokemon attributes into a fixed-dimensional vector.

    Converts discrete attributes (type, color, shape, etc.) into continuous
    embeddings suitable for conditioning the VAE.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()

        self.type1_embedding = nn.Embedding(config.NUM_TYPES, 10)
        self.type2_embedding = nn.Embedding(config.NUM_TYPES + 1, 10)
        self.primary_color_embedding = nn.Embedding(config.NUM_COLORS, 8)
        self.secondary_color_embedding = nn.Embedding(config.NUM_COLORS + 1, 8)
        self.shape_embedding = nn.Embedding(config.NUM_SHAPES, 6)
        self.size_embedding = nn.Embedding(config.NUM_SIZES, 6)
        self.evolution_stage_embedding = nn.Embedding(config.NUM_EVOLUTION_STAGES, 6)
        self.habitat_embedding = nn.Embedding(config.NUM_HABITATS, 6)
        self.legendary_embedding = nn.Embedding(2, 2)
        self.mythical_embedding = nn.Embedding(2, 2)

        self.embedding_dim = embedding_dim

    def forward(
        self,
        type1: torch.Tensor,
        type2: torch.Tensor,
        primary_color: torch.Tensor,
        secondary_color: torch.Tensor,
        shape: torch.Tensor,
        size: torch.Tensor,
        evolution_stage: torch.Tensor,
        habitat: torch.Tensor,
        legendary: torch.Tensor,
        mythical: torch.Tensor
    ) -> torch.Tensor:
        """
        Embed categorical attributes and concatenate into a single vector.

        Args:
            type1: Primary Pokemon type indices
            type2: Secondary type indices
            primary_color: Primary color indices
            secondary_color: Secondary color indices
            shape: Body shape indices
            size: Size class indices
            evolution_stage: Evolution stage indices
            habitat: Habitat indices
            legendary: Legendary status (binary)
            mythical: Mythical status (binary)

        Returns:
            Combined categorical embedding tensor
        """
        type1_emb = self.type1_embedding(type1)
        type2_emb = self.type2_embedding(type2)
        primary_color_emb = self.primary_color_embedding(primary_color)
        secondary_color_emb = self.secondary_color_embedding(secondary_color)
        shape_emb = self.shape_embedding(shape)
        size_emb = self.size_embedding(size)
        evolution_stage_emb = self.evolution_stage_embedding(evolution_stage)
        habitat_emb = self.habitat_embedding(habitat)
        legendary_emb = self.legendary_embedding(legendary)
        mythical_emb = self.mythical_embedding(mythical)

        return torch.cat([
            type1_emb, type2_emb,
            primary_color_emb, secondary_color_emb,
            shape_emb, size_emb,
            evolution_stage_emb, habitat_emb,
            legendary_emb, mythical_emb
        ], dim=1)


class Encoder(nn.Module):
    """
    Convolutional encoder mapping images and conditions to latent distributions.

    Uses strided convolutions without batch normalization to avoid VAE training
    instability. GELU activation provides smoother gradients.
    """

    def __init__(
        self,
        condition_dim: int = config.TOTAL_CONDITION_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        latent_dim: int = config.LATENT_DIM
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        conv_output_dim = 128 * 8 * 8
        combined_dim = conv_output_dim + condition_dim

        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Initialize mu to output small values close to 0
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.constant_(self.fc_mu.bias, 0.0)

        # Initialize logvar to output small negative values (var ≈ 0.05-0.5)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)
        nn.init.constant_(self.fc_logvar.bias, -3.0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent distribution parameters.

        Args:
            x: Flattened RGB images (batch_size, INPUT_DIM)
            condition: Conditioning vector (batch_size, condition_dim)

        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        batch_size = x.size(0)
        x = x.view(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))

        h = h.view(batch_size, -1)
        combined = torch.cat([h, condition], dim=1)

        h = F.gelu(self.fc1(combined))
        h = self.dropout(h)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    Convolutional decoder reconstructing images from latent codes and conditions.

    Uses transposed convolutions without batch normalization. Simpler architecture
    forces the model to use latent codes effectively.
    """

    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        condition_dim: int = config.TOTAL_CONDITION_DIM,
        hidden_dim: int = config.HIDDEN_DIM
    ):
        super().__init__()
        combined_dim = latent_dim + condition_dim

        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.fc2 = nn.Linear(hidden_dim, 128 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to image.

        Args:
            z: Latent code (batch_size, latent_dim)
            condition: Conditioning vector (batch_size, condition_dim)

        Returns:
            Flattened RGB image (batch_size, INPUT_DIM)
        """
        combined = torch.cat([z, condition], dim=1)

        h = F.gelu(self.fc1(combined))
        h = self.dropout(h)
        h = F.gelu(self.fc2(h))

        batch_size = h.size(0)
        h = h.view(batch_size, 128, 8, 8)

        h = F.gelu(self.deconv1(h))
        h = F.gelu(self.deconv2(h))
        h = torch.tanh(self.deconv3(h))

        return h.view(batch_size, -1)


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder with dual conditioning.

    Combines CLIP text embeddings with categorical Pokemon attributes for
    controlled image generation. Uses learnable gating to balance conditioning
    strength with latent space expressiveness.
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        condition_dim: int = config.TOTAL_CONDITION_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        latent_dim: int = config.LATENT_DIM
    ):
        super().__init__()
        self.encoder = Encoder(condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim)
        self.category_embedder = CategoryEmbedder(embedding_dim=32)

        self.condition_gate = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution: z = μ + σ·ε where ε ~ N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def prepare_condition(
        self,
        text_embedding: torch.Tensor,
        type1: Optional[torch.Tensor] = None,
        type2: Optional[torch.Tensor] = None,
        primary_color: Optional[torch.Tensor] = None,
        secondary_color: Optional[torch.Tensor] = None,
        shape: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
        evolution_stage: Optional[torch.Tensor] = None,
        habitat: Optional[torch.Tensor] = None,
        legendary: Optional[torch.Tensor] = None,
        mythical: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine text embeddings with categorical attributes.

        Args:
            text_embedding: CLIP embeddings (batch_size, 768)
            type1: Primary type indices
            type2: Secondary type indices
            primary_color: Primary color indices
            secondary_color: Secondary color indices
            shape: Body shape indices
            size: Size class indices
            evolution_stage: Evolution stage indices
            habitat: Habitat indices
            legendary: Legendary status (binary)
            mythical: Mythical status (binary)

        Returns:
            Combined conditioning vector (batch_size, condition_dim)
        """
        batch_size = text_embedding.size(0)
        device = text_embedding.device

        if type1 is None:
            type1 = torch.zeros(batch_size, dtype=torch.long, device=device)
        if type2 is None:
            type2 = torch.full((batch_size,), config.NUM_TYPES, dtype=torch.long, device=device)
        if primary_color is None:
            primary_color = torch.zeros(batch_size, dtype=torch.long, device=device)
        if secondary_color is None:
            secondary_color = torch.full((batch_size,), config.NUM_COLORS, dtype=torch.long, device=device)
        if shape is None:
            shape = torch.zeros(batch_size, dtype=torch.long, device=device)
        if size is None:
            size = torch.ones(batch_size, dtype=torch.long, device=device)
        if evolution_stage is None:
            evolution_stage = torch.zeros(batch_size, dtype=torch.long, device=device)
        if habitat is None:
            habitat = torch.zeros(batch_size, dtype=torch.long, device=device)
        if legendary is None:
            legendary = torch.zeros(batch_size, dtype=torch.long, device=device)
        if mythical is None:
            mythical = torch.zeros(batch_size, dtype=torch.long, device=device)

        category_embedding = self.category_embedder(
            type1, type2,
            primary_color, secondary_color,
            shape, size,
            evolution_stage, habitat,
            legendary, mythical
        )

        condition = torch.cat([text_embedding, category_embedding], dim=1)

        gate = self.condition_gate(condition)
        condition_gated = condition * gate

        return condition_gated

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode and reconstruct images.

        Args:
            x: Flattened RGB images (batch_size, INPUT_DIM)
            condition: Conditioning vector (batch_size, condition_dim)

        Returns:
            Tuple of (reconstructed, mu, logvar)
        """
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, condition)
        return reconstructed, mu, logvar

    def generate(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate images by sampling from the prior.

        Args:
            condition: Conditioning vector (batch_size, condition_dim)
            num_samples: Number of samples per condition

        Returns:
            Generated images (batch_size * num_samples, INPUT_DIM)
        """
        device = condition.device
        batch_size = condition.size(0)

        z = torch.randn(batch_size * num_samples, config.LATENT_DIM, device=device)
        condition_repeated = condition.repeat(num_samples, 1)

        with torch.no_grad():
            generated = self.decoder(z, condition_repeated)

        return generated


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using LPIPS (Learned Perceptual Image Patch Similarity).

    Captures high-level feature differences to improve reconstruction quality
    and reduce blurriness in generated images.
    """

    def __init__(self, net: str = 'alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between images.

        Args:
            pred: Predicted images (batch, 3, H, W)
            target: Target images (batch, 3, H, W)

        Returns:
            Scalar loss value
        """
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0

        loss = self.lpips(pred_scaled, target_scaled)
        return loss.mean()


def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    perceptual_loss_model: Optional[PerceptualLoss] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss combining reconstruction, perceptual, and KL terms.

    Uses L1 reconstruction loss for sharper edges, optional perceptual loss for
    high-level feature matching, and KL divergence with dimension-normalized
    annealing for regularization. Uses mean reduction for batch-size invariance.

    Args:
        recon_x: Reconstructed images (batch, INPUT_DIM)
        x: Original images (batch, INPUT_DIM)
        mu: Latent distribution mean (batch, latent_dim)
        logvar: Latent distribution log variance (batch, latent_dim)
        beta: KL annealing factor (0 to 1 during warmup)
        perceptual_loss_model: Optional LPIPS model

    Returns:
        Tuple of (total_loss, l1_loss, kl_loss, perceptual_loss)
    """
    batch_size = recon_x.size(0)

    # L1 Reconstruction Loss (mean reduction for batch-size invariance)
    l1_loss = F.l1_loss(recon_x, x, reduction='mean')

    # Perceptual Loss (LPIPS) - already returns mean
    perceptual_loss = torch.tensor(0.0, device=recon_x.device)
    if perceptual_loss_model is not None:
        recon_img = recon_x.view(batch_size, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
        target_img = x.view(batch_size, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
        perceptual_loss = perceptual_loss_model(recon_img, target_img)

    # KL Divergence (mean reduction for batch-size invariance)
    # Standard VAE formulation: sum over latent dims per sample, then mean over batch
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_per_sample)

    # Compute KL weight with optional dimensionality normalization
    if config.USE_NORMALIZED_KL:
        # Normalize by ratio of latent to input dimensions
        # This accounts for the fact that KL naturally scales with latent_dim
        kl_weight = config.KL_BASE_WEIGHT * (config.LATENT_DIM / config.INPUT_DIM)
    else:
        kl_weight = config.KL_BASE_WEIGHT

    # Total Loss: reconstruction + perceptual + annealed KL
    total_loss = (
        l1_loss +
        config.PERCEPTUAL_LOSS_WEIGHT * perceptual_loss +
        beta * kl_weight * kl_loss
    )

    return total_loss, l1_loss, kl_loss, perceptual_loss
