# SPDX-License-Identifier: Apache-2.0
"""VAE configuration for NVIDIA Cosmos Predict 2.5 world models.

Cosmos uses AutoencoderKLCosmos which encodes videos into a compact latent
representation and decodes latents back to videos.

References:
- https://github.com/nvidia-cosmos/cosmos-predict2.5
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
"""

from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class CosmosVAEArchConfig(VAEArchConfig):
    """Architecture configuration for Cosmos VAE (AutoencoderKLCosmos)."""

    # Cosmos VAE compression ratios
    temporal_compression_ratio: int = 8  # 8x temporal compression
    spatial_compression_ratio: int = 8   # 8x spatial compression (each dimension)

    # Scaling factor for latent space
    scaling_factor: float = 0.18215

    # VAE architecture parameters
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)


@dataclass
class CosmosVAEConfig(VAEConfig):
    """VAE configuration for Cosmos models."""

    arch_config: VAEArchConfig = field(default_factory=CosmosVAEArchConfig)

    # Only need decoder for generation (encoder only needed for Image2World)
    load_encoder: bool = False
    load_decoder: bool = True

    # Tiling parameters for efficient memory usage
    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 17
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12

    use_tiling: bool = True
    use_temporal_tiling: bool = True
    use_parallel_tiling: bool = True
    use_temporal_scaling_frames: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        )


@dataclass
class CosmosVAEEncoderConfig(CosmosVAEConfig):
    """VAE configuration for Cosmos Image2World (requires encoder)."""

    load_encoder: bool = True
    load_decoder: bool = True
