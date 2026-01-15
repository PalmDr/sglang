# SPDX-License-Identifier: Apache-2.0
"""DiT configuration for NVIDIA Cosmos Predict 2.5 world models.

Cosmos uses a 3D Diffusion Transformer (DiT) architecture for video denoising.
The transformer consists of interleaved self-attention, cross-attention,
and feedforward layers.

References:
- https://github.com/nvidia-cosmos/cosmos-predict2.5
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
"""

from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_transformer_block(n: str, m) -> bool:
    """Check if the module is a transformer block."""
    return "transformer_blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class CosmosTransformerArchConfig(DiTArchConfig):
    """Architecture configuration for Cosmos 3D Transformer."""

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_transformer_block]
    )

    _compile_conditions: list = field(
        default_factory=lambda: [is_transformer_block]
    )

    # Cosmos Transformer 3D model parameters (2B variant)
    patch_size: int = 2
    patch_size_t: int = 1
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 28  # 2B model has 28 layers
    mlp_ratio: float = 4.0
    text_embed_dim: int = 4096  # T5-11B text embedding dimension
    dtype: torch.dtype | None = None

    # Additional Cosmos-specific parameters
    rope_axes_dim: tuple[int, int, int] = (16, 56, 56)
    guidance_embeds: bool = True
    qk_norm: str = "rms_norm"

    exclude_lora_layers: list[str] = field(
        default_factory=lambda: ["x_embedder", "time_embed", "context_embedder"]
    )

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.attention_head_dim * self.num_attention_heads
        self.num_channels_latents = self.in_channels


@dataclass
class CosmosTransformer14BArchConfig(CosmosTransformerArchConfig):
    """Architecture configuration for Cosmos 14B variant."""

    num_attention_heads: int = 48
    attention_head_dim: int = 128
    num_layers: int = 48  # 14B model has 48 layers


@dataclass
class CosmosTransformerConfig(DiTConfig):
    """DiT configuration for Cosmos 2B model."""

    arch_config: DiTArchConfig = field(default_factory=CosmosTransformerArchConfig)
    prefix: str = "Cosmos"


@dataclass
class CosmosTransformer14BConfig(DiTConfig):
    """DiT configuration for Cosmos 14B model."""

    arch_config: DiTArchConfig = field(default_factory=CosmosTransformer14BArchConfig)
    prefix: str = "Cosmos14B"
