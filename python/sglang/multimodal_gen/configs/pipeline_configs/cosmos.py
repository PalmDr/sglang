# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for NVIDIA Cosmos Predict 2.5 world models.

Cosmos is a flow-based world foundation model that unifies Text2World,
Image2World, and Video2World capabilities. It uses:
- T5-11B (Cosmos-Reason1) as the text encoder
- CosmosTransformer3DModel for denoising
- AutoencoderKLCosmos for VAE
- EDMEulerScheduler for sampling

References:
- https://github.com/nvidia-cosmos/cosmos-predict2.5
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
"""

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.cosmosvideo import (
    CosmosTransformerConfig,
    CosmosTransformer14BConfig,
)
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, T5Config
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig
from sglang.multimodal_gen.configs.models.vaes.cosmosvae import CosmosVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


# T5-11B configuration for Cosmos-Reason1 text encoder
@dataclass
class CosmosT5ArchConfig(T5ArchConfig):
    """T5-11B architecture config used by Cosmos."""

    vocab_size: int = 32128
    d_model: int = 1024
    d_kv: int = 128
    d_ff: int = 65536
    num_layers: int = 24
    num_decoder_layers: int = 24
    num_heads: int = 128
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    feed_forward_proj: str = "gated-gelu"
    is_encoder_decoder: bool = True
    text_len: int = 512  # Maximum text sequence length


@dataclass
class CosmosT5Config(T5Config):
    """T5 encoder config for Cosmos."""

    arch_config: T5ArchConfig = field(default_factory=CosmosT5ArchConfig)
    prefix: str = "cosmos_t5"


def cosmos_preprocess_text(prompt: str) -> str:
    """Preprocess text for Cosmos T5 encoder."""
    return prompt


def cosmos_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """Extract encoder hidden states for conditioning."""
    # Cosmos uses the last hidden state from T5 encoder
    return outputs.last_hidden_state


@dataclass
class CosmosText2WorldConfig(PipelineConfig):
    """Pipeline configuration for Cosmos Text2World generation."""

    task_type: ModelTaskType = ModelTaskType.T2V

    # DiT configuration
    dit_config: DiTConfig = field(default_factory=CosmosTransformerConfig)

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=CosmosVAEConfig)

    # Denoising parameters
    embedded_cfg_scale: float = 7.0
    flow_shift: float | None = None  # EDM scheduler handles this

    # Text encoder configuration (T5-11B)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (CosmosT5Config(),)
    )

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (cosmos_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (cosmos_postprocess_text,))
    )

    # Precision settings
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16",)
    )

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class CosmosText2World14BConfig(CosmosText2WorldConfig):
    """Pipeline configuration for Cosmos 14B Text2World."""

    dit_config: DiTConfig = field(default_factory=CosmosTransformer14BConfig)


@dataclass
class CosmosImage2WorldConfig(CosmosText2WorldConfig):
    """Pipeline configuration for Cosmos Image2World generation."""

    task_type: ModelTaskType = ModelTaskType.I2V

    def __post_init__(self):
        # Image2World needs VAE encoder for conditioning
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class CosmosVideo2WorldConfig(CosmosText2WorldConfig):
    """Pipeline configuration for Cosmos Video2World generation."""

    task_type: ModelTaskType = ModelTaskType.I2V  # Uses video as conditioning input

    def __post_init__(self):
        # Video2World needs VAE encoder for conditioning
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
