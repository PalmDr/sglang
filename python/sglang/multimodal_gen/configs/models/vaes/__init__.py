# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig
from sglang.multimodal_gen.configs.models.vaes.cosmosvae import (
    CosmosVAEConfig,
    CosmosVAEEncoderConfig,
)

__all__ = [
    "HunyuanVAEConfig",
    "WanVAEConfig",
    "CosmosVAEConfig",
    "CosmosVAEEncoderConfig",
]
