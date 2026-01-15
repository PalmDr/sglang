# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for NVIDIA Cosmos Predict 2.5 world models.

Cosmos is a flow-based world foundation model that unifies Text2World,
Image2World, and Video2World into a single model. It uses Cosmos-Reason1
(T5-based) as the text encoder and a 3D transformer for denoising.

References:
- https://github.com/nvidia-cosmos/cosmos-predict2.5
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class CosmosSamplingParams(SamplingParams):
    """Sampling parameters for Cosmos Predict 2.5 Text2World."""

    num_inference_steps: int = 36

    num_frames: int = 121
    height: int = 704
    width: int = 1280
    fps: int = 30

    guidance_scale: float = 7.0

    # Cosmos supported resolutions (16:9 aspect ratio optimized)
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            # Standard resolutions
            (1280, 704),  # 16:9 (default)
            (704, 1280),  # 9:16 (portrait)
            (960, 544),   # 16:9 smaller
            (544, 960),   # 9:16 smaller
            (832, 480),   # 16:9 small
            (480, 832),   # 9:16 small
        ]
    )


@dataclass
class CosmosImage2WorldSamplingParams(CosmosSamplingParams):
    """Sampling parameters for Cosmos Predict 2.5 Image2World."""

    # Image2World typically requires fewer steps for conditioning
    num_inference_steps: int = 30


@dataclass
class CosmosVideo2WorldSamplingParams(CosmosSamplingParams):
    """Sampling parameters for Cosmos Predict 2.5 Video2World."""

    # Video2World uses video conditioning
    num_inference_steps: int = 30
