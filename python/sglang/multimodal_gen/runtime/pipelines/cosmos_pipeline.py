# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Cosmos Predict 2.5 world model pipeline implementation.

This module implements the Cosmos video diffusion pipeline using the modular
pipeline architecture. Cosmos is a flow-based world foundation model that
unifies Text2World, Image2World, and Video2World capabilities.

References:
- https://github.com/nvidia-cosmos/cosmos-predict2.5
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CosmosText2WorldPipeline(ComposedPipelineBase):
    """Pipeline for NVIDIA Cosmos Text2World generation.

    This pipeline generates videos from text prompts using the Cosmos
    world foundation model. It uses:
    - T5-11B (Cosmos-Reason1) for text encoding
    - CosmosTransformer3DModel for denoising
    - AutoencoderKLCosmos for latent decoding
    - EDMEulerScheduler for sampling

    The pipeline follows the standard diffusion process:
    1. Input validation
    2. Text encoding
    3. Conditioning preparation
    4. Timestep preparation
    5. Latent initialization
    6. Iterative denoising
    7. VAE decoding
    """

    pipeline_name = "CosmosTextToWorldPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # Stage 1: Validate inputs
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )

        # Stage 2: Encode text prompts using T5
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        # Stage 3: Prepare conditioning tensors
        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage()
        )

        # Stage 4: Prepare timesteps for diffusion
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            ),
        )

        # Stage 5: Initialize latent noise
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # Stage 6: Iterative denoising
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Stage 7: Decode latents to video
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae")
            )
        )


class CosmosImage2WorldPipeline(CosmosText2WorldPipeline):
    """Pipeline for NVIDIA Cosmos Image2World generation.

    Generates videos conditioned on both text and an input image.
    The first frame is derived from the input image.
    """

    pipeline_name = "CosmosImageToWorldPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with image conditioning."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages import (
            ImageVAEEncodingStage,
        )

        # Stage 1: Validate inputs
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )

        # Stage 2: Encode text prompts using T5
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        # Stage 3: Encode input image through VAE
        self.add_stage(
            stage_name="image_vae_encoding_stage",
            stage=ImageVAEEncodingStage(
                vae=self.get_module("vae"),
            ),
        )

        # Stage 4: Prepare conditioning tensors
        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage()
        )

        # Stage 5: Prepare timesteps for diffusion
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            ),
        )

        # Stage 6: Initialize latent noise with image conditioning
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # Stage 7: Iterative denoising
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Stage 8: Decode latents to video
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae")
            )
        )


# Entry classes for automatic pipeline discovery
EntryClass = [CosmosText2WorldPipeline, CosmosImage2WorldPipeline]
