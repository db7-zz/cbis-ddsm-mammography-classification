"""Approved phase 7 EfficientNet-V2S migration models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from .config import ConfigurationError
from .model import SinusoidalPositionEncoding2D
from .phase6_model import BiFPNLayer, CBAM, ConvNormActivation, MultiScaleTokenCompressor


PHASE7_MAX_PARAMETERS = 36_000_000
PHASE7_ENDPOINT_INDICES = (2, 3, 5, 7)
PHASE7_ENDPOINT_CHANNELS = (48, 64, 160, 1280)


@dataclass(frozen=True)
class Phase7ModelConfig:
    """Locked architecture settings for the approved phase 7 migration."""

    input_size: int = 384
    backbone: str = "efficientnet_v2_s"
    pretrained: bool = True
    freeze_backbone: bool = False
    projection_dim: int = 512
    transformer_layers: int = 4
    attention_heads: int = 8
    feedforward_dim: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    positional_encoding: str = "sinusoidal_2d"
    pooling: str = "mean"
    output_logits: int = 1
    bifpn_enabled: bool = False
    bifpn_channels: int = 128
    bifpn_repeats: int = 0
    token_grid_size: int = 12
    cbam_enabled: bool = False
    cbam_reduction: int = 16
    cbam_spatial_kernel: int = 7

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Phase7ModelConfig":
        known = set(cls.__dataclass_fields__)
        unexpected = sorted(set(values) - known)
        if unexpected:
            raise ConfigurationError(
                f"Unexpected phase 7 model keys: {', '.join(unexpected)}"
            )
        config = cls(**dict(values))
        config.validate()
        return config

    def validate(self) -> None:
        if self.input_size != 384:
            raise ConfigurationError("Phase 7 locks the input size to 384.")
        if self.backbone != "efficientnet_v2_s":
            raise ConfigurationError("Phase 7 locks the backbone to efficientnet_v2_s.")
        if not isinstance(self.pretrained, bool):
            raise ConfigurationError("model.pretrained must be a boolean.")
        if self.freeze_backbone:
            raise ConfigurationError("Phase 7 fine-tunes the full V2S backbone.")
        if self.projection_dim != 512:
            raise ConfigurationError("Phase 7 locks projection_dim to 512.")
        if self.transformer_layers != 4 or self.attention_heads != 8:
            raise ConfigurationError("Phase 7 locks the Transformer to four layers/eight heads.")
        if self.feedforward_dim != 2048:
            raise ConfigurationError("Phase 7 locks feedforward_dim to 2048.")
        if not math.isclose(self.dropout, 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError("Phase 7 locks Transformer dropout to 0.1.")
        if self.activation != "gelu":
            raise ConfigurationError("Phase 7 locks Transformer activation to GELU.")
        if self.positional_encoding != "sinusoidal_2d":
            raise ConfigurationError("Phase 7 requires sinusoidal_2d position encoding.")
        if self.pooling != "mean" or self.output_logits != 1:
            raise ConfigurationError("Phase 7 requires mean pooling and one output logit.")
        if self.bifpn_channels != 128:
            raise ConfigurationError("Phase 7 locks BiFPN width to 128 channels.")
        if self.bifpn_enabled and self.bifpn_repeats != 1:
            raise ConfigurationError("Phase 7 locks enabled BiFPN to one repeat.")
        if not self.bifpn_enabled and self.bifpn_repeats != 0:
            raise ConfigurationError("Disabled BiFPN requires bifpn_repeats=0.")
        if self.token_grid_size != 12:
            raise ConfigurationError("Phase 7 locks the token grid to 12x12.")
        if self.cbam_reduction != 16 or self.cbam_spatial_kernel != 7:
            raise ConfigurationError("Phase 7 locks CBAM reduction=16 and kernel=7.")


class Phase7BiFPN(nn.Module):
    """Frozen phase 6 C128/R1 BiFPN adapted only to V2S endpoint channels."""

    def __init__(
        self,
        input_channels: Sequence[int] = PHASE7_ENDPOINT_CHANNELS,
        channels: int = 128,
        repeats: int = 1,
    ) -> None:
        super().__init__()
        if tuple(input_channels) != PHASE7_ENDPOINT_CHANNELS:
            raise ValueError("Phase 7 BiFPN input channels do not match locked endpoints.")
        if channels != 128 or repeats != 1:
            raise ValueError("Phase 7 BiFPN is locked to C128/R1.")
        self.lateral = nn.ModuleList(
            ConvNormActivation(in_channels, channels)
            for in_channels in input_channels
        )
        self.layers = nn.ModuleList(BiFPNLayer(channels) for _ in range(repeats))

    def forward(
        self,
        features: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        outputs = tuple(
            projection(feature)
            for projection, feature in zip(self.lateral, features)
        )
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


class Phase7EfficientNetTransformerClassifier(nn.Module):
    """V2S classifier with the approved frozen BiFPN and CBAM migrations."""

    def __init__(self, config: Phase7ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        weights = EfficientNet_V2_S_Weights.DEFAULT if config.pretrained else None
        efficientnet = efficientnet_v2_s(weights=weights)
        self.backbone = efficientnet.features
        self.endpoint_indices = PHASE7_ENDPOINT_INDICES

        self.bifpn: Phase7BiFPN | None
        self.token_compressor: MultiScaleTokenCompressor | None
        self.channel_projection: nn.Conv2d | None
        if config.bifpn_enabled:
            self.bifpn = Phase7BiFPN(
                PHASE7_ENDPOINT_CHANNELS,
                config.bifpn_channels,
                config.bifpn_repeats,
            )
            self.token_compressor = MultiScaleTokenCompressor(
                config.bifpn_channels,
                config.token_grid_size,
            )
            self.channel_projection = None
        else:
            self.bifpn = None
            self.token_compressor = None
            self.channel_projection = nn.Conv2d(
                PHASE7_ENDPOINT_CHANNELS[-1],
                config.projection_dim,
                kernel_size=1,
                bias=True,
            )

        self.cbam = (
            CBAM(
                config.projection_dim,
                config.cbam_reduction,
                config.cbam_spatial_kernel,
            )
            if config.cbam_enabled
            else None
        )
        self.position_encoding = SinusoidalPositionEncoding2D(config.projection_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.projection_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
        )
        self.head_norm = nn.LayerNorm(config.projection_dim)
        self.classifier = nn.Linear(config.projection_dim, config.output_logits)

    def _validate_input(self, images: Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [batch, 3, height, width].")
        expected = (self.config.input_size, self.config.input_size)
        if tuple(images.shape[-2:]) != expected:
            raise ValueError(
                f"Expected spatial input size {expected}, got {tuple(images.shape[-2:])}."
            )

    def extract_backbone_features(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._validate_input(images)
        outputs: list[Tensor] = []
        feature = images
        for index, stage in enumerate(self.backbone):
            feature = stage(feature)
            if index in self.endpoint_indices:
                outputs.append(feature)
        if len(outputs) != 4:
            raise RuntimeError("EfficientNet-V2S did not produce all locked endpoints.")
        return tuple(outputs)  # type: ignore[return-value]

    def forward_feature_map(
        self,
        images: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        endpoints = self.extract_backbone_features(images)
        if self.bifpn is not None:
            if self.token_compressor is None:
                raise RuntimeError("BiFPN token compressor is missing.")
            feature_map = self.token_compressor(self.bifpn(endpoints))
        else:
            if self.channel_projection is None:
                raise RuntimeError("V2S channel projection is missing.")
            feature_map = self.channel_projection(endpoints[-1])
        if self.cbam is not None:
            feature_map = self.cbam(feature_map)
        expected = (
            images.shape[0],
            self.config.projection_dim,
            self.config.token_grid_size,
            self.config.token_grid_size,
        )
        if tuple(feature_map.shape) != expected:
            raise RuntimeError(
                f"Phase 7 feature map shape mismatch: {tuple(feature_map.shape)}."
            )
        return feature_map, endpoints

    def forward_tokens(self, images: Tensor) -> tuple[Tensor, Tensor]:
        feature_map, _ = self.forward_feature_map(images)
        batch, channels, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        positions = self.position_encoding(
            height,
            width,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        encoded = self.transformer(tokens + positions)
        if encoded.shape != (batch, height * width, channels):
            raise RuntimeError("Transformer output shape does not match phase 7 tokens.")
        return feature_map, encoded

    def forward(self, images: Tensor) -> Tensor:
        _, tokens = self.forward_tokens(images)
        pooled = tokens.mean(dim=1)
        return self.classifier(self.head_norm(pooled)).squeeze(-1)


def build_phase7_model(
    config: Phase7ModelConfig | Mapping[str, Any],
) -> Phase7EfficientNetTransformerClassifier:
    if not isinstance(config, Phase7ModelConfig):
        config = Phase7ModelConfig.from_mapping(config)
    model = Phase7EfficientNetTransformerClassifier(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    if total > PHASE7_MAX_PARAMETERS:
        raise ConfigurationError(
            f"Phase 7 model has {total:,} parameters, above {PHASE7_MAX_PARAMETERS:,}."
        )
    return model


def count_phase7_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
