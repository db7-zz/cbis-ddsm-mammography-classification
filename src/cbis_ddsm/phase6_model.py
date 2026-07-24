"""Approved phase 6 EfficientNet-B0, BiFPN, CBAM, and Transformer models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .config import ConfigurationError
from .model import SinusoidalPositionEncoding2D


PHASE6_MAX_PARAMETERS = 18_000_000
PHASE6_ENDPOINT_INDICES = (2, 3, 5, 8)
PHASE6_ENDPOINT_CHANNELS = (24, 40, 112, 1280)


@dataclass(frozen=True)
class Phase6ModelConfig:
    """Locked architecture settings for the approved phase 6 ablations."""

    input_size: int = 384
    backbone: str = "efficientnet_b0"
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
    def from_mapping(cls, values: Mapping[str, Any]) -> "Phase6ModelConfig":
        known = set(cls.__dataclass_fields__)
        unexpected = sorted(set(values) - known)
        if unexpected:
            raise ConfigurationError(
                f"Unexpected phase 6 model keys: {', '.join(unexpected)}"
            )
        config = cls(**dict(values))
        config.validate()
        return config

    def validate(self) -> None:
        if self.input_size != 384:
            raise ConfigurationError("Phase 6 locks the input size to 384.")
        if self.backbone != "efficientnet_b0":
            raise ConfigurationError("Phase 6 locks the backbone to efficientnet_b0.")
        if not isinstance(self.pretrained, bool):
            raise ConfigurationError("model.pretrained must be a boolean.")
        if self.freeze_backbone:
            raise ConfigurationError("Phase 6 fine-tunes the full B0 backbone.")
        if self.projection_dim != 512:
            raise ConfigurationError("Phase 6 locks projection_dim to 512.")
        if self.transformer_layers != 4 or self.attention_heads != 8:
            raise ConfigurationError("Phase 6 locks the Transformer to four layers/eight heads.")
        if self.feedforward_dim != 2048:
            raise ConfigurationError("Phase 6 locks feedforward_dim to 2048.")
        if not math.isclose(self.dropout, 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError("Phase 6 locks Transformer dropout to 0.1.")
        if self.activation != "gelu":
            raise ConfigurationError("Phase 6 locks Transformer activation to GELU.")
        if self.positional_encoding != "sinusoidal_2d":
            raise ConfigurationError("Phase 6 requires sinusoidal_2d position encoding.")
        if self.pooling != "mean" or self.output_logits != 1:
            raise ConfigurationError("Phase 6 requires mean pooling and one output logit.")
        if self.bifpn_channels != 128:
            raise ConfigurationError("Phase 6 locks BiFPN width to 128 channels.")
        if self.bifpn_enabled and self.bifpn_repeats not in {1, 2}:
            raise ConfigurationError("Enabled BiFPN repeats must be 1 or 2.")
        if not self.bifpn_enabled and self.bifpn_repeats != 0:
            raise ConfigurationError("Disabled BiFPN requires bifpn_repeats=0.")
        if self.token_grid_size != 12:
            raise ConfigurationError("Phase 6 locks the token grid to 12x12.")
        if self.cbam_reduction != 16 or self.cbam_spatial_kernel != 7:
            raise ConfigurationError("Phase 6 locks CBAM reduction=16 and kernel=7.")


class ConvNormActivation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class WeightedFeatureFusion(nn.Module):
    """Fast normalized non-negative fusion used by each BiFPN node."""

    def __init__(self, inputs: int, epsilon: float = 1e-4) -> None:
        super().__init__()
        if inputs < 2:
            raise ValueError("Weighted fusion requires at least two inputs.")
        self.weights = nn.Parameter(torch.ones(inputs, dtype=torch.float32))
        self.epsilon = float(epsilon)

    def normalized_weights(self) -> Tensor:
        weights = F.relu(self.weights)
        return weights / (weights.sum() + self.epsilon)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != self.weights.numel():
            raise ValueError("Feature count does not match fusion weight count.")
        reference_shape = features[0].shape
        if any(feature.shape != reference_shape for feature in features[1:]):
            raise ValueError("All weighted fusion inputs must have the same shape.")
        weights = self.normalized_weights().to(
            device=features[0].device,
            dtype=features[0].dtype,
        )
        return sum(weight * feature for weight, feature in zip(weights, features))


class SeparableConvBlock(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )


class BiFPNLayer(nn.Module):
    """One exact top-down and bottom-up phase 6 BiFPN repeat."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.td4_fusion = WeightedFeatureFusion(2)
        self.td3_fusion = WeightedFeatureFusion(2)
        self.td2_fusion = WeightedFeatureFusion(2)
        self.out3_fusion = WeightedFeatureFusion(3)
        self.out4_fusion = WeightedFeatureFusion(3)
        self.out5_fusion = WeightedFeatureFusion(2)
        self.td4_conv = SeparableConvBlock(channels)
        self.td3_conv = SeparableConvBlock(channels)
        self.td2_conv = SeparableConvBlock(channels)
        self.out3_conv = SeparableConvBlock(channels)
        self.out4_conv = SeparableConvBlock(channels)
        self.out5_conv = SeparableConvBlock(channels)
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @staticmethod
    def _upsample(feature: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(feature, size=target.shape[-2:], mode="nearest")

    def forward(
        self,
        inputs: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        i2, i3, i4, i5 = inputs
        t5 = i5
        t4 = self.td4_conv(self.td4_fusion((i4, self._upsample(t5, i4))))
        t3 = self.td3_conv(self.td3_fusion((i3, self._upsample(t4, i3))))
        t2 = self.td2_conv(self.td2_fusion((i2, self._upsample(t3, i2))))
        o2 = t2
        o3 = self.out3_conv(
            self.out3_fusion((i3, t3, self.downsample(o2)))
        )
        o4 = self.out4_conv(
            self.out4_fusion((i4, t4, self.downsample(o3)))
        )
        o5 = self.out5_conv(self.out5_fusion((i5, self.downsample(o4))))
        return o2, o3, o4, o5


class BiFPN(nn.Module):
    def __init__(
        self,
        input_channels: Sequence[int],
        channels: int,
        repeats: int,
    ) -> None:
        super().__init__()
        if tuple(input_channels) != PHASE6_ENDPOINT_CHANNELS:
            raise ValueError("Phase 6 BiFPN input channels do not match locked endpoints.")
        if repeats not in {1, 2}:
            raise ValueError("Phase 6 BiFPN repeats must be 1 or 2.")
        self.lateral = nn.ModuleList(
            ConvNormActivation(in_channels, channels)
            for in_channels in input_channels
        )
        self.layers = nn.ModuleList(BiFPNLayer(channels) for _ in range(repeats))

    def forward(
        self,
        features: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        projected = tuple(
            projection(feature)
            for projection, feature in zip(self.lateral, features)
        )
        outputs = projected
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


class MultiScaleTokenCompressor(nn.Module):
    """Pool four 128-channel levels and mix them into a 512-channel token map."""

    def __init__(self, channels: int = 128, grid_size: int = 12) -> None:
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.mixer = ConvNormActivation(channels * 4, 512)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != 4:
            raise ValueError("Token compressor requires exactly four feature levels.")
        pooled = []
        for feature in features:
            height, width = feature.shape[-2:]
            if (
                height % self.grid_size != 0
                or width % self.grid_size != 0
            ):
                raise ValueError("Feature sizes must divide the locked token grid exactly.")
            kernel = (height // self.grid_size, width // self.grid_size)
            pooled.append(F.avg_pool2d(feature, kernel_size=kernel, stride=kernel))
        return self.mixer(torch.cat(pooled, dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int = 512, reduction: int = 16) -> None:
        super().__init__()
        hidden = channels // reduction
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, feature: Tensor) -> Tensor:
        average = self.shared_mlp(feature.mean(dim=(-2, -1), keepdim=True))
        maximum = self.shared_mlp(feature.amax(dim=(-2, -1), keepdim=True))
        return feature * torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, feature: Tensor) -> Tensor:
        average = feature.mean(dim=1, keepdim=True)
        maximum = feature.amax(dim=1, keepdim=True)
        return feature * torch.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class CBAM(nn.Module):
    def __init__(
        self,
        channels: int = 512,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, feature: Tensor) -> Tensor:
        return self.spatial_attention(self.channel_attention(feature))


class Phase6EfficientNetTransformerClassifier(nn.Module):
    """B0 classifier with independently switchable approved BiFPN and CBAM."""

    def __init__(self, config: Phase6ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
        efficientnet = efficientnet_b0(weights=weights)
        self.backbone = efficientnet.features
        self.endpoint_indices = PHASE6_ENDPOINT_INDICES

        self.bifpn: BiFPN | None
        self.token_compressor: MultiScaleTokenCompressor | None
        self.channel_projection: nn.Conv2d | None
        if config.bifpn_enabled:
            self.bifpn = BiFPN(
                PHASE6_ENDPOINT_CHANNELS,
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
                PHASE6_ENDPOINT_CHANNELS[-1],
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
            raise RuntimeError("EfficientNet-B0 did not produce all locked endpoints.")
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
                raise RuntimeError("Baseline channel projection is missing.")
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
                f"Phase 6 feature map shape mismatch: {tuple(feature_map.shape)}."
            )
        return feature_map, endpoints

    def forward_tokens(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
            raise RuntimeError("Transformer output shape does not match phase 6 tokens.")
        return feature_map, encoded

    def forward(self, images: Tensor) -> Tensor:
        _, tokens = self.forward_tokens(images)
        pooled = tokens.mean(dim=1)
        return self.classifier(self.head_norm(pooled)).squeeze(-1)


def build_phase6_model(
    config: Phase6ModelConfig | Mapping[str, Any],
) -> Phase6EfficientNetTransformerClassifier:
    if not isinstance(config, Phase6ModelConfig):
        config = Phase6ModelConfig.from_mapping(config)
    model = Phase6EfficientNetTransformerClassifier(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    if total > PHASE6_MAX_PARAMETERS:
        raise ConfigurationError(
            f"Phase 6 model has {total:,} parameters, above {PHASE6_MAX_PARAMETERS:,}."
        )
    return model


def count_phase6_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
