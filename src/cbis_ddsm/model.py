"""EfficientNet-B0 plus Transformer model for the locked binary baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .config import ConfigurationError


@dataclass(frozen=True)
class ModelConfig:
    """Auditable model settings loaded from ``configs/model.yaml``."""

    input_size: int = 256
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

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        known = set(cls.__dataclass_fields__)
        unexpected = sorted(set(values) - known)
        if unexpected:
            raise ConfigurationError(
                f"Unexpected model configuration keys: {', '.join(unexpected)}"
            )
        config = cls(**dict(values))
        config.validate()
        return config

    def validate(self) -> None:
        if self.input_size not in {256, 384, 512}:
            raise ConfigurationError(
                "Phase 5 permits only input sizes 256, 384, and 512."
            )
        if self.backbone != "efficientnet_b0":
            raise ConfigurationError("The locked baseline backbone is efficientnet_b0.")
        if not isinstance(self.pretrained, bool):
            raise ConfigurationError("model.pretrained must be a boolean.")
        if self.freeze_backbone:
            raise ConfigurationError("The locked baseline fine-tunes the full backbone.")
        if self.projection_dim != 512:
            raise ConfigurationError("The locked baseline projection dimension is 512.")
        if self.transformer_layers != 4:
            raise ConfigurationError("The locked baseline uses four Transformer layers.")
        if self.attention_heads != 8:
            raise ConfigurationError("The locked baseline uses eight attention heads.")
        if self.feedforward_dim != 2048:
            raise ConfigurationError(
                "The phase 2 reproduction assumption uses feed-forward dimension 2048."
            )
        if not math.isclose(self.dropout, 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError(
                "The phase 2 reproduction assumption uses Transformer dropout 0.1."
            )
        if self.activation != "gelu":
            raise ConfigurationError(
                "The phase 2 reproduction assumption uses GELU activation."
            )
        if self.positional_encoding != "sinusoidal_2d":
            raise ConfigurationError(
                "The phase 2 reproduction assumption is sinusoidal_2d positional encoding."
            )
        if self.pooling != "mean":
            raise ConfigurationError("The phase 2 reproduction assumption is mean pooling.")
        if self.output_logits != 1:
            raise ConfigurationError("Binary classification must output exactly one logit.")


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate the model section of a UTF-8 YAML configuration."""
    import yaml

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Model configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("model"), dict):
        raise ConfigurationError("Model configuration must contain a model mapping.")
    return ModelConfig.from_mapping(document["model"])


class SinusoidalPositionEncoding2D(nn.Module):
    """Parameter-free row/column sine-cosine encoding for spatial tokens."""

    def __init__(self, d_model: int, temperature: float = 10_000.0) -> None:
        super().__init__()
        if d_model <= 0 or d_model % 4 != 0:
            raise ValueError("d_model must be positive and divisible by 4.")
        self.d_model = d_model
        self.temperature = float(temperature)

    def _encode_axis(self, length: int, device: torch.device) -> Tensor:
        axis_dim = self.d_model // 2
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(self.temperature) / axis_dim)
        )
        encoding = torch.zeros(length, axis_dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        return encoding

    def forward(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if height <= 0 or width <= 0:
            raise ValueError("Spatial dimensions must be positive.")
        y_encoding = self._encode_axis(height, device)
        x_encoding = self._encode_axis(width, device)
        y_grid = y_encoding[:, None, :].expand(height, width, -1)
        x_grid = x_encoding[None, :, :].expand(height, width, -1)
        encoding = torch.cat((y_grid, x_grid), dim=-1)
        return encoding.reshape(1, height * width, self.d_model).to(dtype=dtype)


class EfficientNetTransformerClassifier(nn.Module):
    """Project B0 features to spatial tokens and return one logit per image."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
        efficientnet = efficientnet_b0(weights=weights)
        backbone_channels = int(efficientnet.classifier[1].in_features)
        self.backbone = efficientnet.features
        self.backbone_out_channels = backbone_channels
        self.channel_projection = nn.Conv2d(
            backbone_channels,
            config.projection_dim,
            kernel_size=1,
            bias=True,
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

        if config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _validate_input(self, images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError("Expected images with shape [batch, 3, height, width].")
        if images.shape[1] != 3:
            raise ValueError("EfficientNet-B0 expects three input channels.")
        expected = (self.config.input_size, self.config.input_size)
        if tuple(images.shape[-2:]) != expected:
            raise ValueError(
                f"Expected spatial input size {expected}, got {tuple(images.shape[-2:])}."
            )

    def forward_tokens(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return the projected feature map and encoded spatial tokens."""
        self._validate_input(images)
        feature_map = self.backbone(images)
        projected_map = self.channel_projection(feature_map)
        batch_size, channels, height, width = projected_map.shape
        tokens = projected_map.flatten(2).transpose(1, 2)
        positions = self.position_encoding(
            height,
            width,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        if positions.shape != (1, height * width, channels):
            raise RuntimeError("Position encoding shape does not match spatial tokens.")
        encoded_tokens = self.transformer(tokens + positions)
        if encoded_tokens.shape[0] != batch_size:
            raise RuntimeError("Transformer unexpectedly changed the batch dimension.")
        return projected_map, encoded_tokens

    def forward(self, images: Tensor) -> Tensor:
        _, encoded_tokens = self.forward_tokens(images)
        pooled = encoded_tokens.mean(dim=1)
        logits = self.classifier(self.head_norm(pooled))
        return logits.squeeze(-1)


def build_model(config: ModelConfig | Mapping[str, Any]) -> EfficientNetTransformerClassifier:
    """Build the locked model from a validated dataclass or mapping."""
    if not isinstance(config, ModelConfig):
        config = ModelConfig.from_mapping(config)
    return EfficientNetTransformerClassifier(config)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
