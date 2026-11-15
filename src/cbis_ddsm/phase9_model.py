"""Frozen V2S features + global average pooling + binary linear head."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torchvision.models import efficientnet_v2_s

from .config import ConfigurationError
from .training import sha256_file

WEIGHT_SHA256 = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"


@dataclass(frozen=True)
class Phase9ModelConfig:
    input_size: int = 384
    backbone: str = "efficientnet_v2_s"
    pretrained: bool = True
    freeze_backbone: bool = False
    pooling: str = "global_average"
    output_logits: int = 1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Phase9ModelConfig":
        if set(values) != set(cls.__dataclass_fields__):
            raise ConfigurationError("Phase 9 requires exactly the frozen CNN-only model keys.")
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if (self.input_size, self.backbone, self.freeze_backbone, self.pooling,
                self.output_logits) != (384, "efficientnet_v2_s", False, "global_average", 1):
            raise ConfigurationError("Phase 9 locks V2S, 384, full fine-tuning, GAP and one logit.")
        if type(self.pretrained) is not bool or type(self.freeze_backbone) is not bool:
            raise ConfigurationError("Pretrained/freeze flags must be booleans.")


def pretrained_weight_metadata() -> dict[str, Any]:
    path = Path(torch.hub.get_dir()) / "checkpoints" / "efficientnet_v2_s-dd5fe13b.pth"
    if not path.is_file() or sha256_file(path) != WEIGHT_SHA256:
        raise ValueError("Cached official V2S weights missing or hash mismatch; no download allowed.")
    return {"enum": "EfficientNet_V2_S_Weights.DEFAULT", "path": str(path),
            "sha256": WEIGHT_SHA256, "size_bytes": path.stat().st_size,
            "url": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"}


class Phase9CNNOnly(nn.Module):
    def __init__(self, config: Phase9ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        network = efficientnet_v2_s(weights=None)
        if config.pretrained:
            metadata = pretrained_weight_metadata()
            state = torch.load(metadata["path"], map_location="cpu", weights_only=True)
            network.load_state_dict(state, strict=True)
        self.backbone = network.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(1280, 1)

    def forward_feature_map(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 384, 384):
            raise ValueError("Expected [batch, 3, 384, 384] input.")
        return self.backbone(images)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.flatten(self.pool(self.forward_feature_map(images)))).squeeze(-1)


def build_phase9_model(config: Phase9ModelConfig) -> Phase9CNNOnly:
    return Phase9CNNOnly(config)
