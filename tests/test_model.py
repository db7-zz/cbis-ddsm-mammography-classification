from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.model import (
    ModelConfig,
    SinusoidalPositionEncoding2D,
    build_model,
    count_parameters,
    load_model_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def offline_config(**overrides: object) -> ModelConfig:
    config = load_model_config(PROJECT_ROOT / "configs" / "model.yaml")
    return replace(config, pretrained=False, **overrides)


def test_locked_model_config_matches_phase2_specification() -> None:
    config = load_model_config(PROJECT_ROOT / "configs" / "model.yaml")

    assert config.input_size == 256
    assert config.backbone == "efficientnet_b0"
    assert config.pretrained is True
    assert config.projection_dim == 512
    assert config.transformer_layers == 4
    assert config.attention_heads == 8
    assert config.output_logits == 1


def test_2d_position_encoding_has_one_vector_per_spatial_token() -> None:
    encoding = SinusoidalPositionEncoding2D(512)
    first = encoding(8, 8, device=torch.device("cpu"), dtype=torch.float32)
    second = encoding(8, 8, device=torch.device("cpu"), dtype=torch.float32)

    assert first.shape == (1, 64, 512)
    assert torch.equal(first, second)
    assert not torch.equal(first[:, 0], first[:, 1])
    assert not any(parameter.requires_grad for parameter in encoding.parameters())


def test_model_structure_forward_and_bce_backward() -> None:
    torch.manual_seed(42)
    config = offline_config()
    model = build_model(config)
    model.train()

    assert model.backbone_out_channels == 1280
    assert model.channel_projection.in_channels == 1280
    assert model.channel_projection.out_channels == 512
    assert len(model.transformer.layers) == 4
    assert all(layer.self_attn.num_heads == 8 for layer in model.transformer.layers)
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.out_features == 1

    observed: dict[str, tuple[int, ...]] = {}
    hooks = [
        model.backbone.register_forward_hook(
            lambda _module, _inputs, output: observed.__setitem__(
                "backbone", tuple(output.shape)
            )
        ),
        model.channel_projection.register_forward_hook(
            lambda _module, _inputs, output: observed.__setitem__(
                "projected", tuple(output.shape)
            )
        ),
        model.transformer.register_forward_hook(
            lambda _module, _inputs, output: observed.__setitem__(
                "tokens", tuple(output.shape)
            )
        ),
    ]
    images = torch.randn(1, 3, 256, 256)
    logits = model(images)
    for hook in hooks:
        hook.remove()

    assert observed["backbone"] == (1, 1280, 8, 8)
    assert observed["projected"] == (1, 512, 8, 8)
    assert observed["tokens"] == (1, 64, 512)
    assert logits.shape == (1,)

    labels = torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.84]))
    loss = criterion(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()

    total, trainable = count_parameters(model)
    assert total == trainable
    assert total > 10_000_000


def test_model_returns_one_logit_per_sample_in_eval_mode() -> None:
    model = build_model(offline_config()).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 256, 256))
    assert logits.shape == (2,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_size", 224),
        ("backbone", "efficientnet_b3"),
        ("freeze_backbone", True),
        ("projection_dim", 256),
        ("transformer_layers", 2),
        ("attention_heads", 4),
        ("feedforward_dim", 1024),
        ("dropout", 0.0),
        ("activation", "relu"),
        ("positional_encoding", "learned"),
        ("pooling", "cls_token"),
        ("output_logits", 2),
    ],
)
def test_invalid_or_out_of_scope_model_variants_are_rejected(
    field: str, value: object
) -> None:
    config = offline_config()
    invalid = replace(config, **{field: value})
    with pytest.raises(ConfigurationError):
        invalid.validate()


def test_model_rejects_non_baseline_input_size() -> None:
    model = build_model(offline_config()).eval()
    with pytest.raises(ValueError, match="spatial input size"):
        model(torch.zeros(1, 3, 224, 224))
