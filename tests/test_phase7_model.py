from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.phase7_model import (
    PHASE7_ENDPOINT_CHANNELS,
    PHASE7_ENDPOINT_INDICES,
    PHASE7_MAX_PARAMETERS,
    Phase7ModelConfig,
    build_phase7_model,
    count_phase7_parameters,
)


def offline_config(**overrides: object) -> Phase7ModelConfig:
    return replace(Phase7ModelConfig(pretrained=False), **overrides)


@pytest.mark.parametrize(
    ("bifpn", "repeats", "cbam", "expected_parameters"),
    [
        (False, 0, False, 33_444_433),
        (True, 1, False, 33_358_175),
        (False, 0, True, 33_477_843),
        (True, 1, True, 33_391_585),
    ],
)
def test_phase7_parameter_ceiling_and_exact_counts(
    bifpn: bool,
    repeats: int,
    cbam: bool,
    expected_parameters: int,
) -> None:
    model = build_phase7_model(
        offline_config(
            bifpn_enabled=bifpn,
            bifpn_repeats=repeats,
            cbam_enabled=cbam,
        )
    )
    total, trainable = count_phase7_parameters(model)
    assert total == trainable == expected_parameters
    assert total <= PHASE7_MAX_PARAMETERS


def test_phase7_locked_v2s_endpoints_tokens_and_backward() -> None:
    torch.manual_seed(42)
    model = build_phase7_model(
        offline_config(bifpn_enabled=True, bifpn_repeats=1, cbam_enabled=True)
    ).train()
    images = torch.randn(1, 3, 384, 384)
    feature_map, endpoints = model.forward_feature_map(images)
    assert model.endpoint_indices == PHASE7_ENDPOINT_INDICES
    assert [feature.shape[1] for feature in endpoints] == list(
        PHASE7_ENDPOINT_CHANNELS
    )
    assert [tuple(feature.shape[-2:]) for feature in endpoints] == [
        (96, 96),
        (48, 48),
        (24, 24),
        (12, 12),
    ]
    assert feature_map.shape == (1, 512, 12, 12)
    _, tokens = model.forward_tokens(images)
    assert tokens.shape == (1, 144, 512)
    logits = model(images)
    assert logits.shape == (1,)
    loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.84])
    )(logits, torch.tensor([1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_phase7_baseline_uses_only_v2s_p5_projection() -> None:
    model = build_phase7_model(offline_config())
    assert model.bifpn is None
    assert model.token_compressor is None
    assert model.cbam is None
    assert model.channel_projection is not None
    assert model.channel_projection.in_channels == 1280
    assert model.channel_projection.out_channels == 512


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_size", 512),
        ("backbone", "efficientnet_b0"),
        ("freeze_backbone", True),
        ("projection_dim", 256),
        ("bifpn_channels", 256),
        ("bifpn_repeats", 2),
        ("token_grid_size", 24),
        ("cbam_reduction", 8),
        ("cbam_spatial_kernel", 3),
    ],
)
def test_out_of_scope_phase7_model_values_are_rejected(
    field: str,
    value: object,
) -> None:
    config = offline_config(bifpn_enabled=True, bifpn_repeats=1)
    with pytest.raises(ConfigurationError):
        replace(config, **{field: value}).validate()


def test_approved_v2s_cbam_without_bifpn_uses_the_p5_projection() -> None:
    model = build_phase7_model(offline_config(cbam_enabled=True))
    assert model.bifpn is None
    assert model.channel_projection is not None
    assert model.cbam is not None
