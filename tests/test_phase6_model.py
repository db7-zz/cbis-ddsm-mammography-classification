from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.model import ModelConfig, build_model
from cbis_ddsm.phase6_model import (
    CBAM,
    PHASE6_ENDPOINT_CHANNELS,
    PHASE6_MAX_PARAMETERS,
    Phase6ModelConfig,
    WeightedFeatureFusion,
    build_phase6_model,
    count_phase6_parameters,
)


def offline_config(**overrides: object) -> Phase6ModelConfig:
    return replace(Phase6ModelConfig(pretrained=False), **overrides)


@pytest.mark.parametrize(
    ("bifpn", "repeats", "cbam"),
    [
        (False, 0, False),
        (True, 1, False),
        (True, 2, False),
        (False, 0, True),
        (True, 1, True),
        (True, 2, True),
    ],
)
def test_phase6_parameter_ceiling(
    bifpn: bool,
    repeats: int,
    cbam: bool,
) -> None:
    model = build_phase6_model(
        offline_config(
            bifpn_enabled=bifpn,
            bifpn_repeats=repeats,
            cbam_enabled=cbam,
        )
    )
    total, trainable = count_phase6_parameters(model)
    assert total == trainable
    assert total <= PHASE6_MAX_PARAMETERS


def test_locked_endpoints_bifpn_tokens_and_backward() -> None:
    torch.manual_seed(42)
    model = build_phase6_model(
        offline_config(bifpn_enabled=True, bifpn_repeats=1, cbam_enabled=True)
    ).train()
    images = torch.randn(1, 3, 384, 384)
    feature_map, endpoints = model.forward_feature_map(images)
    assert [feature.shape[1] for feature in endpoints] == list(
        PHASE6_ENDPOINT_CHANNELS
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


def test_weighted_fusion_is_non_negative_and_normalized() -> None:
    fusion = WeightedFeatureFusion(3)
    with torch.no_grad():
        fusion.weights.copy_(torch.tensor([1.0, -2.0, 3.0]))
    weights = fusion.normalized_weights()
    assert torch.all(weights >= 0)
    assert float(weights.sum().detach()) == pytest.approx(1.0, abs=1e-4)
    features = tuple(torch.full((1, 2, 3, 3), value) for value in (1.0, 2.0, 3.0))
    output = fusion(features)
    assert output.shape == features[0].shape
    assert torch.isfinite(output).all()


def test_cbam_has_one_shared_channel_mlp_and_seven_by_seven_spatial_conv() -> None:
    cbam = CBAM()
    convolutions = [
        module
        for module in cbam.channel_attention.shared_mlp
        if isinstance(module, nn.Conv2d)
    ]
    assert [(module.in_channels, module.out_channels) for module in convolutions] == [
        (512, 32),
        (32, 512),
    ]
    assert cbam.spatial_attention.conv.kernel_size == (7, 7)
    assert cbam.spatial_attention.conv.in_channels == 2
    assert cbam.spatial_attention.conv.out_channels == 1


def test_phase6_no_module_control_matches_phase5_model() -> None:
    torch.manual_seed(7)
    phase5 = build_model(ModelConfig(input_size=384, pretrained=False)).eval()
    phase6 = build_phase6_model(offline_config()).eval()
    phase6.load_state_dict(phase5.state_dict(), strict=True)
    images = torch.randn(1, 3, 384, 384)
    with torch.inference_mode():
        expected = phase5(images)
        observed = phase6(images)
    assert torch.equal(observed, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_size", 512),
        ("backbone", "efficientnet_v2_s"),
        ("projection_dim", 256),
        ("bifpn_channels", 256),
        ("bifpn_repeats", 3),
        ("token_grid_size", 24),
        ("cbam_reduction", 8),
        ("cbam_spatial_kernel", 3),
    ],
)
def test_out_of_scope_phase6_model_values_are_rejected(
    field: str,
    value: object,
) -> None:
    config = offline_config(bifpn_enabled=True, bifpn_repeats=1)
    with pytest.raises(ConfigurationError):
        replace(config, **{field: value}).validate()
