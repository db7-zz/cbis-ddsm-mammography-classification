from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.phase9_model import Phase9ModelConfig, build_phase9_model


def test_cnn_only_direct_gap_head_shapes_and_backward():
    model = build_phase9_model(Phase9ModelConfig(pretrained=False))
    assert isinstance(model.pool, nn.AdaptiveAvgPool2d)
    assert model.classifier.in_features == 1280 and model.classifier.out_features == 1
    assert not any(isinstance(m, (nn.TransformerEncoder, nn.LayerNorm, nn.Dropout)) for m in model.modules())
    x = torch.randn(1, 3, 384, 384)
    assert model.forward_feature_map(x).shape == (1, 1280, 12, 12)
    logits = model(x)
    assert logits.shape == (1,)
    loss = nn.BCEWithLogitsLoss()(logits, torch.ones(1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 256, 256))


@pytest.mark.parametrize("field,value", [("backbone", "efficientnet_b0"), ("input_size", 256),
                                        ("freeze_backbone", True), ("output_logits", 2),
                                        ("pooling", "max")])
def test_rejects_other_architectures(field, value):
    with pytest.raises(ConfigurationError):
        replace(Phase9ModelConfig(), **{field: value}).validate()


def test_rejects_extra_transformer_config():
    document = asdict(Phase9ModelConfig())
    document["transformer_layers"] = 0
    with pytest.raises(ConfigurationError):
        Phase9ModelConfig.from_mapping(document)
