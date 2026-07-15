"""Run a no-optimizer forward/backward smoke test for the phase 2 model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.model import build_model, count_parameters, load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the phase 2 model.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "model.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use configured ImageNet weights; this can trigger a download.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = select_device(args.device)
    config = load_model_config(args.config)
    config = replace(config, pretrained=bool(args.pretrained))
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(config).to(device).train()
    observed: dict[str, list[int]] = {}
    hooks = [
        model.backbone.register_forward_hook(
            lambda _module, _inputs, output: observed.__setitem__(
                "backbone_feature_map", list(output.shape)
            )
        ),
        model.channel_projection.register_forward_hook(
            lambda _module, _inputs, output: observed.__setitem__(
                "projected_feature_map", list(output.shape)
            )
        ),
        model.transformer.register_forward_hook(
            lambda _module, _inputs, output: observed.__setitem__(
                "encoded_tokens", list(output.shape)
            )
        ),
    ]
    images = torch.randn(
        args.batch_size,
        3,
        config.input_size,
        config.input_size,
        device=device,
    )
    labels = torch.randint(0, 2, (args.batch_size,), device=device).float()
    logits = model(images)
    for hook in hooks:
        hook.remove()
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.84], device=device))(
        logits, labels
    )
    loss.backward()

    total, trainable = count_parameters(model)
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": asdict(config),
        "input_shape": list(images.shape),
        **observed,
        "logit_shape": list(logits.shape),
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "cuda_peak_memory_mib": (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else None
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
