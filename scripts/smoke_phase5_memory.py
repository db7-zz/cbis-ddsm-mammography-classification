"""Synthetic forward/backward memory gate for an approved phase 5 configuration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.model import build_model
from cbis_ddsm.phase5 import load_phase5_experiment
from cbis_ddsm.training import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one synthetic phase 5 optimizer step without a checkpoint."
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = load_phase5_experiment(args.experiment_config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    set_global_seed(experiment.training.seed, deterministic=True)
    model_config = replace(experiment.model, pretrained=False)
    model = build_model(model_config).to(device).train()
    optimizer = AdamW(
        model.parameters(),
        lr=experiment.training.learning_rate,
        weight_decay=experiment.training.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([experiment.training.pos_weight], device=device)
    )
    amp_enabled = experiment.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    loss_value = float("nan")
    try:
        for _ in range(experiment.training.gradient_accumulation_steps):
            images = torch.randn(
                experiment.training.batch_size,
                3,
                experiment.model.input_size,
                experiment.model.input_size,
                device=device,
            )
            labels = torch.randint(
                0,
                2,
                (experiment.training.batch_size,),
                device=device,
            ).float()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
                scaled_loss = loss / experiment.training.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError("Synthetic smoke loss is not finite.")
            scaler.scale(scaled_loss).backward()
            loss_value = float(loss.detach().item())
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), experiment.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {
            "status": "PASS",
            "experiment_id": experiment.experiment_id,
            "device": str(device),
            "cuda_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "input_size": experiment.model.input_size,
            "physical_batch_size": experiment.training.batch_size,
            "gradient_accumulation_steps": (
                experiment.training.gradient_accumulation_steps
            ),
            "effective_batch_size": (
                experiment.training.batch_size
                * experiment.training.gradient_accumulation_steps
            ),
            "loss": loss_value,
            "loss_is_finite": True,
            "cuda_peak_memory_mib": (
                round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                if device.type == "cuda"
                else None
            ),
            "pretrained_weights_loaded": False,
            "checkpoint_written": False,
            "formal_metrics_claimed": False,
            "test_split_loaded": False,
            "model_config": asdict(model_config),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except torch.OutOfMemoryError as error:
        result = {
            "status": "OOM",
            "experiment_id": experiment.experiment_id,
            "error": str(error),
            "checkpoint_written": False,
            "formal_metrics_claimed": False,
            "test_split_loaded": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
