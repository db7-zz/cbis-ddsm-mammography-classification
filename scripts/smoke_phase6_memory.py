"""CUDA memory gate for approved phase 6 candidates using synthetic tensors only."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.phase6 import load_phase6_experiment
from cbis_ddsm.phase6_model import build_phase6_model, count_phase6_parameters
from cbis_ddsm.training import set_global_seed, write_run_metadata


CONFIG_NAMES = (
    "P6-E01A_BiFPN_R1_C128_seed42.yaml",
    "P6-E01B_BiFPN_R2_C128_seed42.yaml",
    "P6-E02_CBAM_seed42.yaml",
)
MEMORY_LIMIT_MIB = 8 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phase 6 synthetic memory gates without loading data."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase6"
            / "memory_smoke_v1.json"
        ),
    )
    return parser.parse_args()


def candidates() -> list[tuple[str, object, object]]:
    root = PROJECT_ROOT / "configs" / "phase6"
    result = []
    bifpn = []
    for name in CONFIG_NAMES:
        experiment = load_phase6_experiment(root / name)
        result.append((experiment.experiment_id, experiment.model, experiment.training))
        if experiment.model.bifpn_enabled:
            bifpn.append((experiment.experiment_id, experiment.model, experiment.training))
    for experiment_id, model, training in bifpn:
        result.append(
            (
                experiment_id.replace("P6-E01", "P6-E03-CANDIDATE-") + "_CBAM",
                replace(model, cbam_enabled=True),
                training,
            )
        )
    return result


def run_candidate(
    candidate_id: str,
    model_config: object,
    training: object,
    device: torch.device,
) -> dict[str, object]:
    set_global_seed(training.seed, deterministic=True)
    offline = replace(model_config, pretrained=False)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = build_phase6_model(offline).to(device).train()
    total, trainable = count_phase6_parameters(model)
    optimizer = AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([training.pos_weight], device=device)
    )
    amp_enabled = training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer.zero_grad(set_to_none=True)
    loss_value = float("nan")
    try:
        for _ in range(training.gradient_accumulation_steps):
            images = torch.randn(
                training.batch_size,
                3,
                model_config.input_size,
                model_config.input_size,
                device=device,
            )
            labels = torch.randint(
                0,
                2,
                (training.batch_size,),
                device=device,
            ).float()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
                scaled_loss = loss / training.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError("Synthetic phase 6 loss is not finite.")
            scaler.scale(scaled_loss).backward()
            loss_value = float(loss.detach().item())
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak = (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else None
        )
        status = (
            "PASS"
            if peak is None or peak <= MEMORY_LIMIT_MIB
            else "LIMIT_EXCEEDED"
        )
        return {
            "status": status,
            "candidate_id": candidate_id,
            "device": str(device),
            "input_size": model_config.input_size,
            "physical_batch_size": training.batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "effective_batch_size": (
                training.batch_size * training.gradient_accumulation_steps
            ),
            "loss": loss_value,
            "loss_is_finite": True,
            "cuda_peak_memory_mib": peak,
            "memory_limit_mib": MEMORY_LIMIT_MIB,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "pretrained_weights_loaded": False,
            "checkpoint_written": False,
            "formal_metrics_claimed": False,
            "data_split_loaded": False,
            "test_split_loaded": False,
            "model_config": asdict(offline),
        }
    except torch.OutOfMemoryError as error:
        return {
            "status": "OOM",
            "candidate_id": candidate_id,
            "error": str(error),
            "physical_batch_size": training.batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "checkpoint_written": False,
            "formal_metrics_claimed": False,
            "data_split_loaded": False,
            "test_split_loaded": False,
        }
    finally:
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Memory smoke output already exists: {destination}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    results = [
        run_candidate(candidate_id, model_config, training, device)
        for candidate_id, model_config, training in candidates()
    ]
    status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
    document = {
        "status": status,
        "created_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "checkpoint_written": False,
        "formal_metrics_claimed": False,
        "data_split_loaded": False,
        "test_split_loaded": False,
        "candidates": results,
    }
    write_run_metadata(destination, document)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
