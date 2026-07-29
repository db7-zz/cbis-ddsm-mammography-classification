"""Run synthetic Grad-CAM++ checks on all three frozen phase 8 models."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.gradcam_pp import (
    generate_gradcam_pp,
    load_gradcam_pp_config,
    select_phase8_target_layer,
)
from cbis_ddsm.phase8 import (
    PHASE8_MODEL_IDS,
    load_phase8_artifact_lock,
    load_phase8_frozen_model,
)
from cbis_ddsm.training import set_global_seed, write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic phase 8 Grad-CAM++ audit without any data split."
    )
    parser.add_argument(
        "--lock",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "artifact_lock.yaml"),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "gradcam_pp.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase8"
            / "gradcam_pp_synthetic_audit.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Grad-CAM++ audit output already exists: {output}")
    lock = load_phase8_artifact_lock(args.lock)
    config = load_gradcam_pp_config(args.config)
    set_global_seed(42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = torch.linspace(
        -1.0,
        1.0,
        steps=3 * config.input_size * config.input_size,
        dtype=torch.float32,
    ).reshape(1, 3, config.input_size, config.input_size)
    audits = {}
    for model_id in PHASE8_MODEL_IDS:
        model, metadata = load_phase8_frozen_model(PROJECT_ROOT, lock, model_id)
        model = model.to(device)
        target_layer = select_phase8_target_layer(model_id, model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = generate_gradcam_pp(model, target_layer, base.to(device))
        expected = (1, *config.expected_shapes[model_id])
        if result.target_shape != expected:
            raise RuntimeError(
                f"Grad-CAM++ target shape mismatch for {model_id}: {result.target_shape}"
            )
        audits[model_id] = {
            **metadata,
            "target_layer": config.target_layers[model_id],
            "target_shape": list(result.target_shape),
            "heatmap_shape": list(result.heatmaps.shape),
            "heatmap_min": float(result.heatmaps.min().cpu()),
            "heatmap_max": float(result.heatmaps.max().cpu()),
            "logit": float(result.logits.item()),
            "finite": bool(
                torch.isfinite(result.heatmaps).all()
                and torch.isfinite(result.logits).all()
            ),
            "degenerate": result.degenerate[0],
            "degenerate_handling": (
                "recorded_no_localization_metrics"
                if result.degenerate[0]
                else "not_applicable"
            ),
            "peak_memory_mib": (
                round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                if device.type == "cuda"
                else None
            ),
        }
        del result, target_layer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    document = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "input_shape": [1, 3, config.input_size, config.input_size],
        "target_logit": "malignant",
        "precision": "float32",
        "data_split_loaded": False,
        "official_test_loaded": False,
        "test_split_loaded": False,
        "formal_metrics_claimed": False,
        "audits": audits,
    }
    write_run_metadata(output, document)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
