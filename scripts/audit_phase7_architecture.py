"""Synthetic shape, parameter, and pretrained-weight audit for phase 7."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from torchvision.models import EfficientNet_V2_S_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.phase7 import load_phase7_experiment
from cbis_ddsm.phase7_model import (
    PHASE7_MAX_PARAMETERS,
    build_phase7_model,
    count_phase7_parameters,
)
from cbis_ddsm.training import sha256_file, write_run_metadata


CONFIG_NAMES = (
    "P7-E01_V2S_seed42.yaml",
    "P7-E02_V2S_BiFPN_R1_C128_seed42.yaml",
    "P7-E03_V2S_CBAM_seed42.yaml",
    "P7-E04_V2S_BiFPN_CBAM_seed42.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit phase 7 shapes and parameter ceilings without data."
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase7"
            / "architecture_audit_v2.json"
        ),
    )
    parser.add_argument("--verify-pretrained", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Audit output already exists: {destination}")

    root = PROJECT_ROOT / "configs" / "phase7"
    experiments = [load_phase7_experiment(root / name) for name in CONFIG_NAMES]
    pretrained = EfficientNet_V2_S_Weights.DEFAULT
    weight_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(pretrained.url).name
    )
    weight_metadata: dict[str, object] | None = None
    if args.verify_pretrained:
        if not weight_path.is_file():
            raise FileNotFoundError(f"Locked V2S weights are missing: {weight_path}")
        verified = build_phase7_model(experiments[0].model)
        del verified
        weight_metadata = {
            "enum": "EfficientNet_V2_S_Weights.DEFAULT",
            "url": pretrained.url,
            "path": str(weight_path),
            "size_bytes": weight_path.stat().st_size,
            "sha256": sha256_file(weight_path),
            "strict_load_pass": True,
        }

    audits = []
    for experiment in experiments:
        offline = replace(experiment.model, pretrained=False)
        model = build_phase7_model(offline).eval()
        total, trainable = count_phase7_parameters(model)
        images = torch.zeros(1, 3, 384, 384)
        with torch.inference_mode():
            feature_map, endpoints = model.forward_feature_map(images)
            _, tokens = model.forward_tokens(images)
            logits = model(images)
        audits.append(
            {
                "candidate_id": experiment.experiment_id,
                "source_config": str(
                    experiment.source_path.relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
                "source_config_sha256": sha256_file(experiment.source_path),
                "model_config": asdict(offline),
                "endpoint_shapes": [list(feature.shape) for feature in endpoints],
                "feature_map_shape": list(feature_map.shape),
                "token_shape": list(tokens.shape),
                "logit_shape": list(logits.shape),
                "total_parameters": total,
                "trainable_parameters": trainable,
                "parameter_limit": PHASE7_MAX_PARAMETERS,
                "parameter_limit_pass": total <= PHASE7_MAX_PARAMETERS,
            }
        )
        del model

    result = {
        "status": (
            "PASS"
            if all(item["parameter_limit_pass"] for item in audits)
            else "FAIL"
        ),
        "created_at": datetime.now().astimezone().isoformat(),
        "input_shape": [1, 3, 384, 384],
        "expected_token_shape": [1, 144, 512],
        "pretrained_weights": weight_metadata,
        "checkpoint_written": False,
        "formal_metrics_claimed": False,
        "data_split_loaded": False,
        "test_split_loaded": False,
        "candidates": audits,
    }
    write_run_metadata(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
