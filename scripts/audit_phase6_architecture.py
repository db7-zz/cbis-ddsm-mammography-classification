"""Synthetic shape and parameter audit for all approved phase 6 candidates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.phase6 import load_phase6_experiment
from cbis_ddsm.phase6_model import (
    PHASE6_MAX_PARAMETERS,
    build_phase6_model,
    count_phase6_parameters,
)
from cbis_ddsm.training import sha256_file, write_run_metadata


CONFIG_NAMES = (
    "P6-E01A_BiFPN_R1_C128_seed42.yaml",
    "P6-E01B_BiFPN_R2_C128_seed42.yaml",
    "P6-E02_CBAM_seed42.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit phase 6 shapes and parameter ceilings without data."
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase6"
            / "architecture_audit_v2.json"
        ),
    )
    parser.add_argument("--verify-pretrained", action="store_true")
    return parser.parse_args()


def candidate_configs() -> list[tuple[str, object, Path]]:
    root = PROJECT_ROOT / "configs" / "phase6"
    candidates: list[tuple[str, object, Path]] = []
    bifpn_models = []
    for name in CONFIG_NAMES:
        path = root / name
        experiment = load_phase6_experiment(path)
        candidates.append((experiment.experiment_id, experiment.model, path))
        if experiment.model.bifpn_enabled:
            bifpn_models.append((experiment.experiment_id, experiment.model, path))
    for experiment_id, model_config, path in bifpn_models:
        candidates.append(
            (
                experiment_id.replace("P6-E01", "P6-E03-CANDIDATE-") + "_CBAM",
                replace(model_config, cbam_enabled=True),
                path,
            )
        )
    return candidates


def main() -> int:
    args = parse_args()
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Audit output already exists: {destination}")

    audits = []
    candidates = candidate_configs()
    pretrained_verified = False
    if args.verify_pretrained:
        pretrained_model = build_phase6_model(candidates[0][1])
        pretrained_verified = True
        del pretrained_model

    for candidate_id, config, source_path in candidates:
        offline = replace(config, pretrained=False)
        model = build_phase6_model(offline).eval()
        total, trainable = count_phase6_parameters(model)
        images = torch.zeros(1, 3, 384, 384)
        with torch.inference_mode():
            feature_map, endpoints = model.forward_feature_map(images)
            _, tokens = model.forward_tokens(images)
            logits = model(images)
        audits.append(
            {
                "candidate_id": candidate_id,
                "source_config": str(source_path.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
                "source_config_sha256": sha256_file(source_path),
                "model_config": asdict(offline),
                "endpoint_shapes": [list(feature.shape) for feature in endpoints],
                "feature_map_shape": list(feature_map.shape),
                "token_shape": list(tokens.shape),
                "logit_shape": list(logits.shape),
                "total_parameters": total,
                "trainable_parameters": trainable,
                "parameter_limit": PHASE6_MAX_PARAMETERS,
                "parameter_limit_pass": total <= PHASE6_MAX_PARAMETERS,
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
        "pretrained_weights_loaded": pretrained_verified,
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
