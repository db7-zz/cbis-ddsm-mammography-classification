"""Audit and strictly load the frozen phase 8 source artifacts without test access."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.phase8 import (
    PHASE8_MODEL_IDS,
    audit_phase8_artifacts,
    load_phase8_artifact_lock,
    load_phase8_frozen_model,
)
from cbis_ddsm.training import write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit phase 8 manifests, predictions, and checkpoints without test."
    )
    parser.add_argument(
        "--lock",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "artifact_lock.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase8"
            / "gate0_artifact_audit.json"
        ),
    )
    return parser.parse_args()


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Gate 0 audit output already exists: {output}")
    lock = load_phase8_artifact_lock(args.lock)
    result = audit_phase8_artifacts(PROJECT_ROOT, lock)
    strict_loads = {}
    for model_id in PHASE8_MODEL_IDS:
        model, metadata = load_phase8_frozen_model(PROJECT_ROOT, lock, model_id)
        strict_loads[model_id] = metadata
        del model
        gc.collect()
    result.update(
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "git_status_short": git_value("status", "--short"),
            "strict_model_loads": strict_loads,
            "checkpoint_written": False,
            "formal_metrics_claimed": False,
            "official_test_authorized": False,
            "official_test_loaded": False,
            "test_split_loaded": False,
        }
    )
    write_run_metadata(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
