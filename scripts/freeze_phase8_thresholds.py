"""Freeze P7-E01 and P7-E04 validation thresholds before any official-test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.phase8 import (
    freeze_phase8_thresholds,
    load_phase8_artifact_lock,
    load_phase8_official_test_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze phase 8 validation thresholds without loading official test."
    )
    parser.add_argument(
        "--lock",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "artifact_lock.yaml"),
    )
    parser.add_argument(
        "--policy",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "official_test.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = load_phase8_artifact_lock(args.lock)
    policy = load_phase8_official_test_policy(args.policy)
    results = freeze_phase8_thresholds(PROJECT_ROOT, lock, policy)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
