"""CLI entry point for phase 1 manifest preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.config import (
    ConfigurationError,
    apply_path_overrides,
    load_config,
    validate_data_config,
)
from cbis_ddsm.indexing import DataIntegrityError, prepare_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe CBIS-DDSM calcification manifests."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--data-root")
    parser.add_argument("--train-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Record unresolved paths instead of failing. Intended only for metadata diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        config = apply_path_overrides(
            config,
            data_root=args.data_root,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
        )
        if args.allow_missing_images:
            config["data"]["strict_missing_images"] = False
        validate_data_config(config)
        manifest, summary = prepare_manifests(config)
        output_dir = Path(config["data"]["output_dir"]).expanduser().resolve()
        print(f"Wrote {len(manifest)} samples to {output_dir}")
        print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
        return 0
    except (ConfigurationError, DataIntegrityError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
