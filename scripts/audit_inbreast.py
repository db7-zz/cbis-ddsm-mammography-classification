"""Run the phase 8 Gate D INbreast audit without model inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.evaluation import write_json
from cbis_ddsm.phase8_external import (
    build_inbreast_audit,
    load_inbreast_external_policy,
    validate_inbreast_authorization,
)
from cbis_ddsm.training import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit INbreast Release 1.0 before Gate D external inference."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--policy",
        default=str(
            PROJECT_ROOT / "configs" / "phase8" / "inbreast_external.yaml"
        ),
    )
    parser.add_argument(
        "--authorization",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase8"
            / "external"
            / "inbreast_authorization.json"
        ),
    )
    parser.add_argument(
        "--decode-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    return result.stdout.strip() or None


def inventory_files(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(item for item in dataset_root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(dataset_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    policy = load_inbreast_external_policy(args.policy)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    authorization = validate_inbreast_authorization(
        args.authorization,
        dataset_root=dataset_root,
        policy=policy,
    )
    output_dir = (PROJECT_ROOT / policy.audit_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Gate D audit output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(
        output_dir / "audit_started.json",
        {
            "format_version": 1,
            "stage": 8,
            "gate": "D",
            "status": "started",
            "started_at_utc": started_at,
            "dataset_root": str(dataset_root),
            "approved_by_user": authorization["approved_by_user"],
            "raw_data_mutated": False,
            "external_inference_performed": False,
        },
    )
    try:
        inventory = inventory_files(dataset_root)
        inventory_path = output_dir / "source_file_sha256.csv"
        inventory.to_csv(inventory_path, index=False, encoding="utf-8")
        manifest, dicom_audit, outcome = build_inbreast_audit(
            dataset_root,
            policy,
            decode_pixels=args.decode_pixels,
        )
        manifest_path = output_dir / "candidate_manifest.csv"
        dicom_path = output_dir / "dicom_audit.csv"
        manifest.to_csv(manifest_path, index=False, encoding="utf-8")
        dicom_audit.to_csv(dicom_path, index=False, encoding="utf-8")
        outcome.update(
            {
                "format_version": 1,
                "stage": 8,
                "gate": "D",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "authorization_sha256": sha256_file(Path(args.authorization)),
                "policy_sha256": sha256_file(policy.source_path),
                "git_commit": git_value("rev-parse", "HEAD"),
                "git_branch": git_value("branch", "--show-current"),
                "git_status_short": git_value("status", "--short"),
                "source_inventory": {
                    "path": str(inventory_path),
                    "sha256": sha256_file(inventory_path),
                    "files": len(inventory),
                    "total_bytes": int(inventory["size_bytes"].sum()),
                },
                "candidate_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "dicom_audit": {
                    **outcome["dicom"],
                    "path": str(dicom_path),
                    "sha256": sha256_file(dicom_path),
                },
            }
        )
        write_json(output_dir / "outcome.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        write_json(
            output_dir / "failure.json",
            {
                "format_version": 1,
                "stage": 8,
                "gate": "D",
                "status": "failed_no_automatic_rerun",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "reason": str(error),
                "raw_data_mutated": False,
                "external_inference_performed": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
