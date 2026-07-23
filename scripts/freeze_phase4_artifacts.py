"""Create and verify a read-only snapshot of the formal phase 4 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP = PROJECT_ROOT / "backups" / "phase4_final_frozen_20260721"

LOCKED_FILES: tuple[tuple[str, str, bool], ...] = (
    ("artifacts/data/dicom_validation.json", "data_integrity", True),
    ("artifacts/data/manifest.csv", "unified_manifest", True),
    ("artifacts/data/summary.json", "split_summary", True),
    ("artifacts/data/train.csv", "train_manifest", True),
    ("artifacts/data/val.csv", "validation_manifest", True),
    ("artifacts/data/test.csv", "official_test_manifest", True),
    ("artifacts/training/baseline_seed42/best.pt", "numerical_best_checkpoint", True),
    ("artifacts/training/baseline_seed42/last.pt", "final_checkpoint", True),
    ("artifacts/training/baseline_seed42/metrics.csv", "epoch_metrics", True),
    ("artifacts/training/baseline_seed42/outcome.json", "training_outcome", True),
    ("artifacts/training/baseline_seed42/run_config.json", "training_provenance", True),
    (
        "artifacts/training/baseline_seed42/val_predictions_best.csv",
        "validation_predictions",
        True,
    ),
    (
        "artifacts/training/baseline_seed42/tensorboard/"
        "events.out.tfevents.1784136514.DESKTOP-C532KB2.14304.0",
        "tensorboard_event",
        True,
    ),
    ("artifacts/evaluation/baseline_seed42/confusion_matrix.png", "test_figure", True),
    ("artifacts/evaluation/baseline_seed42/metrics.json", "test_metrics", True),
    ("artifacts/evaluation/baseline_seed42/outcome.json", "evaluation_outcome", True),
    ("artifacts/evaluation/baseline_seed42/pr_curve.png", "test_figure", True),
    ("artifacts/evaluation/baseline_seed42/roc_curve.png", "test_figure", True),
    (
        "artifacts/evaluation/baseline_seed42/test_evaluation_started.json",
        "test_access_audit",
        True,
    ),
    (
        "artifacts/evaluation/baseline_seed42/test_predictions.csv",
        "official_test_predictions",
        True,
    ),
    ("artifacts/evaluation/baseline_seed42/threshold.json", "frozen_threshold", True),
    (
        "artifacts/evaluation/baseline_seed42/validation_selection.json",
        "validation_threshold_audit",
        True,
    ),
    ("configs/data.yaml", "tracked_configuration", False),
    ("configs/data.local.yaml", "formal_local_data_configuration", False),
    ("configs/model.yaml", "tracked_configuration", False),
    ("configs/train.yaml", "tracked_configuration", False),
    ("configs/eval.yaml", "tracked_configuration", False),
    ("docs/phase1_data_audit.md", "tracked_report", False),
    ("docs/phase2_model_spec.md", "tracked_report", False),
    ("docs/phase3_training_spec.md", "tracked_report", False),
    ("docs/phase3_training_report.md", "tracked_report", False),
    ("docs/phase4_evaluation_spec.md", "tracked_report", False),
    ("docs/phase4_evaluation_report.md", "tracked_report", False),
    ("README.md", "tracked_project_documentation", False),
    ("pyproject.toml", "dependency_specification", False),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def set_read_only(path: Path) -> None:
    path.chmod(stat.S_IREAD)
    if os.name == "nt":
        current = path.stat().st_mode
        if current & stat.S_IWRITE:
            raise PermissionError(f"Failed to remove write permission: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP))
    args = parser.parse_args()
    backup_root = Path(args.backup_dir).expanduser().resolve()
    if backup_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing freeze: {backup_root}")

    missing = [relative for relative, _, _ in LOCKED_FILES if not (PROJECT_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Formal artifact inventory is incomplete: " + ", ".join(missing))

    snapshot_root = backup_root / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for relative, role, freeze_source in LOCKED_FILES:
        source = (PROJECT_ROOT / relative).resolve()
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256_file(source)
        shutil.copy2(source, destination)
        backup_hash = sha256_file(destination)
        if source_hash != backup_hash or source.stat().st_size != destination.stat().st_size:
            raise IOError(f"Backup verification failed: {relative}")
        total_bytes += source.stat().st_size
        records.append(
            {
                "relative_path": relative.replace("\\", "/"),
                "role": role,
                "bytes": source.stat().st_size,
                "sha256": source_hash,
                "source_frozen_read_only": freeze_source,
                "backup_verified": True,
            }
        )

    commit = git_output("rev-parse", "HEAD")
    tags = git_output("tag", "--points-at", "HEAD").splitlines()
    manifest: dict[str, Any] = {
        "format_version": 1,
        "freeze_status": "verified_read_only_snapshot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_at_freeze": str(PROJECT_ROOT),
        "backup_root": str(backup_root),
        "git_commit": commit,
        "git_tags_at_commit": tags,
        "formal_evaluation_date": "2026-07-17",
        "formal_test_inference_passes": 1,
        "test_selection_policy": (
            "Validation selected checkpoint and Youden threshold; official test was used only "
            "for the completed phase 4 evaluation."
        ),
        "scope_note": (
            "This snapshot contains formal manifests, configs, reports, training/evaluation "
            "records and checkpoints. It intentionally excludes the approximately 160 GB raw "
            "DICOM collection."
        ),
        "disaster_recovery_note": (
            "This snapshot is on the same D: volume as the project. It protects against routine "
            "overwrites but must be copied to another physical disk or managed object storage "
            "for true disaster recovery."
        ),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "files": records,
    }
    manifest_path = backup_root / "freeze_manifest.json"
    write_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    checksum_path = backup_root / "freeze_manifest.sha256"
    checksum_path.write_text(
        f"{manifest_hash}  freeze_manifest.json\n", encoding="ascii", newline="\n"
    )
    readme_path = backup_root / "READ_ONLY_FREEZE.txt"
    readme_path.write_text(
        "PHASE 4 FORMAL BASELINE - READ ONLY\n"
        "Do not overwrite, resume training into, retune from test results, or reuse output paths.\n"
        "Verify freeze_manifest.sha256 and every file hash before restoration.\n",
        encoding="utf-8",
        newline="\n",
    )

    # Freeze copied files and formal runtime sources. Tracked configs/docs remain writable;
    # their immutable version is the Git tag phase-4-final-evaluation.
    for relative, _, freeze_source in LOCKED_FILES:
        set_read_only(snapshot_root / relative)
        if freeze_source:
            set_read_only(PROJECT_ROOT / relative)
    for control_file in (manifest_path, checksum_path, readme_path):
        set_read_only(control_file)

    # Post-freeze verification reads every file again and confirms read-only state.
    for record in records:
        relative = record["relative_path"]
        destination = snapshot_root / relative
        if sha256_file(destination) != record["sha256"]:
            raise IOError(f"Post-freeze hash mismatch: {relative}")
        if destination.stat().st_mode & stat.S_IWRITE:
            raise PermissionError(f"Backup is not read-only: {relative}")
    if sha256_file(manifest_path) != manifest_hash:
        raise IOError("Freeze manifest changed during final verification.")

    outcome = {
        "backup_root": str(backup_root),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "manifest_sha256": manifest_hash,
        "git_commit": commit,
        "git_tags": tags,
        "status": "PASS",
    }
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
