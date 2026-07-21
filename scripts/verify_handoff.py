"""Read-only integrity checks for the phase 1-4 handoff package.

This script does not train, evaluate, download, edit, or delete anything.  It
compares the machine-readable handoff state with Git and the frozen formal
artifacts so a fresh conversation can detect drift before doing new work.
"""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "docs" / "handoff_state.yaml"


class Verification:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        (self.passed if condition else self.failures).append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def is_read_only(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    readonly_flag = getattr(stat, "FILE_ATTRIBUTE_READONLY", 1)
    return bool(attributes & readonly_flag)


def load_json(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open(encoding="utf-8") as stream:
        return json.load(stream)


def compare_exact(
    verification: Verification,
    actual: Any,
    expected: Any,
    label: str,
) -> None:
    verification.check(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def verify_required_files(verification: Verification) -> None:
    required = [
        "AGENTS.md",
        "README.md",
        "docs/HANDOFF.md",
        "docs/handoff_state.yaml",
        "docs/research_plan_v2_post_baseline.md",
        "docs/phase5_preprocessing_resolution_spec.md",
        "docs/phase4_artifact_freeze_record.md",
        "configs/data.yaml",
        "configs/model.yaml",
        "configs/train.yaml",
        "configs/eval.yaml",
        "artifacts/data/summary.json",
        "artifacts/training/baseline_seed42/outcome.json",
        "artifacts/evaluation/baseline_seed42/metrics.json",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    verification.check(not missing, f"required handoff files present; missing={missing}")


def verify_git(verification: Verification, state: dict[str, Any]) -> None:
    repository = state["repository"]
    tag = repository["tags"]["phase4"]
    tag_result = git("rev-list", "-n", "1", tag)
    verification.check(tag_result.returncode == 0, f"Git tag {tag!r} resolves")
    if tag_result.returncode == 0:
        compare_exact(
            verification,
            tag_result.stdout.strip(),
            repository["main_head"],
            "phase-4 tag commit",
        )

    ancestor = git("merge-base", "--is-ancestor", tag, "HEAD")
    verification.check(ancestor.returncode == 0, "phase-4 tag is an ancestor of HEAD")

    remote = git("remote", "get-url", "origin")
    verification.check(remote.returncode == 0, "origin remote is configured")
    if remote.returncode == 0:
        compare_exact(
            verification,
            remote.stdout.strip().removesuffix(".git"),
            repository["remote"].removesuffix(".git"),
            "origin URL",
        )

    branch = git("branch", "--show-current")
    if branch.returncode == 0 and branch.stdout.strip() != repository["current_branch"]:
        verification.warn(
            "current branch differs from the branch recorded at handoff: "
            f"{branch.stdout.strip()!r} vs {repository['current_branch']!r}"
        )

    status = git("status", "--short")
    if status.returncode == 0 and status.stdout.strip():
        verification.warn("working tree is not clean; inspect `git status -sb` before staging")


def verify_data(verification: Verification, state: dict[str, Any]) -> None:
    summary_path = ROOT / "artifacts" / "data" / "summary.json"
    compare_exact(
        verification,
        sha256(ROOT / "artifacts" / "data" / "manifest.csv"),
        state["data"]["manifest_sha256"],
        "unified manifest SHA-256",
    )
    summary = load_json("artifacts/data/summary.json")
    name_map = {"train": "train", "validation": "val", "test": "test"}
    for state_name, summary_name in name_map.items():
        expected = state["data"]["splits"][state_name]
        actual = summary["splits"][summary_name]
        for key in ("samples", "patients", "negative", "positive"):
            compare_exact(
                verification,
                actual[key],
                expected[key],
                f"{state_name} {key}",
            )
    for pair, expected_count in state["data"]["patient_overlap"].items():
        compare_exact(
            verification,
            len(summary["patient_overlap"][pair]),
            expected_count,
            f"patient overlap {pair}",
        )
    verification.check(summary_path.is_file(), "data summary is readable")


def verify_training_and_evaluation(
    verification: Verification, state: dict[str, Any]
) -> None:
    training_outcome = load_json("artifacts/training/baseline_seed42/outcome.json")
    for key in (
        "best_epoch",
        "best_val_roc_auc",
        "completed_epochs",
        "stopped_early",
        "test_split_loaded",
    ):
        compare_exact(
            verification,
            training_outcome[key],
            state["training"][key],
            f"training {key}",
        )
    compare_exact(
        verification,
        sha256(ROOT / "artifacts" / "training" / "baseline_seed42" / "best.pt"),
        state["training"]["checkpoint_sha256"],
        "best checkpoint SHA-256",
    )
    compare_exact(
        verification,
        sha256(
            ROOT
            / "artifacts"
            / "training"
            / "baseline_seed42"
            / "val_predictions_best.csv"
        ),
        state["training"]["validation_predictions_sha256"],
        "validation predictions SHA-256",
    )

    metrics_document = load_json("artifacts/evaluation/baseline_seed42/metrics.json")
    metrics = metrics_document["metrics"]
    evaluation = state["evaluation"]
    mapping = {
        "samples": "official_test_samples",
        "threshold": "threshold",
        "roc_auc": "roc_auc",
        "pr_auc": "pr_auc",
        "sensitivity": "sensitivity",
        "specificity": "specificity",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "f1": "f1",
    }
    for metric_key, state_key in mapping.items():
        compare_exact(
            verification,
            metrics[metric_key],
            evaluation[state_key],
            f"evaluation {metric_key}",
        )
    for key in ("tn", "fp", "fn", "tp"):
        compare_exact(
            verification,
            metrics[key],
            evaluation["confusion_matrix"][key],
            f"confusion matrix {key}",
        )
    compare_exact(
        verification,
        sha256(
            ROOT / "artifacts" / "evaluation" / "baseline_seed42" / "test_predictions.csv"
        ),
        evaluation["test_predictions_sha256"],
        "test predictions SHA-256",
    )


def verify_freeze(verification: Verification, state: dict[str, Any]) -> None:
    freeze = state["freeze"]
    backup = ROOT / freeze["backup_dir"]
    snapshot = backup / "snapshot"
    manifest_path = backup / "freeze_manifest.json"
    checksum_path = backup / "freeze_manifest.sha256"
    verification.check(manifest_path.is_file(), "freeze manifest exists")
    verification.check(checksum_path.is_file(), "freeze manifest checksum exists")
    if not manifest_path.is_file() or not checksum_path.is_file():
        return

    expected_checksum = checksum_path.read_text(encoding="ascii").split()[0]
    actual_checksum = sha256(manifest_path)
    compare_exact(verification, actual_checksum, expected_checksum, "freeze manifest sidecar SHA-256")
    compare_exact(
        verification,
        actual_checksum,
        freeze["freeze_manifest_sha256"],
        "freeze manifest handoff-state SHA-256",
    )

    manifest = load_json(str(manifest_path.relative_to(ROOT)).replace("\\", "/"))
    compare_exact(verification, manifest["file_count"], freeze["file_count"], "freeze file count")
    compare_exact(verification, manifest["total_bytes"], freeze["total_bytes"], "freeze total bytes")

    hash_mismatches: list[str] = []
    backup_not_read_only: list[str] = []
    source_not_read_only: list[str] = []
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        backup_file = snapshot / relative
        if (
            not backup_file.is_file()
            or backup_file.stat().st_size != entry["bytes"]
            or sha256(backup_file) != entry["sha256"]
        ):
            hash_mismatches.append(entry["relative_path"])
        elif not is_read_only(backup_file):
            backup_not_read_only.append(entry["relative_path"])

        if entry.get("source_frozen_read_only"):
            source = ROOT / relative
            if not source.is_file() or not is_read_only(source):
                source_not_read_only.append(entry["relative_path"])

    verification.check(not hash_mismatches, f"frozen backup hashes match; mismatches={hash_mismatches}")
    verification.check(
        not backup_not_read_only,
        f"all frozen backup files are read-only; exceptions={backup_not_read_only}",
    )
    verification.check(
        not source_not_read_only,
        f"formal runtime source files remain read-only; exceptions={source_not_read_only}",
    )


def verify_external_reference(verification: Verification, state: dict[str, Any]) -> None:
    reference = state.get("external_reference", {})
    path_text = reference.get("original_research_plan")
    expected = reference.get("original_research_plan_sha256")
    if not path_text or not expected:
        return
    path = Path(path_text)
    if not path.is_file():
        verification.warn(f"external historical research plan is unavailable: {path}")
        return
    compare_exact(verification, sha256(path), expected, "external historical research plan SHA-256")


def main() -> int:
    verification = Verification()
    if not STATE_PATH.is_file():
        print(json.dumps({"status": "FAIL", "failures": [f"missing {STATE_PATH}"]}, indent=2))
        return 1

    with STATE_PATH.open(encoding="utf-8") as stream:
        state = yaml.safe_load(stream)

    verify_required_files(verification)
    verify_git(verification, state)
    verify_data(verification, state)
    verify_training_and_evaluation(verification, state)
    verify_freeze(verification, state)
    verify_external_reference(verification, state)

    result = {
        "status": "PASS" if not verification.failures else "FAIL",
        "passed_checks": len(verification.passed),
        "warnings": verification.warnings,
        "failures": verification.failures,
        "safety": "read-only verification; no training or test inference performed",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not verification.failures else 1


if __name__ == "__main__":
    sys.exit(main())
