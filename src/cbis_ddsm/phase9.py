"""Strict configuration, artifact and train/validation gates for CNN-only controls."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ConfigurationError
from .metrics import sigmoid_numpy
from .phase5 import verify_locked_training_manifests
from .phase9_model import Phase9ModelConfig
from .training import TrainingConfig, sha256_file

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 2026, 3407)
REFERENCE_CONFIG = ROOT / "configs/phase7/P7-E01_V2S_seed42.yaml"
SOURCE_ROOT = Path("C:/Users/db_7/.codex/worktrees/117d/surf")


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class Phase9Experiment:
    experiment_id: str
    source_path: Path
    document: dict
    preprocessing: dict
    model: Phase9ModelConfig
    training: TrainingConfig


def load_phase9_experiment(path: str | Path) -> Phase9Experiment:
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    reference = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != set(reference):
        raise ConfigurationError("Phase 9 rejects additional/missing sections, including test paths.")
    if document["preprocessing"] != reference["preprocessing"]:
        raise ConfigurationError("Phase 9 preprocessing must match P7-E01 exactly.")
    for section in ("loss", "optimizer", "scheduler", "early_stopping", "logging"):
        if document[section] != reference[section]:
            raise ConfigurationError(f"Phase 9 locks {section} to P7-E01.")
    training_values = document["training"]
    if set(training_values) != set(reference["training"]):
        raise ConfigurationError("Unexpected/missing training keys.")
    for key, value in reference["training"].items():
        if key not in ("seed", "output_dir") and training_values[key] != value:
            raise ConfigurationError(f"Phase 9 locks training.{key} to {value}.")
    seed = training_values["seed"]
    if type(seed) is not int or seed not in SEEDS:
        raise ConfigurationError("Phase 9 only permits seeds 42, 2026 and 3407.")
    identifier = f"P9-E01_V2S_CNN_only_seed{seed}"
    if document["experiment"] != {"id": identifier, "stage": 9, "variant": "v2s_cnn_only",
                                  "purpose": "CNN-only control under the frozen phase 7 protocol"}:
        raise ConfigurationError("Unexpected phase 9 experiment identity.")
    if training_values["output_dir"] != f"artifacts/experiments/phase9/{identifier}":
        raise ConfigurationError("Output must be the unique phase 9 seed directory.")
    model = Phase9ModelConfig.from_mapping(document["model"])
    if not model.pretrained:
        raise ConfigurationError("Formal phase 9 runs require official pretrained weights.")
    return Phase9Experiment(identifier, path, document, document["preprocessing"], model,
                            TrainingConfig.from_document(document))


def verify_phase9_manifests():
    lock = ROOT / "configs/phase9/manifest_lock.yaml"
    approved = yaml.safe_load((ROOT / "configs/phase7/manifest_lock.yaml").read_text())
    current = yaml.safe_load(lock.read_text())
    if current != {**approved, "stage": 9}:
        raise ConfigurationError("Phase 9 manifest lock differs from the approved train/val lock.")
    train, val, hashes = verify_locked_training_manifests(ROOT / "artifacts/data", lock)
    if len(train) != 986 or len(val) != 241 or set(train.patient_id) & set(val.patient_id):
        raise ValueError("Unexpected train/validation sample counts or overlapping patients.")
    if not train.sample_id.is_unique or not val.sample_id.is_unique:
        raise ValueError("Duplicate sample IDs.")
    if set(train.label) != {0, 1} or set(val.label) != {0, 1}:
        raise ValueError("Both classes are required.")
    return train, val, hashes


def verify_run_predictions(directory: Path, val: pd.DataFrame) -> dict:
    predictions = pd.read_csv(directory / "val_predictions_best.csv")
    config = json.loads((directory / "run_config.json").read_text())
    outcome = json.loads((directory / "outcome.json").read_text())
    metrics = pd.read_csv(directory / "metrics.csv")
    if len(predictions) != 241 or not predictions.sample_id.is_unique:
        raise ValueError("Validation predictions must cover 241 unique samples.")
    if predictions.sample_id.tolist() != val.sample_id.tolist() or predictions.label.tolist() != val.label.tolist():
        raise ValueError("Validation identity/order/labels differ from the locked manifest.")
    probabilities = sigmoid_numpy(predictions.logit.to_numpy())
    if not np.allclose(probabilities, predictions.probability, atol=1e-7, rtol=1e-6):
        raise ValueError("Stored logit and probability disagree.")
    auc = float(roc_auc_score(predictions.label, predictions.probability))
    ap = float(average_precision_score(predictions.label, predictions.probability))
    best = metrics.loc[metrics.val_roc_auc.idxmax()]
    if outcome["status"] != "completed" or config["test_split_loaded"] is not False or outcome["test_split_loaded"] is not False:
        raise ValueError("Run incomplete or test split was loaded.")
    if (int(best.epoch) != outcome["best_epoch"] or
            not np.allclose([auc, ap, auc], [best.val_roc_auc, best.val_pr_auc,
                                           outcome["best_val_roc_auc"]], atol=1e-12, rtol=0)):
        raise ValueError("Best epoch/prediction metrics disagree.")
    if set(predictions.best_epoch) != {outcome["best_epoch"]}:
        raise ValueError("Predictions not from the selected epoch.")
    return {"experiment_id": directory.name, "seed": config["training_config"]["seed"],
            "roc_auc": auc, "pr_auc": ap, "best_epoch": outcome["best_epoch"],
            "completed_epochs": len(metrics), "training_seconds": float(metrics.epoch_seconds.sum()),
            "peak_memory_mib": outcome.get("peak_memory_mib"),
            "total_parameters": config["audit_metadata"]["total_parameters"],
            "train_loss_at_best": float(best.train_loss), "val_loss_at_best": float(best.val_loss),
            "sha256": {name: sha256_file(directory / name) for name in
                       ("best.pt", "last.pt", "val_predictions_best.csv", "metrics.csv", "run_config.json", "outcome.json")},
            "test_split_loaded": False}


def runtime_hashes() -> dict[str, str]:
    paths = ["src/cbis_ddsm/phase9.py", "src/cbis_ddsm/phase9_model.py",
             "src/cbis_ddsm/training.py", "src/cbis_ddsm/dataset.py", "src/cbis_ddsm/preprocessing.py",
             "scripts/run_phase9_experiment.py", "scripts/audit_phase9_cnn_only.py",
             "configs/phase9/manifest_lock.yaml"]
    paths += [f"configs/phase9/P9-E01_V2S_CNN_only_seed{seed}.yaml" for seed in SEEDS]
    return {name: sha256_file(ROOT / name) for name in paths}
