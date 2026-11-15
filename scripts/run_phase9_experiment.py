"""Run one approved CNN-only seed; no test arguments, downloads or resume."""

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbis_ddsm.phase9 import (load_phase9_experiment, runtime_hashes, verify_phase9_manifests,
                             verify_run_predictions, write_exclusive)
from cbis_ddsm.phase9_model import build_phase9_model, pretrained_weight_metadata
from cbis_ddsm.training import build_dataloaders, set_global_seed, train_model


def git_value(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    args = parser.parse_args()
    exp = load_phase9_experiment(args.experiment_config)
    output = ROOT / exp.training.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing an existing output directory: {output}")
    started = datetime.now().astimezone().isoformat()
    audit = {"experiment_id": exp.experiment_id, "source_hashes": runtime_hashes(),
             "git_commit": git_value("rev-parse", "HEAD"),
             "git_branch": git_value("branch", "--show-current"),
             "git_status_short": git_value("status", "--short"),
             "physical_batch_size": 8, "gradient_accumulation_steps": 2,
             "effective_batch_size": 16, "test_split_loaded": False}
    result = {"experiment_id": exp.experiment_id, "test_split_loaded": False}
    try:
        gate = json.loads((ROOT / "artifacts/experiments/phase9/gates_v1.json").read_text())
        if gate["status"] != "PASS" or gate["source_hashes"] != audit["source_hashes"]:
            raise ValueError("Architecture/memory gates missing, failed or source hashes changed.")
        if not torch.cuda.is_available():
            raise RuntimeError("Approved CUDA device unavailable.")
        train, val, hashes = verify_phase9_manifests()
        audit["pretrained_weights"] = pretrained_weight_metadata()
        set_global_seed(exp.training.seed, deterministic=True)
        train_loader, val_loader = build_dataloaders(train, val, preprocessing=exp.preprocessing,
                                                   config=exp.training)
        model = build_phase9_model(exp.model)
        audit["total_parameters"] = sum(p.numel() for p in model.parameters())
        audit["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(json.dumps({**audit, "started_at": started, "train_samples": len(train),
                          "val_samples": len(val), "manifest_hashes": hashes}), flush=True)
        outcome = train_model(model, train_loader, val_loader, config=exp.training,
                              device=torch.device("cuda"), output_dir=output,
                              model_config=asdict(exp.model), manifest_hashes=hashes,
                              audit_metadata=audit)
        result.update({"status": "completed", "best_epoch": outcome.best_epoch,
                       "best_val_roc_auc": outcome.best_val_roc_auc,
                       "completed_epochs": outcome.completed_epochs, "stopped_early": outcome.stopped_early,
                       "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                       "total_parameters": audit["total_parameters"],
                       "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
                       "started_at": started, "finished_at": datetime.now().astimezone().isoformat()})
        write_exclusive(output / "outcome.json", result)
        evidence = verify_run_predictions(output, val)
        write_exclusive(output / "verification.json", evidence)
    except BaseException as error:
        result.update({"status": "failed", "failure_type": type(error).__name__, "reason": str(error),
                       "started_at": started, "finished_at": datetime.now().astimezone().isoformat(),
                       "audit_metadata": audit, "resolved_config": exp.document})
        write_exclusive(output / "failure.json", result)
        raise
    finally:
        registry = ROOT / "artifacts/experiments/phase9/experiment_registry.csv"
        registry.parent.mkdir(parents=True, exist_ok=True)
        exists = registry.exists()
        fields = ("experiment_id", "status", "started_at", "finished_at", "best_epoch",
                  "best_val_roc_auc", "failure_type", "reason", "test_split_loaded")
        with registry.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: result.get(key, "") for key in fields})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
