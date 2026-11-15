"""Recompute validation-only controls and paired seed differences without inference."""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbis_ddsm.phase9 import (SEEDS, SOURCE_ROOT, load_phase9_experiment, runtime_hashes,
                             verify_phase9_manifests, verify_run_predictions, write_exclusive)


def main():
    destination = ROOT / "artifacts/experiments/phase9/comparison"
    if destination.exists():
        raise FileExistsError(destination)
    _, val, hashes = verify_phase9_manifests()
    gates = json.loads((ROOT / "artifacts/experiments/phase9/gates_v1.json").read_text())
    if gates["status"] != "PASS" or gates["source_hashes"] != runtime_hashes():
        raise ValueError("Gate source hashes changed.")
    rows = []
    for label, prefix, source in (("CNN-only", "P9-E01_V2S_CNN_only", ROOT),
                                  ("CNN+Transformer", "P7-E01_V2S", SOURCE_ROOT),
                                  ("CNN+BiFPN+CBAM+Transformer", "P7-E04_V2S_BiFPN_CBAM", SOURCE_ROOT)):
        for seed in SEEDS:
            directory = source / "artifacts/experiments" / ("phase9" if source == ROOT else "phase7") / f"{prefix}_seed{seed}"
            evidence = verify_run_predictions(directory, val)
            config = json.loads((directory / "run_config.json").read_text())
            assert config["manifest_hashes"] == hashes
            if source == ROOT:
                exp = load_phase9_experiment(ROOT / f"configs/phase9/{directory.name}.yaml")
                assert config["audit_metadata"]["source_hashes"] == runtime_hashes()
                assert config["training_config"]["seed"] == exp.training.seed
            else:
                frozen = next(r for r in gates["existing_controls"] if r["experiment_id"] == directory.name)
                assert frozen["sha256"] == evidence["sha256"]
            rows.append({"model": label, **evidence})
    summary = []
    for label in dict.fromkeys(r["model"] for r in rows):
        group = [r for r in rows if r["model"] == label]
        summary.append({"model": label, "roc_auc_mean": statistics.mean(r["roc_auc"] for r in group),
                        "roc_auc_sample_std": statistics.stdev(r["roc_auc"] for r in group),
                        "pr_auc_mean": statistics.mean(r["pr_auc"] for r in group),
                        "pr_auc_sample_std": statistics.stdev(r["pr_auc"] for r in group)})
    differences = []
    for left, right in (("CNN+Transformer", "CNN-only"),
                        ("CNN+BiFPN+CBAM+Transformer", "CNN+Transformer")):
        values = []
        for seed in SEEDS:
            a = next(r for r in rows if r["model"] == left and r["seed"] == seed)
            b = next(r for r in rows if r["model"] == right and r["seed"] == seed)
            values.append({"seed": seed, "roc_auc_difference": a["roc_auc"] - b["roc_auc"],
                           "pr_auc_difference": a["pr_auc"] - b["pr_auc"]})
        delta = [v["roc_auc_difference"] for v in values]
        differences.append({"comparison": f"{left} minus {right}", "seeds": values,
                            "roc_auc_difference_mean": statistics.mean(delta),
                            "roc_auc_difference_sample_std": statistics.stdev(delta),
                            "positive_seed_count": sum(v > 0 for v in delta),
                            "negative_seed_count": sum(v < 0 for v in delta),
                            "all_seeds_positive": all(v > 0 for v in delta),
                            "statistical_significance_claimed": False})
    result = {"status": "completed", "test_split_loaded": False, "inference_performed": False,
              "new_training_runs": 3, "reused_training_runs": 6, "rows": rows,
              "summary": summary, "paired_differences": differences,
              "limitation": "Post-hoc validation supplement after known historical test results. Configuration comparison, not isolated encoder causality; three seeds do not establish significance."}
    destination.mkdir(parents=True, exist_ok=False)
    write_exclusive(destination / "controls_summary.json", result)
    with (destination / "validation_comparison.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        fields = [k for k in rows[0] if k != "sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    with (destination / "three_seed_summary.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    colors = ("#247BA0", "#E09F3E", "#8B5FBF")
    for index, group in enumerate(summary):
        y = [r["roc_auc"] for r in rows if r["model"] == group["model"]]
        ax.errorbar(index, group["roc_auc_mean"], yerr=group["roc_auc_sample_std"],
                    fmt="s", color=colors[index], capsize=7, markersize=8)
        for offset, value, seed in zip((-.12, 0, .12), y, SEEDS):
            ax.scatter(index+offset, value, color=colors[index], alpha=.75)
            ax.annotate(str(seed), (index+offset, value), xytext=(3,4), textcoords="offset points", fontsize=8)
    ax.set_xticks(range(3), ["V2S CNN-only", "V2S + Transformer", "V2S + BiFPN + CBAM\n+ Transformer"])
    ax.set_ylabel("Validation ROC AUC")
    ax.set_title("Three seeds: mean ± sample standard deviation")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(destination / "validation_auc_comparison.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"summary": summary, "paired_differences": differences}, indent=2))


if __name__ == "__main__":
    main()
