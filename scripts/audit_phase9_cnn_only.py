"""Architecture, provenance, preprocessing equivalence and synthetic CUDA gates."""

import gc
import json
import subprocess
import sys
import types
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbis_ddsm import preprocessing
from cbis_ddsm.phase9 import (SEEDS, SOURCE_ROOT, load_phase9_experiment, runtime_hashes,
                             verify_phase9_manifests, verify_run_predictions, write_exclusive)
from cbis_ddsm.phase9_model import build_phase9_model, pretrained_weight_metadata
from cbis_ddsm.training import set_global_seed, sha256_file


def preprocessing_equivalence(config):
    original = subprocess.check_output(["git", "show", "phase-7-final:src/cbis_ddsm/preprocessing.py"],
                                       cwd=ROOT, text=True)
    module = types.ModuleType("phase7_original_preprocessing")
    sys.modules[module.__name__] = module
    exec(compile(original, "phase7_original_preprocessing.py", "exec"), module.__dict__)
    rng = np.random.default_rng(42)
    cases = [np.zeros((96, 80, 3), dtype=np.uint8), np.full((96, 80, 3), 255, dtype=np.uint8)]
    for index in range(30):
        array = np.zeros((97+index, 81+index, 3), dtype=np.uint8)
        array[10:80, 6:60] = rng.integers(1, 256, (70, 54, 3), dtype=np.uint8)
        if index % 2:
            array[0] = 255
        cases.append(array)
    for array in cases:
        image = Image.fromarray(array)
        a = module.apply_spatial_preprocessing(image, config)
        b = preprocessing.apply_spatial_preprocessing(image, config)
        assert np.array_equal(a.image, b.image)
        for field in ("crop_box", "foreground_fraction", "used_crop", "fallback", "fallback_reason"):
            assert getattr(a, field) == getattr(b, field)
    return {"cases": len(cases), "pixel_and_geometry_equal": True,
            "reference": "phase-7-final:src/cbis_ddsm/preprocessing.py"}


def main():
    destination = ROOT / "artifacts/experiments/phase9/gates_v1.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = {"status": "FAIL", "created_at": datetime.now().astimezone().isoformat(),
              "test_split_loaded": False, "checkpoint_written": False,
              "source_hashes": runtime_hashes()}
    try:
        exp = load_phase9_experiment(ROOT / "configs/phase9/P9-E01_V2S_CNN_only_seed42.yaml")
        train, val, hashes = verify_phase9_manifests()
        result["manifests"] = hashes
        result["train_val_patient_overlap"] = 0
        result["train_image_paths_exist"] = all(Path(p).is_file() for p in train.image_path)
        result["val_image_paths_exist"] = all(Path(p).is_file() for p in val.image_path)
        assert result["train_image_paths_exist"] and result["val_image_paths_exist"]
        readiness = json.loads((ROOT / "artifacts/experiments/cnn_only_supplement/readiness_20261112/existing_result_audit.json").read_text())
        report = (ROOT / "docs/phase7_progress_report.md").read_text(encoding="utf-8").lower()
        reference_training = readiness["training_protocol"]
        controls = []
        for item in readiness["runs"]:
            path = SOURCE_ROOT / "artifacts/experiments/phase7" / item["id"]
            for filename, key in (("best.pt", "checkpoint_sha256"), ("val_predictions_best.csv", "prediction_sha256")):
                assert sha256_file(path / filename) == item[key] and item[key] in report
            config = json.loads((path / "run_config.json").read_text())
            assert config["manifest_hashes"] == hashes
            assert {k:v for k,v in config["training_config"].items() if k not in ("seed", "output_dir")} == reference_training
            controls.append(verify_run_predictions(path, val))
        result["existing_controls"] = controls
        result["preprocessing_equivalence"] = preprocessing_equivalence(exp.preprocessing)
        result["pretrained_weights"] = pretrained_weight_metadata()
        set_global_seed(42, deterministic=True)
        model = build_phase9_model(exp.model).eval()
        weights = torch.load(result["pretrained_weights"]["path"], map_location="cpu", weights_only=True)
        assert all(torch.equal(tensor, weights[f"features.{key}"]) for key,tensor in model.backbone.state_dict().items())
        result["features_strict_pretrained_match"] = True
        del weights
        with torch.inference_mode():
            features = model.forward_feature_map(torch.zeros(1, 3, 384, 384))
            logits = model(torch.zeros(1, 3, 384, 384))
        assert features.shape == (1,1280,12,12) and logits.shape == (1,)
        assert torch.isfinite(features).all() and torch.isfinite(logits).all()
        assert not any(isinstance(m, (nn.TransformerEncoder, nn.LayerNorm, nn.Dropout)) for m in model.modules())
        result["feature_shape"] = list(features.shape)
        result["total_parameters"] = sum(p.numel() for p in model.parameters())
        result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model, features, logits
        gc.collect()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_phase9_model(replace(exp.model, pretrained=False)).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.84], device="cuda"))
        for _ in range(2):
            x = torch.randn(8, 3, 384, 384, device="cuda")
            y = torch.randint(0, 2, (8,), device="cuda").float()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(x), y)
            assert torch.isfinite(loss)
            scaler.scale(loss / 2).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
        assert peak <= 8192
        result["memory_smoke"] = {"status": "PASS", "physical_batch_size": 8,
                                  "gradient_accumulation_steps": 2, "effective_batch_size": 16,
                                  "peak_memory_mib": round(peak, 2), "loss": float(loss.detach()),
                                  "gradients_finite": True, "optimizer_steps": 1,
                                  "data_split_loaded": False, "checkpoint_written": False}
        result["cuda_name"] = torch.cuda.get_device_name()
        result["status"] = "PASS"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_exclusive(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
