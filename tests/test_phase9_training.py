from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.phase9 import ROOT, SEEDS, load_phase9_experiment, verify_phase9_manifests


def test_three_seeds_share_the_frozen_protocol():
    configs = [load_phase9_experiment(ROOT / f"configs/phase9/P9-E01_V2S_CNN_only_seed{s}.yaml") for s in SEEDS]
    assert [c.training.seed for c in configs] == list(SEEDS)
    assert len({c.training.output_dir for c in configs}) == 3
    reference = deepcopy(configs[0].document)
    for c in configs:
        candidate = deepcopy(c.document)
        for d in (reference, candidate):
            d["experiment"].pop("id", None)
            d["training"].pop("seed", None)
            d["training"].pop("output_dir", None)
        assert candidate == reference


@pytest.mark.parametrize("section,key,value", [
    ("training", "batch_size", 16), ("training", "gradient_accumulation_steps", 1),
    ("training", "max_epochs", 25), ("training", "seed", 100),
    ("optimizer", "learning_rate", 0.01), ("loss", "pos_weight", 1.0),
    ("early_stopping", "patience", 5), ("model", "pretrained", False),
    ("preprocessing", "image_size", 512), ("training", "test_path", "forbidden.csv")])
def test_rejects_budget_changes_and_test_paths(tmp_path, section, key, value):
    source = ROOT / "configs/phase9/P9-E01_V2S_CNN_only_seed42.yaml"
    document = yaml.safe_load(source.read_text())
    document[section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError):
        load_phase9_experiment(path)


def test_manifest_gate_exposes_only_locked_train_validation():
    train, val, hashes = verify_phase9_manifests()
    assert (len(train), len(val)) == (986, 241)
    assert set(hashes) == {"train.csv", "val.csv"}
    assert not set(train.patient_id) & set(val.patient_id)
