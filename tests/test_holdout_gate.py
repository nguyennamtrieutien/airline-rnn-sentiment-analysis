import pytest

from airline_rnn.common import initialize, save_json
from airline_rnn.config import Configuration
from airline_rnn.evaluation import evaluate_final, freeze_selection, verified_runs


def test_cannot_read_test_before_selection(tmp_path):
    with pytest.raises(RuntimeError, match="Test is locked"):
        evaluate_final(tmp_path)
    assert not list((tmp_path / "results/predictions").glob("*test*"))


def test_selection_rejects_incomplete_matrix(tmp_path):
    initialize(tmp_path)
    save_json(tmp_path / "configs/frozen_matrix.json", {
        "tier": "minimum", "matrix": [{"config": Configuration("A1").to_dict(), "seeds": [42]}]
    })
    with pytest.raises(RuntimeError, match="Matrix incomplete"):
        freeze_selection(tmp_path)
    assert not (tmp_path / "results/metrics/final_selection.json").exists()


def test_modified_checkpoint_is_rejected(tmp_path):
    from airline_rnn.common import training_source_digest
    initialize(tmp_path)
    config = Configuration("C0").to_dict()
    save_json(tmp_path / "configs/frozen_matrix.json", {"matrix": [{"config": config, "seeds": [42]}]})
    folder = tmp_path / "models/runs/C0/seed-42"
    folder.mkdir(parents=True)
    (folder / "best.keras").write_text("modified checkpoint")
    save_json(folder / "run.json", {"status": "completed", "smoke": False, "config": config,
              "training_source_sha256": training_source_digest(), "artifact_hashes": {"best.keras": "old-checksum"}})
    with pytest.raises(RuntimeError, match="Modified run artifact"):
        verified_runs(tmp_path)
