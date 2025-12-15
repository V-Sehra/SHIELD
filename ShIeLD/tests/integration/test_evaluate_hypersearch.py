import pytest
from pathlib import Path
import pickle
import sys
import os


@pytest.mark.integration
def test_evaluate_hypersearch_smoke(training_artifacts_dir, monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    cache_dir = repo_root / ".pytest_cache" / "shield_artifacts"
    req_path = cache_dir / "requirements_train_smoke.pkl"
    assert req_path.exists(), f"Missing patched req: {req_path}"

    # Make a copy so this test writes into tmp_path (clean per run)
    req = pickle.load(open(req_path, "rb"))
    req["path_training_results"] = tmp_path / "training_results"
    req["path_training_results"].mkdir(parents=True, exist_ok=True)
    req["path_to_model"] = tmp_path / "models"
    req["path_to_model"].mkdir(parents=True, exist_ok=True)

    # Copy the produced training CSV into this test's output folder
    src_csv = Path(training_artifacts_dir) / "validation_split_results_training.csv"
    dst_csv = (
        Path(req["path_training_results"]) / "validation_split_results_training.csv"
    )
    dst_csv.write_bytes(src_csv.read_bytes())

    patched = tmp_path / "requirements_eval_hs.pkl"
    with open(patched, "wb") as f:
        pickle.dump(req, f)

    from ShIeLD import evaluate_hypersearch

    # kill tqdm + skip heavy plotting
    monkeypatch.setattr(evaluate_hypersearch, "tqdm", lambda x, **k: x)
    monkeypatch.setattr(
        evaluate_hypersearch.evaluation_utils,
        "create_parameter_influence_plots",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_hypersearch.py",
            "--requirements_file_path",
            str(patched),
            "--retain_best_model_config_bool",
            "False",
        ],
    )
    evaluate_hypersearch.main()

    out_csv = Path(req["path_training_results"]) / "hyper_search_results.csv"
    assert out_csv.exists() and out_csv.stat().st_size > 0
