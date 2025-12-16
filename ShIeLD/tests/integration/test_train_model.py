import pickle
from pathlib import Path
import pytest


@pytest.mark.integration
def test_train_model_smoke(training_artifacts_dir, artifacts_root):
    """
    Training step smoke test.
    Uses the session pipeline fixture (graph -> train) and asserts outputs exist.
    """
    training_artifacts_dir = Path(training_artifacts_dir)
    artifacts_root = Path(artifacts_root)

    results_csv = training_artifacts_dir / "validation_split_results_training.csv"
    assert results_csv.exists() and results_csv.stat().st_size > 0, (
        f"Missing/empty: {results_csv}"
    )

    req_path = artifacts_root / "requirements_train_smoke.pkl"
    assert req_path.exists() and req_path.stat().st_size > 0, (
        f"Missing/empty: {req_path}"
    )

    # Optional sanity: make sure it's a dict
    req = pickle.load(open(req_path, "rb"))
    assert isinstance(req, dict)
