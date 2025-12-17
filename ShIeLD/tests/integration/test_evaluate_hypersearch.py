from pathlib import Path
import pytest


@pytest.mark.integration
def test_evaluate_hypersearch_smoke(hypersearch_artifacts_dir):
    """
    Hypersearch evaluation step smoke test.
    Consumes the session pipeline fixture (train -> eval) and asserts outputs exist.
    """
    hypersearch_artifacts_dir = Path(hypersearch_artifacts_dir)

    out_csv = hypersearch_artifacts_dir / "hyper_search_results.csv"
    assert out_csv.exists() and out_csv.stat().st_size > 0, f"Missing/empty: {out_csv}"

    model_dir = hypersearch_artifacts_dir / "model"
    assert model_dir.exists(), f"Missing model dir: {model_dir}"

    best_cfg = model_dir / "best_config.pt"
    assert best_cfg.exists() and best_cfg.stat().st_size > 0, (
        f"Missing/empty: {best_cfg}"
    )
