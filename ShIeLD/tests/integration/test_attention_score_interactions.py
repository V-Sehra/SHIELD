from pathlib import Path
import pytest


@pytest.mark.integration
def test_attention_score_interactions_smoke(attention_artifacts_dir, artifacts_root):
    """
    Attention/c2c step smoke test.
    Consumes the full session pipeline fixture and asserts outputs exist.
    """
    attention_artifacts_dir = Path(attention_artifacts_dir)
    artifacts_root = Path(artifacts_root)

    # interaction dict is written to model dir (from requirements)
    # your conftest sets req["path_to_model"] to <hypersearch>/model
    # so the dict should live there.
    # easiest: just check that we produced *something* non-empty in attention dir:
    pngs = list(attention_artifacts_dir.rglob("*.png"))
    assert pngs, f"No plots created under: {attention_artifacts_dir}"

    # and also check the dict existence where conftest expects it
    # (you already computed dict_path in conftest; keep that assertion there too)
