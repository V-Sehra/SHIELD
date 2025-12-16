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
    pngs = list(attention_artifacts_dir.rglob("*.png"))
    assert pngs, f"No plots created under: {attention_artifacts_dir}"
