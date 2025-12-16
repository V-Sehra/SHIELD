from pathlib import Path


def test_graph_artifacts_exist(graph_artifacts_dir: Path):
    assert list(graph_artifacts_dir.rglob("*.pt"))
