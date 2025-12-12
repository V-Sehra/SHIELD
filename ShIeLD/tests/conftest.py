# ShIeLD/tests/conftest.py
import pickle
import shutil
import sys
from pathlib import Path

import pytest
from ShIeLD.utils.testing import patch_create_graphs_mp


def pytest_addoption(parser):
    parser.addoption("--regen-graphs", action="store_true")


@pytest.fixture(scope="session")
def graph_artifacts_dir(request):
    repo_root = Path(__file__).resolve().parents[2]
    requirements_file_path = repo_root / "ShIeLD" / "tests" / "data" / "requirements.pt"

    cache_dir = repo_root / ".pytest_cache" / "shield_artifacts"
    out_root = cache_dir / "data_sets"

    if request.config.getoption("--regen-graphs") and out_root.exists():
        shutil.rmtree(out_root)

    if list(out_root.rglob("*.pt")):
        return out_root

    out_root.mkdir(parents=True, exist_ok=True)

    requirements = pickle.load(open(requirements_file_path, "rb"))
    requirements["path_to_data_set"] = out_root

    req_path = cache_dir / "requirements_test.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(req_path, "wb") as f:
        pickle.dump(requirements, f)

    from ShIeLD import create_graphs

    patch_create_graphs_mp(create_graphs)

    for segmentation_type in ["random", "voronoi"]:
        sys.argv = [
            "create_graphs.py",
            "--requirements_file_path",
            str(req_path),
            "--data_set_type",
            "test",
            "--segmentation",
            segmentation_type,
            "--column_celltype_name",
            "cell_type",
            "--noisy_labeling",
            "False",
            "--node_prob",
            "False",
            "--randomise_edges",
            "False",
            "--reduce_population",
            "False",
            "--reverse_sampling",
            "False",
            "--max_graphs",
            "5",
        ]
        create_graphs.main()

    # global smoke assert (don’t assert every config combo when max_graphs caps output)
    pt_files = list(out_root.rglob("*.pt"))
    assert pt_files, f"No .pt graphs written under: {out_root}"
    return out_root
