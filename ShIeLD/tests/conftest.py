# ShIeLD/tests/conftest.py
import pickle
import shutil
import sys
from pathlib import Path

import pytest


class _SerialPool:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, func, iterable, chunksize=1):
        for x in iterable:
            yield func(x)


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

    create_graphs.mp.Pool = _SerialPool
    create_graphs.mp.cpu_count = lambda: 2
    create_graphs.mp.set_start_method = lambda *a, **k: None

    for segmentation_type in ["voronoi", "random"]:
        for data_type in ["train", "test"]:
            sys.argv = [
                "create_graphs.py",
                "--requirements_file_path",
                str(req_path),
                "--data_set_type",
                data_type,
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
                "2",
                "--skip_existing",
                "False",
            ]
            create_graphs.main()

    # global smoke assert (don’t assert every config combo when max_graphs caps output)
    pt_files = list(out_root.rglob("*.pt"))
    assert pt_files, f"No .pt graphs written under: {out_root}"
    return out_root
