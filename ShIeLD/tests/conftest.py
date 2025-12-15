# ShIeLD/tests/conftest.py
import pickle
import shutil
import sys
from pathlib import Path
import pandas as pd
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
    parser.addoption(
        "--regen-train",
        action="store_true",
        help="Regenerate cached training artifacts even if they already exist.",
    )


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


@pytest.fixture(scope="session")
def training_artifacts_dir(graph_artifacts_dir, request, tmp_path_factory):
    """
    Ensures training has been run once on cached graph artifacts.
    Returns a folder that contains: validation_split_results_training.csv
    """
    repo_root = Path(__file__).resolve().parents[2]  # .../ShIeLD

    cache_dir = repo_root / ".pytest_cache" / "shield_artifacts"
    out_dir = cache_dir / "training_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "validation_split_results_training.csv"
    if results_csv.exists() and results_csv.stat().st_size > 0:
        return out_dir

    # --- patch requirements ---
    base_req_path = repo_root / "ShIeLD" / "tests" / "data" / "requirements.pt"
    req = pickle.load(open(base_req_path, "rb"))
    req["path_to_data_set"] = Path(graph_artifacts_dir)
    req["path_training_results"] = out_dir
    req["batch_size"] = 1
    req["patience"] = 8

    # make split selection robust for dummy CSV
    csv = pd.read_csv(req["path_raw_data"])
    all_folds = sorted(csv[req["validation_split_column"]].unique().tolist())
    req["number_validation_splits"] = all_folds  # “catch-all” split
    split_number = 0

    patched_req_path = cache_dir / "requirements_train_smoke.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(patched_req_path, "wb") as f:
        pickle.dump(req, f)

    # --- pick a graphs_dir and write the file list pkl the dataset expects ---
    train_graph_dirs = sorted(
        p
        for p in Path(graph_artifacts_dir).rglob("train/graphs")
        if "fussy_limit_" in str(p)
    )
    assert train_graph_dirs, f"No voronoi train/graphs dirs under {graph_artifacts_dir}"
    graphs_dir = train_graph_dirs[0]
    config_root = graphs_dir.parents[1]  # radius_*
    train_list_pkl = (
        config_root / f"train_set_validation_split_{split_number}_file_names.pkl"
    )

    pt_files = [p.name for p in graphs_dir.glob("*.pt") if "graph_subSample_" in p.name]
    assert pt_files, f"No .pt graphs found in {graphs_dir}"
    with open(train_list_pkl, "wb") as f:
        pickle.dump(pt_files, f)

    # --- run train_model with a local MonkeyPatch (since fixture is session-scoped) ---
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        from ShIeLD import train_model
        import ShIeLD.utils.train_utils as train_utils
        from torch_geometric.loader import DataListLoader as _RealDataListLoader

        def _SafeDataListLoader(*args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs.pop("prefetch_factor", None)
            return _RealDataListLoader(*args, **kwargs)

        mp.setattr(train_model, "DataListLoader", _SafeDataListLoader)
        mp.setattr(train_model, "tqdm", lambda x, **k: x)
        mp.setattr(
            train_model.torch.multiprocessing, "set_start_method", lambda *a, **k: None
        )
        mp.setattr(
            train_model.torch.multiprocessing,
            "set_sharing_strategy",
            lambda *a, **k: None,
        )

        def _train_loop_one_batch(
            *, optimizer, model, data_loader, loss_fkt, noise_yLabel, device, **kw
        ):
            for batch in data_loader:
                optimizer.zero_grad()
                from ShIeLD.utils import model_utils

                _, _, out, y, _ = model_utils.prediction_step(
                    batch_sample=batch,
                    model=model,
                    device=device,
                    per_patient=False,
                    noise_yLabel=noise_yLabel,
                )
                loss = loss_fkt(out, y)
                loss.backward()
                optimizer.step()
                return model, [float(loss.item())]
            return model, [0.0]

        mp.setattr(train_utils, "train_loop_shield", _train_loop_one_batch)

        mp.setattr(
            sys,
            "argv",
            [
                "train_model.py",
                "--requirements_file_path",
                str(patched_req_path),
                "--split_number",
                str(split_number),
                "--number_of_training_repeats",
                "1",
                "--comment",
                "pytest",
                "--noisy_edge",
                "False",
                "--noise_yLabel",
                "False",
                "--reverse_sampling",
                "False",
            ],
        )

        train_model.main()

    finally:
        mp.undo()

    assert results_csv.exists() and results_csv.stat().st_size > 0, (
        "Training did not write a results CSV."
    )
    return out_dir
