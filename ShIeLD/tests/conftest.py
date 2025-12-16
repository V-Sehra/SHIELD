# ShIeLD/ShIeLD/tests/conftest.py

import pickle
import shutil
import sys
from pathlib import Path
import tempfile
import pytest


# -------------------------
# Helpers
# -------------------------
def _find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find the outer repo root (pyproject.toml or .git).
    Works regardless of whether pytest rootdir is SHIELD/ or SHIELD/ShIeLD/.
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # fallback: assume 3 levels up is repo
    return start.parents[3]


@pytest.fixture(scope="session")
def artifacts_root(tmp_path_factory, request) -> Path:
    """
    A per-test-session artifact root. Default: deleted after test run.

    If --keep-artifacts is set, we copy the temp dir to:
      <repo>/.pytest_cache/shield_artifacts_last
    """
    # Use a *real* OS temp dir so we can reliably delete it
    td = tempfile.TemporaryDirectory(prefix="shield_artifacts_")
    root = Path(td.name)

    yield root

    if request.config.getoption("--keep-artifacts"):
        repo_root = _find_repo_root(Path(__file__).resolve())
        dst = repo_root / ".pytest_cache" / "shield_artifacts_last"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, dst)

    td.cleanup()


@pytest.fixture(scope="session", autouse=True)
def _no_gui_plots():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib

    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    plt.ioff()

    # make show a no-op so nothing blocks
    _old_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.show = _old_show
        plt.close("all")


class _SerialPool:
    def __init__(self, *a, **k): ...
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, func, iterable, chunksize=1):
        for x in iterable:
            yield func(x)


def _safe_tqdm(module):
    try:
        module.tqdm = lambda x, **k: x
    except Exception:
        pass


# -------------------------
# Core pipeline fixtures
# -------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--keep-artifacts",
        action="store_true",
        default=False,
        help="Keep integration artifacts for debugging (copies them into .pytest_cache/shield_artifacts_last).",
    )
    parser.addoption(
        "--regen-graphs",
        action="store_true",
        default=False,
        help="Regenerate test graphs even if they already exist.",
    )


@pytest.fixture(scope="session")
def graph_artifacts_dir(request, artifacts_root) -> Path:
    """
    Create and cache graphs once under .pytest_cache/shield_artifacts/data_sets
    Re-used across all integration tests.
    """
    repo_root = _find_repo_root(Path(__file__).resolve())
    cache_dir = artifacts_root
    out_root = cache_dir / "data_sets"

    regen = request.config.getoption("--regen-graphs")
    if regen and out_root.exists():
        shutil.rmtree(out_root)

    # reuse if already created
    if out_root.exists() and list(out_root.rglob("*.pt")):
        return out_root

    # requirements file lives in: repo_root/ShIeLD/tests/data/requirements.pt
    base_req = repo_root / "ShIeLD" / "tests" / "data" / "requirements.pt"
    if not base_req.exists():
        raise FileNotFoundError(f"Missing requirements.pt at: {base_req}")

    out_root.mkdir(parents=True, exist_ok=True)

    req = pickle.load(open(base_req, "rb"))
    req["path_to_data_set"] = out_root  # critical: write into cached artifact folder

    patched_req_path = cache_dir / "requirements_test.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(patched_req_path, "wb") as f:
        pickle.dump(req, f)

    # run create_graphs in a CI-safe way
    from ShIeLD import create_graphs

    # enforce serial behavior
    create_graphs.mp.Pool = _SerialPool
    create_graphs.mp.cpu_count = lambda: 2
    create_graphs.mp.set_start_method = lambda *a, **k: None
    _safe_tqdm(create_graphs)

    for segmentation in ["voronoi", "random"]:
        for data_type in ["train", "test"]:
            sys.argv = [
                "create_graphs.py",
                "--requirements_file_path",
                str(patched_req_path),
                "--data_set_type",
                data_type,
                "--segmentation",
                segmentation,
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
                "3",
            ]
            create_graphs.main()

    assert list(out_root.rglob("*.pt")), f"No graphs written under: {out_root}"
    return out_root


@pytest.fixture(scope="session")
def training_artifacts_dir(graph_artifacts_dir, artifacts_root) -> Path:
    """
    Run train_model once (smoke) and cache:
      - training_results/validation_split_results_training.csv
      - requirements_train_smoke.pkl (for downstream steps)
    """
    repo_root = _find_repo_root(Path(__file__).resolve())
    cache_dir = artifacts_root
    out_dir = cache_dir / "training_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "validation_split_results_training.csv"
    req_path = cache_dir / "requirements_train_smoke.pkl"

    if results_csv.exists() and results_csv.stat().st_size > 0 and req_path.exists():
        return out_dir

    base_req = repo_root / "ShIeLD" / "tests" / "data" / "requirements.pt"
    if not base_req.exists():
        raise FileNotFoundError(f"Missing requirements.pt at: {base_req}")

    req = pickle.load(open(base_req, "rb"))
    req["path_to_data_set"] = Path(graph_artifacts_dir)
    req["path_training_results"] = out_dir
    req["batch_size"] = 1
    req["patience"] = 9

    # keep your known-good “catch-all split” style (no list-of-lists!)
    import pandas as pd

    csv = pd.read_csv(req["path_raw_data"])
    all_folds = sorted(csv[req["validation_split_column"]].unique().tolist())
    req["number_validation_splits"] = all_folds
    split_number = 0

    # save patched req for reuse by later fixtures/tests
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(req_path, "wb") as f:
        pickle.dump(req, f)

    # ensure file-name list exists so graph_dataset is non-empty
    # pick *any* train/graphs folder (voronoi preferred)
    train_graph_dirs = sorted(
        p
        for p in Path(graph_artifacts_dir).rglob("train/graphs")
        if "fussy_limit_" in str(p)
    )
    if not train_graph_dirs:
        train_graph_dirs = sorted(Path(graph_artifacts_dir).rglob("train/graphs"))
    if not train_graph_dirs:
        raise FileNotFoundError(f"No train/graphs found under: {graph_artifacts_dir}")

    graphs_dir = train_graph_dirs[0]
    config_root = graphs_dir.parents[1]  # radius_*
    train_list_pkl = (
        config_root / f"train_set_validation_split_{split_number}_file_names.pkl"
    )
    pt_files = [p.name for p in graphs_dir.glob("*.pt") if "graph_subSample_" in p.name]
    if not pt_files:
        pt_files = [p.name for p in graphs_dir.glob("*.pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt graphs found in: {graphs_dir}")

    with open(train_list_pkl, "wb") as f:
        pickle.dump(pt_files, f)

    # run train_model
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        from ShIeLD import train_model
        import ShIeLD.utils.train_utils as train_utils
        from torch_geometric.loader import DataListLoader as _RealDataListLoader

        # CI-safe loader
        def _SafeDataListLoader(*args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs.pop("prefetch_factor", None)
            return _RealDataListLoader(*args, **kwargs)

        mp.setattr(train_model, "DataListLoader", _SafeDataListLoader)
        mp.setattr(train_model, "tqdm", lambda x, **k: x)

        # one-batch train loop
        def _train_loop_one_batch(
            *,
            optimizer,
            model,
            data_loader,
            loss_fkt,
            attr_bool,
            device,
            patience,
            noise_yLabel,
            **kw,
        ):
            for train_sample in data_loader:
                optimizer.zero_grad()
                from ShIeLD.utils import model_utils

                _, _, out, y, _ = model_utils.prediction_step(
                    batch_sample=train_sample,
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
                str(req_path),
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
        f"Missing/empty: {results_csv}"
    )
    return out_dir


@pytest.fixture(scope="session")
def hypersearch_artifacts_dir(training_artifacts_dir, artifacts_root) -> Path:
    """
    Run evaluate_hypersearch once; cache under .pytest_cache/shield_artifacts/hypersearch_results
    """

    cache_dir = artifacts_root
    out_dir = cache_dir / "hypersearch_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"
    out_csv = out_dir / "hyper_search_results.csv"

    patterns = [
        "*best*config*.pt",
        "*best*config*.pkl",
        "*best_config*.pt",
        "*best_config*.pkl",
    ]
    has_best_cfg = any(any(model_dir.rglob(p)) for p in patterns)

    if (
        out_csv.exists()
        and out_csv.is_file()
        and out_csv.stat().st_size > 0
        and model_dir.exists()
        and has_best_cfg
    ):
        return out_dir

    req_path = cache_dir / "requirements_train_smoke.pkl"
    if not req_path.exists():
        raise FileNotFoundError(f"Missing patched req: {req_path}")

    req = pickle.load(open(req_path, "rb"))
    req["path_training_results"] = out_dir
    req["path_to_model"] = model_dir
    req["patience"] = 9
    req["path_to_model"].mkdir(parents=True, exist_ok=True)

    patched_req = out_dir / "requirements_eval_hypersearch.pkl"
    with open(patched_req, "wb") as f:
        pickle.dump(req, f)

    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        from ShIeLD import evaluate_hypersearch

        _safe_tqdm(evaluate_hypersearch)
        try:
            mp.setattr(
                evaluate_hypersearch.evaluation_utils,
                "create_parameter_influence_plots",
                lambda **kwargs: None,
            )
        except Exception:
            pass

        mp.setattr(
            sys,
            "argv",
            [
                "evaluate_hypersearch.py",
                "--requirements_file_path",
                str(patched_req),
                "--retain_best_model_config_bool",
                "True",
                "--best_config_dict_path",
                str(model_dir / "best_config.pt"),
                "--number_of_training_repeats",
                "2",
            ],
        )
        import shutil

        src_csv = Path(training_artifacts_dir) / "validation_split_results_training.csv"
        dst_csv = out_dir / "validation_split_results_training.csv"

        if not src_csv.exists():
            raise FileNotFoundError(f"Training CSV missing at: {src_csv}")

        # Ensure evaluate_hypersearch finds it where it looks (path_training_results)
        shutil.copy2(src_csv, dst_csv)
        assert dst_csv.exists() and dst_csv.stat().st_size > 0
        evaluate_hypersearch.main()
    finally:
        mp.undo()

    assert out_csv.exists() and out_csv.stat().st_size > 0, f"Missing/empty: {out_csv}"
    return out_dir


@pytest.fixture(scope="session")
def attention_artifacts_dir(hypersearch_artifacts_dir, artifacts_root) -> Path:
    """
    Run create_attention_score_interactions once; cache under .pytest_cache/shield_artifacts/attention_results
    """

    cache_dir = artifacts_root
    out_dir = cache_dir / "attention_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # reuse if already produced
    if any(p.is_file() and p.stat().st_size > 0 for p in out_dir.rglob("*")):
        return out_dir

    req_path = cache_dir / "requirements_train_smoke.pkl"
    if not req_path.exists():
        raise FileNotFoundError(f"Missing patched req: {req_path}")
    req = pickle.load(open(req_path, "rb"))
    req["path_training_results"] = hypersearch_artifacts_dir
    model_dir = Path(hypersearch_artifacts_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)  # <-- add this
    req["path_to_model"] = model_dir

    # find best config created by evaluate_hypersearch (adjust if your filename differs)
    candidates = list(Path(req["path_to_model"]).rglob("*best*config*.pt")) + list(
        Path(req["path_to_model"]).rglob("*best*config*.pkl")
    )

    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        raise FileNotFoundError(
            f"No best-config artifact found under: {req['path_to_model']}"
        )
    best_cfg = sorted(candidates)[0]

    # route outputs into cache (if your script uses a different key, set it here)
    req["path_to_interaction_plots"] = out_dir
    patched_req = out_dir / "requirements_attention_smoke.pkl"
    with open(patched_req, "wb") as f:
        pickle.dump(req, f)

    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        from ShIeLD import create_attention_score_interactions

        _safe_tqdm(create_attention_score_interactions)

        mp.setattr(
            sys,
            "argv",
            [
                "create_attention_score_interactions.py",
                "--requirements_file_path",
                str(patched_req),
                "--best_config_dict_path",
                str(best_cfg),
                "--recalculate_cTc_Scroes",
                "True",
                "--cellTypeColumnName",
                "cell_type",
            ],
        )
        create_attention_score_interactions.main()
    finally:
        mp.undo()

    model_dir = Path(req["path_to_model"])
    dict_path = (
        model_dir / "cT_t_cT_interactions_dict_test.pt"
    )  # or train depending on args

    pngs = list(out_dir.rglob("*.png"))

    assert dict_path.exists() and dict_path.stat().st_size > 0, (
        f"Missing interaction dict: {dict_path}"
    )
    assert len(pngs) > 0, f"No plots created under: {out_dir}"
    return out_dir
