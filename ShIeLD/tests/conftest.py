# ShIeLD/ShIeLD/tests/conftest.py
#
# Pytest configuration + integration-test fixtures for SHIELD.
#
# Goal
# ----
# Provide CI-safe, deterministic end-to-end smoke tests for the core pipeline:
#   1) create_graphs  -> writes graph .pt files
#   2) train_model    -> writes a training results CSV
#   3) evaluate_hypersearch -> aggregates results + writes best_config + best_model
#   4) create_attention_score_interactions -> writes interaction dict + plots
#
# Strategy
# --------
# - Use a per-session temporary artifact directory to cache intermediate outputs.
# - Monkeypatch expensive/parallel pieces to run serially in CI.
# - Avoid GUI backends for matplotlib to prevent hangs.
# - Keep runtime small by capping graphs and training to minimal work.
#
# Optional flags
# --------------
# --keep-artifacts : copy temp artifacts into .pytest_cache/shield_artifacts_last
# --regen-graphs   : force regeneration of graph artifacts even if cached

import pickle
import shutil
import sys
from pathlib import Path, PosixPath
import tempfile
import pytest


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _find_repo_root(start: PosixPath) -> PosixPath:
    """
    Heuristically locate the *outer* repo root by walking upwards.

    Why:
    - Depending on how pytest is invoked, the rootdir can be `SHIELD/` or
      `SHIELD/ShIeLD/`.
    - We need a stable reference point to find test data and write artifacts.

    Heuristic:
    - The first parent containing either `pyproject.toml` or `.git` is considered
      the repo root.
    - If neither is found, fall back to `start.parents[3]` (best-effort).
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # fallback: assume 3 levels up is repo
    return start.parents[3]


class _SerialPool:
    """
    Minimal drop-in replacement for multiprocessing.Pool used in tests.

    Why:
    - CI environments can be flaky with multiprocessing (spawn/fork differences,
      pickling issues, limited resources).
    - We want deterministic, fast integration tests.

    Interface:
    - Supports context manager usage and `imap_unordered`.
    - Executes work serially in the current process.
    """

    def __init__(self, *a, **k): ...
    def __enter__(self):
        return self

    def __exit__(self, *a):
        # Returning False means exceptions (if any) propagate.
        return False

    def imap_unordered(self, func, iterable, chunksize=1):
        # Ignore chunksize; yield results in input order.
        for x in iterable:
            yield func(x)


def _safe_tqdm(module):
    """
    Replace tqdm progress bars with a no-op iterator wrapper.

    Why:
    - Progress bars add noise to CI logs and can slow down test output rendering.
    - Some environments handle carriage returns poorly.

    Implementation:
    - Best-effort: if the module has a `tqdm` attribute, replace it.
    """
    try:
        module.tqdm = lambda x, **k: x
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Global / autouse test environment fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _no_gui_plots():
    """
    Force matplotlib into a headless backend for the entire test session.

    Why:
    - GUI backends can hang in CI.
    - Some plotting utilities call `plt.show()` which can block.

    What we do:
    - Set MPLBACKEND=Agg (if not already set)
    - Force matplotlib backend to Agg
    - Disable interactive mode (plt.ioff)
    - Monkeypatch plt.show to a no-op during tests
    - Ensure all figures are closed at teardown
    """
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib

    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    plt.ioff()

    # Make show a no-op so nothing blocks in any test environment.
    _old_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.show = _old_show
        plt.close("all")


@pytest.fixture(scope="session")
def artifacts_root(tmp_path_factory, request):
    """
    Per-session artifact directory.

    Default behavior:
    - Create a real OS temp directory (not pytest's tmp_path) so that:
      * large artifacts can be written without surprises,
      * cleanup is reliable.

    Optional behavior:
    - If `--keep-artifacts` is set, copy the temp directory to:
        <repo>/.pytest_cache/shield_artifacts_last
      This makes debugging failing CI runs much easier.
    """
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


# -----------------------------------------------------------------------------
# Pytest CLI options
# -----------------------------------------------------------------------------
def pytest_addoption(parser):
    """
    Register custom pytest flags for this test suite.

    --keep-artifacts:
        Keep the session artifact directory after tests by copying it into
        .pytest_cache/shield_artifacts_last.

    --regen-graphs:
        Force regeneration of graphs even if they already exist in the session
        artifact directory.
    """
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


# -----------------------------------------------------------------------------
# Core pipeline fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def graph_artifacts_dir(request, artifacts_root) -> Path:
    """
    Create and cache a minimal set of graphs once per test session.

    Output location:
        <artifacts_root>/data_sets

    Behavior:
    - If `--regen-graphs` is passed, the folder is removed and regenerated.
    - If graphs already exist (*.pt found), reuse and return immediately.

    CI-safety:
    - Monkeypatch SHIELD's create_graphs multiprocessing pool to run serially.
    - Cap graph creation via `--max_graphs 3`.
    - Disable tqdm progress bars.

    Returns
    -------
    Path
        The directory that contains the generated dataset folder hierarchy.
    """
    repo_root = _find_repo_root(Path(__file__).resolve())
    cache_dir = artifacts_root
    out_root = cache_dir / "data_sets"

    regen = request.config.getoption("--regen-graphs")
    if regen and out_root.exists():
        shutil.rmtree(out_root)

    # Reuse if already created (fast path).
    if out_root.exists() and list(out_root.rglob("*.pt")):
        return out_root

    # requirements file lives in: repo_root/ShIeLD/tests/data/requirements.pt
    base_req = repo_root / "ShIeLD" / "tests" / "data" / "requirements.pt"
    if not base_req.exists():
        raise FileNotFoundError(f"Missing requirements.pt at: {base_req}")

    out_root.mkdir(parents=True, exist_ok=True)

    # Patch requirements so graphs are written into the session artifact directory,
    # not into whatever default path the project uses.
    req = pickle.load(open(base_req, "rb"))
    req["path_to_data_set"] = out_root  # critical: write into cached artifact folder

    patched_req_path = cache_dir / "requirements_test.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(patched_req_path, "wb") as f:
        pickle.dump(req, f)

    # Run create_graphs in a CI-safe way.
    from ShIeLD import create_graphs

    # Enforce serial behavior and predictable CPU count.
    create_graphs.mp.Pool = _SerialPool
    create_graphs.mp.cpu_count = lambda: 2
    create_graphs.mp.set_start_method = lambda *a, **k: None
    _safe_tqdm(create_graphs)

    # Generate graphs for both segmentation strategies and both train/test splits.
    for segmentation in ["voronoi", "bucket"]:
        for data_type in ["train", "test"]:
            # Simulate CLI invocation by setting sys.argv.
            sys.argv = [
                "create_graphs.py",
                "--requirements_file_path",
                str(patched_req_path),
                "--data_set_type",
                data_type,
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
                "--testing_mode",
                "True",
            ]
            create_graphs.main()

    assert list(out_root.rglob("*.pt")), f"No graphs written under: {out_root}"
    return out_root


@pytest.fixture(scope="session")
def training_artifacts_dir(graph_artifacts_dir, artifacts_root) -> Path:
    """
    Run a minimal/smoke training once per session and cache the outputs.

    Output location:
        <artifacts_root>/training_results

    Produces:
    - validation_split_results_training.csv (must exist and be non-empty)
    - requirements_train_smoke.pkl (patched requirements for downstream steps)

    CI-safety reductions:
    - batch_size=1
    - patched train loop to only run a single batch and return immediately
    - DataListLoader num_workers=0 (no multiprocessing workers in CI)
    - tqdm replaced with no-op
    """
    repo_root = _find_repo_root(Path(__file__).resolve())
    cache_dir = artifacts_root
    out_dir = cache_dir / "training_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "validation_split_results_training.csv"
    req_path = cache_dir / "requirements_train_smoke.pkl"

    # Reuse if already created.
    if results_csv.exists() and results_csv.stat().st_size > 0 and req_path.exists():
        return out_dir

    base_req = repo_root / "ShIeLD" / "tests" / "data" / "requirements.pt"
    if not base_req.exists():
        raise FileNotFoundError(f"Missing requirements.pt at: {base_req}")

    # Patch requirements to point at our cached graphs and output folder.
    req = pickle.load(open(base_req, "rb"))
    req["path_to_data_set"] = Path(graph_artifacts_dir)
    req["path_training_results"] = out_dir
    req["batch_size"] = 1
    req["patience"] = 9

    # Ensure validation split IDs match what exists in the CSV (robust across test data).
    import pandas as pd

    csv = pd.read_csv(req["path_raw_data"])
    all_folds = sorted(csv[req["validation_split_column"]].unique().tolist())
    req["number_validation_splits"] = all_folds
    split_number = 0

    # Save patched requirements for reuse by later fixtures/tests.
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(req_path, "wb") as f:
        pickle.dump(req, f)

    # Ensure file-name list exists so graph_dataset is non-empty.
    # Prefer a Voronoi directory if available (fussy_limit_ in path), else take any.
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
    config_root = graphs_dir.parents[1]  # radius_* directory above train/graphs
    train_list_pkl = (
        config_root / f"train_set_validation_split_{split_number}_file_names.pkl"
    )

    # Prefer graph files matching your naming convention, fallback to any .pt.
    pt_files = [p.name for p in graphs_dir.glob("*.pt") if "graph_subSample_" in p.name]
    if not pt_files:
        pt_files = [p.name for p in graphs_dir.glob("*.pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt graphs found in: {graphs_dir}")

    with open(train_list_pkl, "wb") as f:
        pickle.dump(pt_files, f)

    # Run train_model, with heavy parts monkeypatched for CI.
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        from ShIeLD import train_model
        import ShIeLD.utils.train_utils as train_utils
        from torch_geometric.loader import DataListLoader as _RealDataListLoader

        # Make DataListLoader CI-safe:
        # - num_workers=0 avoids subprocesses
        # - prefetch_factor only valid when num_workers>0, so remove it
        def _SafeDataListLoader(*args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs.pop("prefetch_factor", None)
            return _RealDataListLoader(*args, **kwargs)

        mp.setattr(train_model, "DataListLoader", _SafeDataListLoader)
        mp.setattr(train_model, "tqdm", lambda x, **k: x)

        # Replace the full training loop with a single-batch update (smoke-test).
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

        # Simulate CLI invocation.
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
    Run evaluate_hypersearch once per session and cache the outputs.

    Output location:
        <artifacts_root>/hypersearch_results

    Preconditions:
    - training_artifacts_dir must have produced validation_split_results_training.csv
    - requirements_train_smoke.pkl exists (patched requirements)

    CI-safety reductions:
    - Disable plotting by patching create_parameter_influence_plots to a no-op
    - Disable tqdm
    - Keep number_of_training_repeats small
    """
    cache_dir = artifacts_root
    out_dir = cache_dir / "hypersearch_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"
    out_csv = out_dir / "hyper_search_results.csv"

    # Determine if a best-config artifact exists (name patterns can vary slightly).
    patterns = [
        "*best*config*.pt",
        "*best*config*.pkl",
        "*best_config*.pt",
        "*best_config*.pkl",
    ]
    has_best_cfg = any(any(model_dir.rglob(p)) for p in patterns)

    # Reuse if already produced.
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

    # Patch requirements to route outputs into this fixture's folder.
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

        # Some test runs don't need plots; keep CI lean and stable.
        try:
            mp.setattr(
                evaluate_hypersearch.evaluation_utils,
                "create_parameter_influence_plots",
                lambda **kwargs: None,
            )
        except Exception:
            pass

        # Make sure evaluate_hypersearch sees the training CSV where it expects it:
        # requirements["path_training_results"]/validation_split_results_training.csv
        import shutil

        src_csv = Path(training_artifacts_dir) / "validation_split_results_training.csv"
        dst_csv = out_dir / "validation_split_results_training.csv"

        if not src_csv.exists():
            raise FileNotFoundError(f"Training CSV missing at: {src_csv}")

        shutil.copy2(src_csv, dst_csv)
        assert dst_csv.exists() and dst_csv.stat().st_size > 0

        # Simulate CLI invocation.
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
        evaluate_hypersearch.main()
    finally:
        mp.undo()

    assert out_csv.exists() and out_csv.stat().st_size > 0, f"Missing/empty: {out_csv}"
    return out_dir


@pytest.fixture(scope="session")
def attention_artifacts_dir(hypersearch_artifacts_dir, artifacts_root) -> Path:
    """
    Run create_attention_score_interactions once per session and cache outputs.

    Output location:
        <artifacts_root>/attention_results

    Preconditions:
    - hypersearch_artifacts_dir produced:
        * best_model.pt and a best_config artifact under <hypersearch>/model

    CI-safety reductions:
    - Disable tqdm
    - Keep plots/headless via _no_gui_plots fixture
    """
    cache_dir = artifacts_root
    out_dir = cache_dir / "attention_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse if already produced (any non-empty file under out_dir counts).
    if any(p.is_file() and p.stat().st_size > 0 for p in out_dir.rglob("*")):
        return out_dir

    req_path = cache_dir / "requirements_train_smoke.pkl"
    if not req_path.exists():
        raise FileNotFoundError(f"Missing patched req: {req_path}")

    # Patch requirements to route interaction plots into this fixture's folder.
    req = pickle.load(open(req_path, "rb"))
    req["path_training_results"] = hypersearch_artifacts_dir
    model_dir = Path(hypersearch_artifacts_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    req["path_to_model"] = model_dir

    # Locate best-config artifact created by evaluate_hypersearch.
    candidates = list(Path(req["path_to_model"]).rglob("*best*config*.pt")) + list(
        Path(req["path_to_model"]).rglob("*best*config*.pkl")
    )
    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        raise FileNotFoundError(
            f"No best-config artifact found under: {req['path_to_model']}"
        )
    best_cfg = sorted(candidates)[0]

    # Route outputs into cache.
    req["path_to_interaction_plots"] = out_dir
    patched_req = out_dir / "requirements_attention_smoke.pkl"
    with open(patched_req, "wb") as f:
        pickle.dump(req, f)

    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        from ShIeLD import create_attention_score_interactions

        _safe_tqdm(create_attention_score_interactions)

        # Simulate CLI invocation.
        mp.setattr(
            sys,
            "argv",
            [
                "create_attention_score_interactions.py",
                "--requirements_file_path",
                str(patched_req),
                "--best_config_dict_path",
                str(best_cfg),
                "--recalculate_cTc_Scores",
                "True",
                "--cellTypeColumnName",
                "cell_type",
            ],
        )
        create_attention_score_interactions.main()
    finally:
        mp.undo()

    # Sanity checks: ensure interaction dict and at least one plot exist.
    dict_path = (
        Path(req["path_to_model"]) / "cT_t_cT_interactions_dict_test.pt"
    )  # or train depending on args
    pngs = list(out_dir.rglob("*.png"))

    assert dict_path.exists() and dict_path.stat().st_size > 0, (
        f"Missing interaction dict: {dict_path}"
    )
    assert len(pngs) > 0, f"No plots created under: {out_dir}"
    return out_dir
