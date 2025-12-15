import pickle
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_train_model_smoke(graph_artifacts_dir, monkeypatch, tmp_path):
    """
    Uses cached graph artifacts to run a minimal train_model.main() once.
    Asserts that training results CSV is written and non-empty.
    """

    repo_root = Path(__file__).resolve().parents[3]
    base_req_path = repo_root / "ShIeLD" / "tests" / "data" / "requirements.pt"
    assert base_req_path.exists()

    # --- Patch requirements for test speed & deterministic IO ---
    req = pickle.load(open(base_req_path, "rb"))
    req["path_training_results"] = tmp_path / "training_results"

    # pick a voronoi config (has fussy_limit_*)
    train_graph_dirs = sorted(
        p
        for p in Path(graph_artifacts_dir).rglob("train/graphs")
        if "fussy_limit_" in str(p)
    )
    assert train_graph_dirs, f"No voronoi train/graphs dirs under {graph_artifacts_dir}"
    graphs_dir = train_graph_dirs[
        0
    ]  # .../anker_value_X/min_cells_Y/fussy_limit_Z/radius_R/train/graphs

    radius_dir = graphs_dir.parents[1]  # radius_25
    fussy_dir = graphs_dir.parents[2]  # fussy_limit_0_2
    mincells_dir = graphs_dir.parents[3]  # min_cells_50
    anker_dir = graphs_dir.parents[4]  # anker_value_0_03

    def _parse_float(tag: str, name: str) -> float:
        return float(name.replace(tag, "").replace("_", "."))

    req["radius_distance_all"] = [int(radius_dir.name.replace("radius_", ""))]
    req["fussy_limit_all"] = [_parse_float("fussy_limit_", fussy_dir.name)]
    req["minimum_number_cells"] = int(mincells_dir.name.replace("min_cells_", ""))
    req["anker_value_all"] = [_parse_float("anker_value_", anker_dir.name)]

    # Make it tiny (1 config, 1 model)
    if "droupout_rate" in req:
        req["droupout_rate"] = [req["droupout_rate"][0]]
    req["batch_size"] = 1

    # early_stopping enforces patience >= 8
    req["patience"] = 8

    # --- Make split selection smoke-test stable ---
    # In some dummy configs, number_validation_splits isn't a list-of-lists.
    # For a smoke test we force a single split that includes all folds present in CSV.
    import pandas as pd

    csv = pd.read_csv(req["path_raw_data"])
    all_folds = sorted(csv[req["validation_split_column"]].unique().tolist())
    req["number_validation_splits"] = (
        all_folds  # one "catch-all" split -> use split_number=0
    )

    split_number = 0  # keep consistent with sys.argv below

    # need to find the temp file from .pytest:
    # find the first ancestor that has Shield/utils
    base = next(
        (
            b
            for b in (graphs_dir, *graphs_dir.parents)
            if (b / ".pytest_cache").is_dir()
        ),
        None,
    )
    if base is None:
        raise FileNotFoundError(".pytest_cache not found upwards.")

    data_dir = base
    req["path_to_data_set"] = (
        data_dir / ".pytest_cache" / "shield_artifacts" / "data_sets"
    )
    # --- Persist patched requirements and run train_model with it ---
    patched_req_path = tmp_path / "requirements_train_smoke.pkl"

    with open(patched_req_path, "wb") as f:
        pickle.dump(req, f)

    # file list pkl is stored one level above train/graphs (i.e. the radius_* folder)
    config_root = graphs_dir.parents[1]  # .../radius_25

    train_list_pkl = (
        config_root / f"train_set_validation_split_{split_number}_file_names.pkl"
    )
    import os

    pt_files = [file for file in os.listdir(graphs_dir) if "graph_subSample_" in file]

    assert pt_files, f"No .pt graphs found in {graphs_dir}"

    # Pick a few graphs to guarantee non-empty dataset
    with open(train_list_pkl, "wb") as f:
        pickle.dump(pt_files, f)

    from ShIeLD import train_model
    import ShIeLD.utils.train_utils as train_utils
    from torch_geometric.loader import DataListLoader as _RealDataListLoader

    # --- Make loaders CI-safe ---
    def _SafeDataListLoader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)  # can break with num_workers=0
        return _RealDataListLoader(*args, **kwargs)

    monkeypatch.setattr(train_model, "DataListLoader", _SafeDataListLoader)

    # --- Kill tqdm noise in tests ---
    monkeypatch.setattr(train_model, "tqdm", lambda x, **k: x)

    # --- Force serial multiprocessing settings (if anything tries) ---
    monkeypatch.setattr(
        train_model.torch.multiprocessing, "set_start_method", lambda *a, **k: None
    )
    monkeypatch.setattr(
        train_model.torch.multiprocessing, "set_sharing_strategy", lambda *a, **k: None
    )

    # --- Run only 1 epoch / 1 batch, but do a REAL forward/backward ---
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

    monkeypatch.setattr(train_utils, "train_loop_shield", _train_loop_one_batch)

    # Optional sanity print (safe, no fold logic)
    pt_names = [p.name for p in graphs_dir.glob("*.pt")]
    assert pt_names, f"No .pt graphs found in {graphs_dir}"
    # print(f"Using graphs_dir={graphs_dir} with {len(pt_names)} graphs")

    # --- Run train_model.main() like CLI ---
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--requirements_file_path",
            str(patched_req_path),
            "--split_number",
            f"{split_number}",  # catch-all split
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

    # --- Assert: results CSV created and has at least one row ---
    results_csv = req["path_training_results"] / "validation_split_results_training.csv"
    assert results_csv.exists(), f"Missing results CSV: {results_csv}"
    assert results_csv.stat().st_size > 0, "Results CSV is empty"
