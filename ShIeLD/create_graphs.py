#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHIELD graph generation script (CSV -> per-region graphs on disk).

This script takes a **raw single-cell table** (CSV) and a **requirements file**
(a pickled Python dict) and produces a folder of serialized graphs (via
`data_utils.create_graph_and_save`) for downstream training/testing in SHIELD.

It supports two segmentation modes:

1) **Voronoi sampling** (`--segmentation voronoi`)
   - Selects anchor cells ("anker") per tissue/sample.
   - Builds Voronoi regions (optionally with "fuzzy" boundary constraints).
   - For each region ID, constructs a graph based on a neighborhood radius.

2) **Random sampling** (`--segmentation buckets`)
   - Shuffles the sample and splits it into chunks ("buckets").
   - Each chunk becomes one graph (again using a neighborhood radius).

The script is designed for large datasets and uses multiprocessing (spawn mode)
to parallelize graph creation. Failures in worker processes are caught and
reported without crashing the full run.

---------------------------------------------------------------------------
Inputs
---------------------------------------------------------------------------
- requirements_file_path (pickle, dict):
    Must include (at minimum) keys used in this script, e.g.
    - path_raw_data: str/path to CSV
    - path_to_data_set: output base folder
    - measurement_sample_name: column name identifying tissues/ROIs/samples
    - validation_split_column: column name defining fold assignment
    - number_validation_splits: list/iterable of train folds
    - test_set_fold_number: list/iterable of test folds
    - anker_value_all: list of anchor selection values (fraction or absolute)
    - anker_cell_selection_type: "%" or "absolut"
    - fussy_limit_all: list of fuzzy boundary values (voronoi mode)
    - radius_distance_all: list of neighborhood radii
    - minimum_number_cells: int
    - augmentation_number: int
    - multiple_labels_per_subSample: bool
    - label_dict: dict (class labels)
    - label_column: str
    Optionally:
    - filter_cells: list/tuple with at least one column name
    - cell_type_column: str specifying which col to use for filtering
    - downSampeling: float (ratio for downsampling)

- raw CSV:
    Must contain columns referenced by the requirements dict and CLI options.

---------------------------------------------------------------------------
Outputs
---------------------------------------------------------------------------
Graphs are written under:
    <path_to_data_set>/
        anker_value_<...>/
            min_cells_<...>/
                (voronoi) fussy_limit_<...>/radius_<...>/<train|test>/graphs/
                (bucket)  bucket_sampling/radius_<...>/<train|test>/graphs/

Each graph is created by `data_utils.create_graph_and_save(...)`.
This script mainly handles:
- selecting which cells belong to each "region"/chunk
- iterating over meta-parameters
- dispatching parallel workers
- establishing an output directory structure

---------------------------------------------------------------------------
CLI highlights
---------------------------------------------------------------------------
- --data_set_type {train,test}:
    Controls which folds are included based on requirements.
- --max_graphs:
    Hard cap on graphs created per configuration (useful for tests/CI).
- Benchmarking toggles:
    --noisy_labeling, --node_prob, --randomise_edges, --percent_number_cells
  These are forwarded to `create_graph_and_save`.

Notes
-----
- Multiprocessing uses `spawn` to avoid issues with forked state (common on macOS
  and with complex ML stacks). If the start method is already set, it is not
  changed.
- The script tries to be robust: one failed graph does not stop the whole run.
"""

from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .tests import input_test
from .utils import data_utils


# Helper functions for multiprocessing - must be in same script ---------------------
# Multiprocessing variables
_GLOBAL_SINGLE_SAMPLE = None
_GLOBAL_VORONOI = None
_GLOBAL_SAMPLE_COLLECTION = None
_GLOBAL_REQUIREMENTS = None

# Worker initialisation
def init_worker(single_sample, voronoi_list, sample_collection, requirements):
    global _GLOBAL_SINGLE_SAMPLE
    global _GLOBAL_VORONOI
    global _GLOBAL_SAMPLE_COLLECTION
    global _GLOBAL_REQUIREMENTS

    _GLOBAL_SINGLE_SAMPLE = single_sample
    _GLOBAL_VORONOI = voronoi_list
    _GLOBAL_SAMPLE_COLLECTION = sample_collection
    _GLOBAL_REQUIREMENTS = requirements
    
# Single worker create_graph function for voronoi
def worker_create_graph(region_id, *, save_path_folder, radius_distance,
                        sub_sample, repeat_id, skip_existing,
                        noisy_labeling, node_prob, randomise_edges,
                        percent_number_cells, segmentation, testing_mode):
    return data_utils.create_graph_and_save(
        whole_data=_GLOBAL_SINGLE_SAMPLE,
        voronoi_list=_GLOBAL_VORONOI,
        requiremets_dict=_GLOBAL_REQUIREMENTS,
        voronoi_id=region_id,
        save_path_folder=save_path_folder,
        radius_neibourhood=radius_distance,
        sub_sample=sub_sample,
        repeat_id=repeat_id,
        skip_existing=skip_existing,
        noisy_labeling=noisy_labeling,
        node_prob=node_prob,
        randomise_edges=randomise_edges,
        percent_number_cells=percent_number_cells,
        segmentation=segmentation,
        testing_mode=testing_mode,
    )


def main() -> None:
    """
    Entry point for graph generation.

    High-level flow:
    1) Parse CLI args, normalize boolean flags.
    2) Load and validate requirements dict.
    3) Load raw CSV, optionally filter cells, select folds.
    4) Iterate:
       for each tissue/sample:
         for each anchor selection value:
           optional downsampling
           for each augmentation repeat:
             depending on segmentation:
               - Voronoi: compute regions, then graph-per-region
               - Random: build chunks, then graph-per-chunk
             for each radius:
               build output folder
               parallelize graph creation
    """
    # -----------------------------
    # CLI / configuration
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-req_path",
        "--requirements_file_path",
        default=Path.cwd() / "examples" / "CRC" / "requirements.pt",
        help="Path to pickled requirements dict used to drive graph generation.",
    )
    parser.add_argument(
        "-dat_type",
        "--data_set_type",
        default="test",
        choices=["train", "test"],
        help="Whether to generate graphs for train folds or test fold(s).",
    )
    parser.add_argument(
        "-skip",
        "--skip_existing",
        default="True",
        choices=["True", "False"],
        help="Skip files if they already exits",
    )

    # Benchmarking / perturbation knobs (forwarded into create_graph_and_save)
    parser.add_argument(
        "-n_label",
        "--noisy_labeling",
        default=False,
        help="Inject noisy labels (bool-like string supported).",
    )
    parser.add_argument(
        "-node_prob",
        "--node_prob",
        default=False,
        help="Enable node probability perturbation (bool-like string supported).",
    )
    parser.add_argument(
        "-rnd_edges",
        "--randomise_edges",
        default=False,
        help="Randomize edges (bool-like string supported).",
    )
    parser.add_argument(
        "-p_edges",
        "--percent_number_cells",
        type=float,
        default=0.1,
        help="Edge perturbation strength / percent of cells (forwarded).",
    )

    parser.add_argument(
        "-segmentation",
        "--segmentation",
        type=str,
        default="voronoi",
        choices=["bucket", "voronoi"],
        help="How to partition cells into subgraphs.",
    )

    # Optional downsampling
    parser.add_argument(
        "-downSample",
        "--reduce_population",
        default=False,
        help=(
            "If not False, downsample a specific cell type label given here "
            "(e.g. 'T_cell'). Uses requirements['downSampeling']."
        ),
    )
    parser.add_argument(
        "-column_celltype_name",
        "--column_celltype_name",
        default="Class0",
        help="Column name that encodes cell type labels used for downsampling.",
    )

    parser.add_argument(
        "-rev",
        "--reverse_sampling",
        default=False,
        help="If True, iterate samples in reverse order (useful for debugging).",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )

    parser.add_argument(
        "--max_graphs",
        type=int,
        default=None,
        help="If set, cap the total number of graphs created (useful for tests).",
    )
    parser.add_argument(
        "--testing_mode",
        default=False,
        help="if in testing_mode all steps should be done without a previous run thus no graphs etc should already exist.",
    )
    args = parser.parse_args()

    # Convert "truthy"/"falsy" inputs into real booleans for downstream code.
    # (e.g. users might pass "True"/"False" as strings.)
    args.noisy_labeling = data_utils.bool_passer(args.noisy_labeling)
    args.node_prob = data_utils.bool_passer(args.node_prob)
    args.randomise_edges = data_utils.bool_passer(args.randomise_edges)
    args.reduce_population = data_utils.bool_passer(args.reduce_population)
    args.reverse_sampling = data_utils.bool_passer(args.reverse_sampling)
    args.skip_existing = data_utils.bool_passer(args.skip_existing)
    args.testing_mode = data_utils.bool_passer(args.testing_mode)

    if args.verbose:
        print(args)

    # -----------------------------
    # Multiprocessing setup
    # -----------------------------
    # Use 'spawn' (safer across platforms / complex imports).
    # If already set (e.g. by another module), ignore the RuntimeError.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Keep at least 1 worker; subtracting 2 leaves headroom for OS/UI.
    n_workers = max(1, mp.cpu_count() - 2)

    # -----------------------------
    # Load requirements + dataset
    # -----------------------------
    # Requirements are stored as a pickled dictionary.
    with open(args.requirements_file_path, "rb") as f:
        requirements = pickle.load(f)

    # Validate required keys / schema for this pipeline.
    requirements = input_test.validate_all_keys_in_req(
        req_file=requirements, verbose=args.verbose
    )

    # Determine which folds to keep based on the requested dataset type.
    fold_ids = (
        requirements["number_validation_splits"]
        if args.data_set_type == "train"
        else requirements["test_set_fold_number"]
    )

    # Load raw table of cells.
    input_data = pd.read_csv(requirements["path_raw_data"])

    # Optional cell filtering (e.g., restrict to certain ROI/quality flags).
    if requirements["filter_cells"] is not None:
        cell_col = requirements['cell_type_column']
        # Make sure col is in the dataset.          
        for col in requirements["filter_cells"]:
            if col not in np.unique(input_data[cell_col].values):
                print(f"Warning: Column '{col}' specified in 'filter_cells' not found in the dataset. \n\
                      Continuing without filtering this cell type.")
            else:
                input_data = input_data[
                    input_data[requirements['cell_type_column']] != col
                ]
                print(f'Removing cell type {col} from dataset, remaining cells: {input_data.shape[0]}')

    # Select only rows belonging to the chosen folds.
    data_sample = input_data.loc[
        input_data[requirements["validation_split_column"]].isin(fold_ids)
    ].reset_index(drop=True)

    # Identify unique tissue/sample IDs to iterate over.
    sample_ids = data_sample[requirements["measurement_sample_name"]].unique()
    if args.reverse_sampling:
        sample_ids = sample_ids[::-1]

    # -----------------------------
    # Main iteration loops
    # -----------------------------
    for sub_sample in tqdm(sample_ids, desc="Samples"):
        # Subset the dataframe to this tissue/sample.
        single_sample = data_sample[
            data_sample[requirements["measurement_sample_name"]] == sub_sample
        ]

        # Iterate over anchor selection strengths (fraction or absolute).
        for anker_value in requirements["anker_value_all"]:
            # Optional: downsample a given cell type in this sample.
            if args.reduce_population is not False:
                if args.column_celltype_name not in single_sample.columns:
                    raise ValueError(
                        f"Column '{args.column_celltype_name}' not found in the dataset."
                    )
                if "downSampeling" not in requirements:
                    raise ValueError(
                        "Downsampling ratio 'downSampeling' not specified in requirements."
                    )

                downsample_ratio = requirements["downSampeling"]

                # NOTE: reducePopulation is expected to return a new dataframe.
                single_sample = data_utils.reducePopulation(
                    df=single_sample,
                    columnName=args.column_celltype_name,
                    cellTypeName=args.reduce_population,
                    downsampleRatio=downsample_ratio,
                )

            # Repeat with augmentation seeds/variants.
            for augment_id in range(requirements["augmentation_number"]):
                # ---------------------------------------------------------
                # Voronoi segmentation mode
                # ---------------------------------------------------------
                if args.segmentation == "voronoi":
                    for fussy_limit in requirements["fussy_limit_all"]:
                        # Select anchor cells.
                        # - "%" means fraction of cells
                        # - "absolut" means a fixed number
                        if requirements["anker_cell_selection_type"] == "%":
                            anchors = single_sample.sample(
                                n=int(len(single_sample) * anker_value)
                            )

                        elif requirements["anker_cell_selection_type"] == "absolut":
                            if requirements["multiple_labels_per_subSample"]:
                                # If multiple labels exist per tissue, sample anchors per label.
                                samples_per_tissue = anker_value // len(
                                    requirements["label_dict"].keys()
                                )
                                anchors = pd.DataFrame()
                                for label_dict_key in requirements["label_dict"].keys():
                                    anchors = pd.concat(
                                        [
                                            anchors,
                                            single_sample[
                                                single_sample[
                                                    requirements["label_column"]
                                                ]
                                                == label_dict_key
                                            ].sample(n=samples_per_tissue),
                                        ]
                                    )
                            else:
                                anchors = single_sample.sample(n=int(anker_value))

                        else:
                            raise ValueError(
                                "Unknown requirements['anker_cell_selection_type'] "
                                f"={requirements['anker_cell_selection_type']!r}"
                            )

                        # Compute Voronoi regions with fuzzy boundary constraints.
                        voroni_id_fussy = data_utils.get_voronoi_id(
                            data_set=single_sample,
                            anker_cell=anchors,
                            requiremets_dict=requirements,
                            fussy_limit=fussy_limit,
                        )

                        # Basic QC: drop regions that are too small and recompute.
                        number_cells = np.array([len(arr) for arr in voroni_id_fussy])
                        if any(number_cells < requirements["minimum_number_cells"]):
                            voroni_id_fussy = data_utils.get_voronoi_id(
                                data_set=single_sample,
                                anker_cell=anchors[
                                    number_cells > requirements["minimum_number_cells"]
                                ],
                                requiremets_dict=requirements,
                                fussy_limit=fussy_limit,
                            )

                        # Region indices (one graph per region id).
                        voronoi_id = np.arange(0, len(voroni_id_fussy))
                        if args.max_graphs is not None:
                            voronoi_id = voronoi_id[: int(args.max_graphs)]

                        for radius_distance in requirements["radius_distance_all"]:
                            # Construct the output directory layout deterministically
                            # from the meta-parameters.
                            save_path = Path(
                                requirements["path_to_data_set"]
                                / f"anker_value_{anker_value}".replace(".", "_")
                                / f"min_cells_{requirements['minimum_number_cells']}"
                                / f"fussy_limit_{fussy_limit}".replace(".", "_")
                                / f"radius_{radius_distance}"
                            )

                            save_path_folder_graphs = (
                                save_path / f"{args.data_set_type}" / "graphs"
                            )
                            save_path_folder_graphs.mkdir(parents=True, exist_ok=True)

                            if args.verbose:
                                print(
                                    "Voronoi config:",
                                    dict(
                                        anker_value=anker_value,
                                        radius_neibourhood=radius_distance,
                                        fussy_limit=fussy_limit,
                                        sample=sub_sample,
                                        augment_id=augment_id,
                                        n_regions=len(voronoi_id),
                                        out=str(save_path_folder_graphs),
                                    ),
                                )

                            # Bind all fixed parameters once; the pool only gets region ids.
                            # run_saving_routine = functools.partial(
                            #     data_utils.create_graph_and_save,
                            #     whole_data=single_sample,
                            #     save_path_folder=save_path_folder_graphs,
                            #     radius_neibourhood=radius_distance,
                            #     requiremets_dict=requirements,
                            #     voronoi_list=voroni_id_fussy,
                            #     sub_sample=sub_sample,
                            #     repeat_id=augment_id,
                            #     skip_existing=args.skip_existing,
                            #     noisy_labeling=args.noisy_labeling,
                            #     node_prob=args.node_prob,
                            #     randomise_edges=args.randomise_edges,
                            #     percent_number_cells=args.percent_number_cells,
                            #     segmentation=args.segmentation,
                            #     testing_mode=args.testing_mode,
                            # )

                            # Parallel execution over region ids.
                            with mp.Pool(
                                    n_workers,
                                    initializer=init_worker,
                                    initargs=(single_sample, voroni_id_fussy, None, requirements),
                                    maxtasksperchild=1,
                                ) as pool:

                                    for _ in pool.imap_unordered(
                                        functools.partial(
                                            worker_create_graph,
                                            save_path_folder=save_path_folder_graphs,
                                            radius_distance=radius_distance,
                                            sub_sample=sub_sample,
                                            repeat_id=augment_id,
                                            skip_existing=args.skip_existing,
                                            noisy_labeling=args.noisy_labeling,
                                            node_prob=args.node_prob,
                                            randomise_edges=args.randomise_edges,
                                            percent_number_cells=args.percent_number_cells,
                                            segmentation=args.segmentation,
                                            testing_mode=args.testing_mode,
                                        ),
                                        voronoi_id,
                                        chunksize=10,
                                    ):
                                        pass

                # ---------------------------------------------------------
                # Random segmentation mode
                # ---------------------------------------------------------
                elif args.segmentation == "bucket":
                    # In random mode, we create a list of sub-dataframes ("chunks").
                    # Each chunk becomes one graph (selected via chunk index).
                    if requirements["multiple_labels_per_subSample"]:
                        # Split per label to avoid label imbalance in chunks.
                        if args.max_graphs is not None:
                            samples_per_tissue = int(args.max_graphs)
                        else:
                            samples_per_tissue = anker_value // len(
                                requirements["label_dict"].keys()
                            )

                        sample_collection = []
                        for label_dict_key in requirements["label_dict"].keys():
                            shuffled = (
                                single_sample[
                                    single_sample[requirements["label_column"]]
                                    == label_dict_key
                                ]
                                .sample(frac=1)
                                .reset_index(drop=True)
                            )
                            sample_collection.extend(
                                np.array_split(shuffled, samples_per_tissue)
                            )
                    else:
                        # If no per-label sampling is needed, just shuffle and split.
                        if args.max_graphs is not None:
                            n_chunks = int(args.max_graphs)
                        else:
                            # NOTE: original logic uses int(len * anker_value) as chunk count.
                            # This can be small if anker_value is small; ensure at least 1 chunk.
                            n_chunks = max(1, int(len(single_sample) * anker_value))

                        sample_collection = np.array_split(
                            single_sample.sample(frac=1, random_state=42).reset_index(
                                drop=True
                            ),
                            n_chunks,
                        )

                    subsection_id = np.arange(0, len(sample_collection))

                    # No Voronoi regions in bucket mode.
                    voroni_id_fussy = None

                    for radius_distance in requirements["radius_distance_all"]:
                        save_path = Path(
                            requirements["path_to_data_set"]
                            / f"anker_value_{anker_value}".replace(".", "_")
                            / f"min_cells_{requirements['minimum_number_cells']}"
                            / "bucket_sampling"
                            / f"radius_{radius_distance}"
                        )
                        save_path_folder_graphs = (
                            save_path / f"{args.data_set_type}" / "graphs"
                        )
                        save_path_folder_graphs.mkdir(parents=True, exist_ok=True)

                        if args.verbose:
                            print(
                                "bucket config:",
                                dict(
                                    anker_value=anker_value,
                                    radius_neibourhood=radius_distance,
                                    sample=sub_sample,
                                    augment_id=augment_id,
                                    n_chunks=len(subsection_id),
                                    out=str(save_path_folder_graphs),
                                ),
                            )

                        # Bind all fixed parameters once; the pool only gets chunk ids.
                        # Here `whole_data` is a list of dataframes; the worker selects
                        # the chunk based on the provided id.
                        # run_saving_routine = functools.partial(
                        #     data_utils.create_graph_and_save,
                        #     whole_data=sample_collection,
                        #     save_path_folder=save_path_folder_graphs,
                        #     radius_neibourhood=radius_distance,
                        #     requiremets_dict=requirements,
                        #     voronoi_list=voroni_id_fussy,
                        #     sub_sample=sub_sample,
                        #     repeat_id=augment_id,
                        #     skip_existing=args.skip_existing,
                        #     noisy_labeling=args.noisy_labeling,
                        #     node_prob=args.node_prob,
                        #     randomise_edges=args.randomise_edges,
                        #     percent_number_cells=args.percent_number_cells,
                        #     segmentation=args.segmentation,
                        #     testing_mode=args.testing_mode,
                        # )

                        # with mp.Pool(n_workers) as pool:
                        #     for _ in pool.imap_unordered(
                        #         functools.partial(safe_wrapper, run_saving_routine),
                        #         subsection_id,
                        #         chunksize=1,
                        #     ):
                        #         pass
                            
                        with mp.Pool(
                                n_workers,
                                initializer=init_worker,
                                initargs=(sample_collection, None, None, requirements),
                                maxtasksperchild=1,
                            ) as pool:

                                for _ in pool.imap_unordered(
                                    functools.partial(
                                        worker_create_graph,
                                        save_path_folder=save_path_folder_graphs,
                                        radius_distance=radius_distance,
                                        sub_sample=sub_sample,
                                        repeat_id=augment_id,
                                        skip_existing=args.skip_existing,
                                        noisy_labeling=args.noisy_labeling,
                                        node_prob=args.node_prob,
                                        randomise_edges=args.randomise_edges,
                                        percent_number_cells=args.percent_number_cells,
                                        segmentation=args.segmentation,
                                        testing_mode=args.testing_mode,
                                    ),
                                    subsection_id,
                                    chunksize=10,
                                ):
                                    pass


# Ensure the script runs only when executed directly.
if __name__ == "__main__":
    main()
