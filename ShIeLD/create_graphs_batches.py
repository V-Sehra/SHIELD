#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import multiprocessing as mp
import functools
from pathlib import Path
import pickle

from utils import data_utils
from tests import input_test


# For multiprocessing error handling
def safe_wrapper(func, arg):
    try:
        return func(arg)
    except Exception as e:
        import traceback
        print("ERROR in worker for arg:", arg)
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-req_path", "--requirements_file_path",
        default=Path.cwd() / "examples" / "CRC" / "requirements.pt"
    )
    parser.add_argument("-dat_type", "--data_set_type", default="train")
    parser.add_argument("-n_label", "--noisy_labeling", default=False)
    parser.add_argument("-node_prob", "--node_prob", default=False)
    parser.add_argument("-rnd_edges", "--randomise_edges", default=False)
    parser.add_argument("-p_edges", "--percent_number_cells", type=float, default=0.1)
    parser.add_argument(
        "-segmentation", "--segmentation",
        type=str, default="voronoi", choices=["random", "voronoi"]
    )
    parser.add_argument("-downSample", "--reduce_population", default=False)
    parser.add_argument(
        "-column_celltype_name", "--column_celltype_name", default="Class0"
    )
    parser.add_argument("-rev", "--reverse_sampling", default=False)
    parser.add_argument(
        "-batch_size", "--batch_size",
        type=int, default=50000
    )

    args = parser.parse_args()

    args.noisy_labeling = data_utils.bool_passer(args.noisy_labeling)
    args.node_prob = data_utils.bool_passer(args.node_prob)
    args.randomise_edges = data_utils.bool_passer(args.randomise_edges)
    args.reduce_population = data_utils.bool_passer(args.reduce_population)
    args.reverse_sampling = data_utils.bool_passer(args.reverse_sampling)

    requirements = pickle.load(open(args.requirements_file_path, "rb"))
    input_test.test_all_keys_in_req(req_file=requirements)

    fold_ids = [
        requirements["number_validation_splits"]
        if args.data_set_type == "train"
        else requirements["test_set_fold_number"]
    ][0]

    csv_path = requirements["path_raw_data"]
    batch_size = args.batch_size

    for batch_idx, chunk in enumerate(
        pd.read_csv(csv_path, chunksize=batch_size)
    ):
        if requirements["filter_cells"] != False:
            chunk = chunk[
                chunk[requirements["filter_column"][0]] == requirements["filter_value"]
            ]

        chunk = chunk[
            chunk[requirements["validation_split_column"]].isin(fold_ids)
        ]
        if len(chunk) == 0:
            continue

        chunk.reset_index(drop=True, inplace=True)

        samples = chunk[requirements["measument_sample_name"]].unique()
        if args.reverse_sampling:
            samples = samples[::-1]

        for sub_sample in tqdm(samples, desc=f"Batch {batch_idx}"):
            single_sample = chunk[
                chunk[requirements["measument_sample_name"]] == sub_sample
            ]

            if args.reduce_population:
                if args.column_celltype_name not in single_sample.columns:
                    raise ValueError(f"Column '{args.column_celltype_name}' not found.")
                downsampleRatio = requirements["downSampeling"]
                single_sample = data_utils.reducePopulation(
                    df=single_sample,
                    columnName=args.column_celltype_name,
                    cellTypeName=args.reduce_population,
                    downsampleRatio=downsampleRatio,
                )

            for anker_value in requirements["anker_value_all"]:
                for augment_id in range(requirements["augmentation_number"]):

                    if args.segmentation == "voronoi":
                        for fussy_limit in requirements["fussy_limit_all"]:
                            if requirements["anker_cell_selction_type"] == "%":
                                anchors = single_sample.sample(
                                    n=int(len(single_sample) * anker_value)
                                )
                            else:
                                if requirements["multiple_labels_per_subSample"]:
                                    samples_per_tissue = anker_value // len(
                                        requirements["label_dict"].keys()
                                    )
                                    anchors = pd.DataFrame()
                                    for lbl in requirements["label_dict"].keys():
                                        anchors = pd.concat([
                                            anchors,
                                            single_sample[
                                                single_sample[
                                                    requirements["label_column"]
                                                ] == lbl
                                            ].sample(n=samples_per_tissue)
                                        ])
                                else:
                                    anchors = single_sample.sample(n=int(anker_value))

                            voroni_id_fussy = data_utils.get_voronoi_id(
                                data_set=single_sample,
                                anker_cell=anchors,
                                requiremets_dict=requirements,
                                fussy_limit=fussy_limit,
                            )

                            number_cells = np.array([len(arr) for arr in voroni_id_fussy])
                            if any(number_cells < requirements["minimum_number_cells"]):
                                voroni_id_fussy = data_utils.get_voronoi_id(
                                    data_set=single_sample,
                                    anker_cell=anchors[
                                        number_cells
                                        > requirements["minimum_number_cells"]
                                    ],
                                    requiremets_dict=requirements,
                                    fussy_limit=fussy_limit,
                                )

                            vornoi_id = np.arange(0, len(voroni_id_fussy))

                            for radius_distance in requirements["radius_distance_all"]:
                                save_path = (
                                    Path(requirements["path_to_data_set"])
                                    / f"anker_value_{anker_value}".replace(".", "_")
                                    / f"min_cells_{requirements['minimum_number_cells']}"
                                    / f"fussy_limit_{fussy_limit}".replace(".", "_")
                                    / f"radius_{radius_distance}"
                                )

                                folder = (
                                    save_path / f"{args.data_set_type}" / "graphs"
                                )
                                folder.mkdir(parents=True, exist_ok=True)

                                run_fn = functools.partial(
                                    data_utils.create_graph_and_save,
                                    whole_data=single_sample,
                                    save_path_folder=folder,
                                    radius_neibourhood=radius_distance,
                                    requiremets_dict=requirements,
                                    voronoi_list=voroni_id_fussy,
                                    sub_sample=sub_sample,
                                    repeat_id=augment_id,
                                    skip_existing=True,
                                    noisy_labeling=args.noisy_labeling,
                                    node_prob=args.node_prob,
                                    randomise_edges=args.randomise_edges,
                                    percent_number_cells=args.percent_number_cells,
                                    segmentation=args.segmentation,
                                )

                                with mp.Pool(mp.cpu_count() - 2, maxtasksperchild=1) as pool:
                                    for _ in pool.imap_unordered(
                                        functools.partial(safe_wrapper, run_fn),
                                        vornoi_id,
                                        chunksize=1,
                                    ):
                                        pass

                    else:  # random segmentation
                        if requirements["multiple_labels_per_subSample"]:
                            samples_per_tissue = anker_value // len(
                                requirements["label_dict"].keys()
                            )
                            sample_collection = []
                            for lbl in requirements["label_dict"].keys():
                                sample_collection.extend(
                                    np.array_split(
                                        single_sample[
                                            single_sample[
                                                requirements["label_column"]
                                            ] == lbl
                                        ].sample(frac=1),
                                        samples_per_tissue,
                                    )
                                )
                        else:
                            n_chunks = int(len(single_sample) * anker_value)
                            sample_collection = np.array_split(
                                single_sample.sample(frac=1, random_state=42),
                                n_chunks,
                            )

                        subsection_id = np.arange(0, len(sample_collection))
                        voroni_id_fussy = None

                        for radius_distance in requirements["radius_distance_all"]:
                            save_path = (
                                Path(requirements["path_to_data_set"])
                                / f"anker_value_{anker_value}".replace(".", "_")
                                / f"min_cells_{requirements['minimum_number_cells']}"
                                / "random_sampling"
                                / f"radius_{radius_distance}"
                            )

                            folder = (
                                save_path / f"{args.data_set_type}" / "graphs"
                            )
                            folder.mkdir(parents=True, exist_ok=True)

                            run_fn = functools.partial(
                                data_utils.create_graph_and_save,
                                whole_data=sample_collection,
                                save_path_folder=folder,
                                radius_neibourhood=radius_distance,
                                requiremets_dict=requirements,
                                voronoi_list=voroni_id_fussy,
                                sub_sample=sub_sample,
                                repeat_id=augment_id,
                                skip_existing=False,
                                noisy_labeling=args.noisy_labeling,
                                node_prob=args.node_prob,
                                randomise_edges=args.randomise_edges,
                                percent_number_cells=args.percent_number_cells,
                                segmentation=args.segmentation,
                            )

                            with mp.Pool(mp.cpu_count() - 2, maxtasksperchild=1) as pool:
                                for _ in pool.imap_unordered(
                                    functools.partial(safe_wrapper, run_fn),
                                    subsection_id,
                                    chunksize=1,
                                ):
                                    pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()