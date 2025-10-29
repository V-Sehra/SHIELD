#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""

import torch
from torch_geometric.loader import DataListLoader

import argparse
import pickle
from pathlib import Path
from utils.data_class import graph_dataset
from model import ShIeLD
import utils.evaluation_utils as evaluation_utils
import tests.input_test as input_test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-datSet", "--data_set_name", type=str, default="HCC")

    parser.add_argument("-req_path", "--requirements_file_path", default=None)
    parser.add_argument("-recalc", "--recalculate_cTc_Scroes", default=False)
    parser.add_argument("-config_dict", "--best_config_dict_path", default=None)
    parser.add_argument("-dat_type", "--data_set_type", default="test")
    parser.add_argument(
        "-sig1", "--significance_threshold_1", type=float, default=0.0001
    )
    parser.add_argument(
        "-sig2", "--significance_threshold_2", type=float, default=0.00001
    )
    parser.add_argument(
        "-stars", "--stars", type=bool, default=False, choices=[True, False]
    )
    args = parser.parse_args()

    if args.requirements_file_path is None:
        args.requirements_file_path = (
            Path.cwd() / "examples" / args.data_set_name / "requirements.pt"
        )
        if not Path(args.requirements_file_path).exists():
            raise FileNotFoundError(
                f"Default configuration path {args.requirements_file_path} was selected.\n"
                f"But file found there. Please provide a valid path."
            )
    if args.best_config_dict_path is None:
        args.best_config_dict_path = (
            Path.cwd() / "examples" / args.data_set_name / "best_config.pt"
        )
        if not Path(args.best_config_dict_path).exists():
            raise FileNotFoundError(
                f"Default configuration path {args.best_config_dict_path} was selected.\n"
                f"But file found there. Please provide a valid path."
            )

    print(args)
    with open(args.requirements_file_path, "rb") as file:
        requirements = pickle.load(file)
    with open(args.best_config_dict_path, "rb") as file:
        best_config_dict = pickle.load(file)

    # Check if the requierments and best_config dict are in the correct format
    requirements = input_test.test_all_keys_in_req(req_file=requirements)
    input_test.test_best_config(config_file=best_config_dict)

    if best_config_dict["fussy_limit"] != "random_sampling":
        path_to_graphs = Path(
            requirements["path_to_data_set"]
            / f"anker_value_{best_config_dict['anker_value']}".replace(".", "_")
            / f"min_cells_{requirements['minimum_number_cells']}"
            / f"fussy_limit_{best_config_dict['fussy_limit']}".replace(".", "_")
            / f"radius_{best_config_dict['radius_distance']}"
        )
    else:
        path_to_graphs = Path(
            requirements["path_to_data_set"]
            / f"anker_value_{best_config_dict['anker_value']}".replace(".", "_")
            / f"min_cells_{requirements['minimum_number_cells']}"
            / f"{best_config_dict['fussy_limit']}"
            / f"radius_{best_config_dict['radius_distance']}"
        )

    fold_ids = [
        requirements["number_validation_splits"]
        if args.data_set_type == "train"
        else requirements["test_set_fold_number"]
    ]
    data_loader = DataListLoader(
        graph_dataset(
            root=str(path_to_graphs / args.data_set_type / "graphs"),
            path_to_graphs=path_to_graphs,
            fold_ids=fold_ids[0],
            requirements_dict=requirements,
            graph_file_names=f"{args.data_set_type}_set_file_names.pkl",
        ),
        batch_size=requirements["batch_size"],
        shuffle=True,
        num_workers=8,
        prefetch_factor=50,
    )
    model = ShIeLD(
        num_of_feat=int(requirements["input_layer"]),
        layer_1=requirements["layer_1"],
        layer_final=requirements["output_layer"],
        dp=best_config_dict["droupout_rate"],
        self_att=False,
        attr_bool=requirements["attr_bool"],
        norm_type=requirements["comment_norm"],
    ).to(device)

    model.load_state_dict(torch.load(requirements["path_to_model"] / f"best_model.pt"))
    model.eval()

    if args.recalculate_cTc_Scroes or (
        not Path(
            requirements["path_to_model"]
            / f"cT_t_cT_interactions_dict_{args.data_set_type}.pt"
        ).exists()
    ):
        if args.data_set_name == "HCC":
            column_celltype_name = "Class0"
        else:
            column_celltype_name = "CellType"

        cell_to_cell_interaction_dict = (
            evaluation_utils.get_cell_to_cell_interaction_dict(
                requirements_dict=requirements,
                data_loader=data_loader,
                model=model,
                device=device,
                column_celltype_name=column_celltype_name,
                save_dict_path=Path(
                    requirements["path_to_model"]
                    / f"cT_t_cT_interactions_dict_{args.data_set_type}.pt"
                ),
            )
        )
    else:
        with open(
            Path(
                requirements["path_to_model"]
                / f"cT_t_cT_interactions_dict_{args.data_set_type}.pt"
            ),
            "rb",
        ) as f:
            cell_to_cell_interaction_dict = pickle.load(f)

    observed_tissues = list(requirements["label_dict"].keys())
    Path(requirements["path_to_interaction_plots"] / "boxplots").mkdir(
        parents=True, exist_ok=True
    )

    for number_interactions in [4, len(requirements["cell_type_names"])]:
        print(f"creating the interactions for top {number_interactions}:")
        for observed_tissue in observed_tissues:
            save_path_interaction = Path(
                requirements["path_to_interaction_plots"]
                / "occurance_vs_IS"
                / f"{observed_tissue}"
                / f"Top_{number_interactions}"
            )
            save_path_interaction.mkdir(parents=True, exist_ok=True)

            save_path_boxplots = Path(
                requirements["path_to_interaction_plots"]
                / "boxplots"
                / f"{observed_tissue}"
                / f"Top_{number_interactions}"
            )
            save_path_boxplots.mkdir(parents=True, exist_ok=True)

            interaction_dataFrame, mean_interaction_dataFrame, edge_values = (
                evaluation_utils.get_interaction_DataFrame(
                    tissue_id=requirements["label_dict"][observed_tissue],
                    interaction_dict=cell_to_cell_interaction_dict,
                )
            )

            top_connections, top_connections_edge_values = (
                evaluation_utils.get_top_interaction_per_celltype(
                    interaction_limit=number_interactions,
                    all_interaction_mean_df=mean_interaction_dataFrame,
                    all_interaction_df=interaction_dataFrame,
                    edge_values_df=edge_values,
                )
            )

            evaluation_utils.plot_cell_cell_interaction_boxplots(
                interaction_limit=number_interactions,
                all_interaction_mean_df=interaction_dataFrame,
                top_connections=top_connections,
                save_path=save_path_boxplots
                / f"{observed_tissue}_topInteractions_{number_interactions}.png",
            )

            evaluation_utils.plot_pct_vs_mean(
                cell_types=mean_interaction_dataFrame.columns,
                top_connections_attentionScore=top_connections,  # your "values"
                top_connections_edge_values=top_connections_edge_values,  # your "values_edge"
                save_dir=save_path_interaction,
            )


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
