#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHIELD cell-type–cell-type interaction extraction + plotting script.

This script loads:
- a SHIELD requirements dict (pickle),
- a "best configuration" dict (pickle),
- the trained best model weights (best_model.pt),
- a precomputed graph dataset (train or test split),

and then:
1) Computes (or loads) a cell-type–cell-type interaction dictionary based on the
   model’s attention-derived Interaction Scores.
2) For each tissue/label (from requirements["label_dict"]):
   - builds interaction DataFrames,
   - selects top interactions per cell type (for different limits),
   - saves boxplots and occurrence-vs-Interaction-Score plots.

Key CLI arguments
-----------------
- --cellTypeColumnName:
    Name of the column encoding cell types (used during interaction extraction).
- --requirements_file_path:
    Path to requirements pickle.
- --best_config_dict_path:
    Path to best config pickle (best hyperparameter setup).
- --recalculate_cTc_Scroes:
    If True, recompute the interaction dict even if a cached file exists.
- --data_set_type:
    Either "train" or "test" (controls which graphs + fold IDs are used).
- --verbose:
    If set, prints additional logging.

Outputs
-------
- A cached interaction dictionary (pickle) in:
    requirements["path_to_model"] / f"cT_t_cT_interactions_dict_{data_set_type}.pt"
- Plots under:
    requirements["path_to_interaction_plots"] / <data_set_type> / ...

Created Nov 2024

@author: Vivek
"""

import torch
from torch_geometric.loader import DataListLoader

import argparse
import pickle
from pathlib import Path
from .utils.data_class import graph_dataset
from .model import ShIeLD
from .utils import evaluation_utils, data_utils
from .tests import input_test


# Select device once (GPU if available).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # -----------------------------
    # CLI / configuration
    # -----------------------------
    parser = argparse.ArgumentParser()

    # Column in the underlying cell table that encodes the cell type label/name.
    parser.add_argument("-ct_col", "--cellTypeColumnName", required=True)

    # Input config files (pickled dicts).
    parser.add_argument("-req_path", "--requirements_file_path", required=True)
    parser.add_argument("-config_dict", "--best_config_dict_path", required=True)

    # If True, force recomputation of interaction dict even if cached file exists.
    parser.add_argument("-recalc", "--recalculate_cTc_Scroes", default=False)

    # Choose which split to run on (affects fold IDs + graph root).
    parser.add_argument("-dat_type", "--data_set_type", default="test")

    # Verbose logging.
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )

    args = parser.parse_args()

    # -----------------------------
    # Basic file existence checks
    # -----------------------------
    if not Path(args.requirements_file_path).exists():
        raise FileNotFoundError(
            f"Default configuration path {args.requirements_file_path} was selected.\n"
            f"But file found there. Please provide a valid path."
        )
    if not Path(args.best_config_dict_path).exists():
        raise FileNotFoundError(
            f"Default configuration path {args.best_config_dict_path} was selected.\n"
            f"But file found there. Please provide a valid path."
        )

    # Normalize bool-like input (string -> bool).
    args.recalculate_cTc_Scroes = data_utils.bool_passer(args.recalculate_cTc_Scroes)

    # Print parsed args only if verbose.
    if args.verbose:
        print(args)
        print("device:", device)

    # -----------------------------
    # Load requirements + best config
    # -----------------------------
    with open(args.requirements_file_path, "rb") as file:
        requirements = pickle.load(file)
    with open(args.best_config_dict_path, "rb") as file:
        best_config_dict = pickle.load(file)

    # Validate schema / key presence.
    requirements = input_test.validate_all_keys_in_req(
        req_file=requirements, verbose=args.verbose
    )
    input_test.validate_best_config(config_file=best_config_dict)

    # -----------------------------
    # Resolve path to graphs for best configuration
    # -----------------------------
    # Folder structure differs between Voronoi-like fussy_limit vs random sampling.
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

    if args.verbose:
        print("path_to_graphs:", path_to_graphs)

    # Determine which folds to load depending on train/test selection.
    fold_ids = [
        requirements["number_validation_splits"]
        if args.data_set_type == "train"
        else requirements["test_set_fold_number"]
    ]

    # -----------------------------
    # DataLoader over graphs
    # -----------------------------
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

    # -----------------------------
    # Load model and weights
    # -----------------------------
    model = ShIeLD(
        num_of_feat=int(requirements["input_layer"]),
        layer_1=requirements["layer_1"],
        layer_final=requirements["output_layer"],
        dp=best_config_dict["dropout_rate"],
        self_att=False,
        attr_bool=requirements["attr_bool"],
        norm_type=requirements["comment_norm"],
    ).to(device)

    # Load best model parameters.
    model.load_state_dict(torch.load(requirements["path_to_model"] / "best_model.pt"))
    model.eval()

    # -----------------------------
    # Compute or load cell-type interaction dictionary
    # -----------------------------
    interaction_cache_path = Path(
        requirements["path_to_model"]
        / f"cT_t_cT_interactions_dict_{args.data_set_type}.pt"
    )

    if args.recalculate_cTc_Scroes or (not interaction_cache_path.exists()):
        if args.verbose:
            print(
                "Computing cell-to-cell interaction dict (recalculate="
                f"{args.recalculate_cTc_Scroes}, cache_exists={interaction_cache_path.exists()})"
            )

        cell_to_cell_interaction_dict = (
            evaluation_utils.get_cell_to_cell_interaction_dict(
                requirements_dict=requirements,
                data_loader=data_loader,
                model=model,
                device=device,
                column_celltype_name=args.cellTypeColumnName,
                save_dict_path=interaction_cache_path,
            )
        )
    else:
        if args.verbose:
            print("Loading cached interaction dict from:", interaction_cache_path)

        with open(interaction_cache_path, "rb") as f:
            cell_to_cell_interaction_dict = pickle.load(f)

    # -----------------------------
    # Plotting loop over tissues + interaction limits
    # -----------------------------
    observed_tissues = list(requirements["label_dict"].keys())

    # Plot two regimes: small (top 4) and full (all cell types).
    for number_interactions in [4, len(requirements["cell_type_names"])]:
        print(f"creating the interactions for top {number_interactions}:")
        for observed_tissue in observed_tissues:
            # Extract interaction matrices for this tissue label.
            interaction_dataFrame, mean_interaction_dataFrame, edge_values = (
                evaluation_utils.get_interaction_DataFrame(
                    tissue_id=requirements["label_dict"][observed_tissue],
                    interaction_dict=cell_to_cell_interaction_dict,
                )
            )

            # Identify top interactions per cell type and the corresponding edge-value stats.
            top_connections, top_connections_edge_values = (
                evaluation_utils.get_top_interaction_per_celltype(
                    interaction_limit=number_interactions,
                    all_interaction_mean_df=mean_interaction_dataFrame,
                    all_interaction_df=interaction_dataFrame,
                    edge_values_df=edge_values,
                )
            )

            # -----------------------------
            # Boxplots of top interactions
            # -----------------------------
            save_path_boxplots = Path(
                requirements["path_to_interaction_plots"]
                / f"{args.data_set_type}"
                / "boxplots"
                / f"{observed_tissue}"
                / f"Top_{number_interactions}"
            )

            save_path_boxplots.mkdir(parents=True, exist_ok=True)

            evaluation_utils.plot_cell_cell_interaction_boxplots(
                interaction_limit=number_interactions,
                all_interaction_mean_df=interaction_dataFrame,
                top_connections=top_connections,
                save_path=save_path_boxplots
                / f"{observed_tissue}_topInteractions_{number_interactions}.png",
            )

            # -----------------------------
            # Occurrence vs mean Interaction Score plots
            # -----------------------------
            save_path_interaction = Path(
                requirements["path_to_interaction_plots"]
                / f"{args.data_set_type}"
                / "occurance_vs_IS"
                / f"{observed_tissue}"
                / f"Top_{number_interactions}"
            )
            save_path_interaction.mkdir(parents=True, exist_ok=True)

            evaluation_utils.plot_pct_vs_mean(
                cell_types=mean_interaction_dataFrame.columns,
                top_connections_attentionScore=top_connections,
                top_connections_edge_values=top_connections_edge_values,
                save_dir=save_path_interaction,
            )


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
