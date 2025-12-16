#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHIELD evaluation + optional retraining script (hypersearch aggregation, plots, best-model training).

This script has two main responsibilities:

1) Hyperparameter search evaluation (always runs)
-----------------------------------------------
- Loads the SHIELD training requirements dict (pickle).
- Collects and aggregates hyperparameter search results via
  `evaluation_utils.get_hypersearch_results(...)`.
- Sorts configurations by `total_acc_balanced_mean` (descending).
- Writes a consolidated CSV: `hyper_search_results.csv`.
- Produces per-hyperparameter influence plots for each observed variable in
  `requirements["col_of_variables"]` via
  `evaluation_utils.create_parameter_influence_plots(...)`.

2) Retrain the best configuration and save the best-performing model (optional)
------------------------------------------------------------------------------
If `--retain_best_model_config_bool` evaluates to True (bool-like parsing),
the script:
- Loads the best config dict from `--best_config_dict_path` if present, else
  extracts it from `hyper_search_results` via
  `evaluation_utils.get_best_config_dict(...)` and stores it to disk.
- Constructs the path to the corresponding graph folder (Voronoi vs random)
  using best_config_dict fields (anker_value, fussy_limit, radius_distance).
- Loads the full training set (all validation splits) and the test set
  (requirements["test_set_fold_number"]) using `graph_dataset` and `DataListLoader`.
- Trains `args.number_of_training_repeats` independent model instances using
  the best hyperparameters.
- Tracks the model with the best test F1 score, saves:
    * `best_model.pt` for the best run
    * `model_<num>.pt` for each run
  and updates `best_config_dict["version"]` with the best run id.

Inputs
------
- requirements_file_path (pickle):
    Requirements dict with training paths, label dict, batch size, and plotting vars.
- hypersearch results:
    Expected to be discoverable by evaluation_utils.get_hypersearch_results.
- graphs on disk:
    Precomputed graphs under requirements["path_to_data_set"] following the same
    folder conventions as the graph generation pipeline.
- best_config_dict_path (pickle):
    Optional persisted best configuration dict.

Outputs
-------
- requirements["path_training_results"]/hyper_search_results.csv
- requirements["path_training_results"]/hyper_search_plots/*.png
- requirements["path_to_model"]/best_model.pt
- requirements["path_to_model"]/model_<repeat_id>.pt
- (optionally updated) best_config.pt

Execution details / stability
-----------------------------
- Multiprocessing settings are configured inside main() (not at import time):
  torch multiprocessing start method "fork" and sharing strategy "file_system".
  If already set, the RuntimeError is ignored.
- Device selection uses CUDA if available.

Created Nov 2024

@author: Vivek
"""

import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.loader import DataListLoader

from .model import ShIeLD
from .utils import evaluation_utils
from .utils import data_utils
from .utils import train_utils
from .utils import model_utils
from .utils.data_class import graph_dataset

from .tests import input_test


def main():
    # -----------------------------
    # CLI / configuration
    # -----------------------------
    parser = argparse.ArgumentParser()

    # Requirements file: pickled dict driving paths + hyperparameters + plotting vars.
    parser.add_argument(
        "-req_path",
        "--requirements_file_path",
        default=Path.cwd() / "examples" / "CRC" / "requirements.pt",
    )

    # Whether to (re)train the best configuration and save a best model.
    parser.add_argument("-retrain", "--retain_best_model_config_bool", default=True)

    # Where to load/save the best configuration dict (pickle).
    parser.add_argument(
        "-config_dict",
        "--best_config_dict_path",
        default=Path.cwd() / "examples" / "CRC" / "best_config.pt",
    )

    # Number of retraining repeats for best config (select best by test F1).
    parser.add_argument("-rep", "--number_of_training_repeats", default=5)

    # Optional max epochs override forwarded to training loop.
    parser.add_argument("-maxEpoch", "--maxEpoch", default=None)

    # Verbose console logging.
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )

    args = parser.parse_args()

    # Normalize user-provided paths.
    args.number_of_training_repeats = int(args.number_of_training_repeats)
    args.best_config_dict_path = Path(args.best_config_dict_path)
    args.requirements_file_path = Path(args.requirements_file_path)

    # Parse bool-like retrain flag (string -> bool).
    # NOTE: code uses args.retain_best_model_config_bool in the parser,
    # but passes args.retain_best_model_config_bool into bool_passer.
    args.retrain_best_model_config_bool = data_utils.bool_passer(
        args.retain_best_model_config_bool
    )

    if args.verbose:
        print(args, flush=True)

    # -----------------------------
    # Torch multiprocessing setup
    # -----------------------------
    # Do this at runtime (not module import time), so tests can monkeypatch.
    try:
        torch.multiprocessing.set_start_method("fork", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        # already set
        pass

    # -----------------------------
    # Device selection
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(device)

    # -----------------------------
    # Load and validate requirements
    # -----------------------------
    with open(args.requirements_file_path, "rb") as file:
        requirements = pickle.load(file)

    requirements = input_test.validate_all_keys_in_req(
        req_file=requirements, verbose=args.verbose
    )

    # Ensure model directory exists.
    requirements["path_to_model"].mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Hyperparameter search evaluation
    # -----------------------------
    if args.verbose:
        print("evaluating the training results")

    hyper_search_results = evaluation_utils.get_hypersearch_results(
        requirements_dict=requirements, verbose=args.verbose
    )

    # Sort best-to-worst by mean balanced accuracy.
    hyper_search_results = hyper_search_results.sort_values(
        "total_acc_balanced_mean", ascending=False
    )

    # Persist full hypersearch table.
    hyper_search_results.to_csv(
        Path(requirements["path_training_results"] / "hyper_search_results.csv"),
        index=False,
    )

    # Reshape into long form to facilitate per-hyperparameter influence plots.
    melted_results = hyper_search_results.melt(
        id_vars=["total_acc_balanced_mean"], var_name="hyperparameter"
    )

    # Output folder for influence plots.
    save_path_folder = Path(
        requirements["path_training_results"] / "hyper_search_plots"
    )
    save_path_folder.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("creating hyperparameter search plots")

    # Create one plot per "observable" configured in requirements.
    for observable_of_interest in requirements["col_of_variables"]:
        evaluation_utils.create_parameter_influence_plots(
            df=melted_results,
            observed_variable=observable_of_interest,
            save_path=save_path_folder / f"{observable_of_interest}.png",
        )

    # -----------------------------
    # Optional: retrain best config and save best model
    # -----------------------------
    if data_utils.bool_passer(args.retain_best_model_config_bool):
        # Load existing best_config.pt if present, otherwise infer best config and save it.
        if args.best_config_dict_path.exists():
            with open(args.best_config_dict_path, "rb") as file:
                best_config_dict = pickle.load(file)
        else:
            best_config_dict = evaluation_utils.get_best_config_dict(
                hyper_search_results=hyper_search_results,
                requirements_dict=requirements,
            )

            with open(args.best_config_dict_path, "wb") as file:
                pickle.dump(best_config_dict, file)

        if args.verbose:
            print("best configuration:")
            print(best_config_dict)

        # Determine graph folder path for this configuration.
        # Voronoi-style folder uses fussy_limit_<...>, random uses the fussy_limit string directly.
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

        # -----------------------------
        # Data loaders (train = all folds, test = test fold(s))
        # -----------------------------
        data_loader_train = DataListLoader(
            graph_dataset(
                root=str(path_to_graphs / "train" / "graphs"),
                path_to_graphs=path_to_graphs,
                fold_ids=requirements["number_validation_splits"],
                requirements_dict=requirements,
                graph_file_names="train_set_file_names.pkl",
                verbose=args.verbose,
            ),
            batch_size=requirements["batch_size"],
            shuffle=True,
            num_workers=8,
            prefetch_factor=50,
        )

        data_loader_test = DataListLoader(
            graph_dataset(
                root=str(path_to_graphs / "test" / "graphs"),
                path_to_graphs=path_to_graphs,
                fold_ids=requirements["test_set_fold_number"],
                requirements_dict=requirements,
                graph_file_names="test_set_file_names.pkl",
                verbose=args.verbose,
            ),
            batch_size=requirements["batch_size"],
            shuffle=True,
            num_workers=8,
            prefetch_factor=50,
        )

        # -----------------------------
        # Retraining repeats: select best by test F1 score
        # -----------------------------
        # Track which of the repeated trainings achieves the best predictive power.
        best_model_f1score = 0
        for num in tqdm(range(args.number_of_training_repeats)):
            # Initialize loss (typically class-weighted based on label_dict).
            loss_fkt = train_utils.initialize_loss(
                path=Path(path_to_graphs / "train_set_file_names.pkl"),
                tissue_dict=requirements["label_dict"],
                device=device,
            )

            # Construct model with best hyperparameters.
            model = ShIeLD(
                num_of_feat=int(best_config_dict["input_layer"]),
                layer_1=best_config_dict["layer_1"],
                layer_final=best_config_dict["output_layer"],
                dp=best_config_dict["droupout_rate"],
                self_att=False,
                attr_bool=best_config_dict["attr_bool"],
                norm_type=best_config_dict["comment_norm"],
            ).to(device)

            model.train()

            # Train with early stopping patience (from config if available).
            model, train_loss = train_utils.train_loop_shield(
                optimizer=torch.optim.Adam(
                    model.parameters(), lr=requirements["learning_rate"]
                ),
                model=model,
                data_loader=data_loader_train,
                loss_fkt=loss_fkt,
                attr_bool=best_config_dict["attr_bool"],
                device=device,
                patience=best_config_dict["patience"]
                if "patience" in best_config_dict.keys()
                else 9,
                max_epochs=args.maxEpoch,
            )

            model.eval()
            if args.verbose:
                print("start validation")

            # Evaluate on training loader.
            train_bal_acc, train_f1_score, train_cm = model_utils.get_acc_metrics(
                model=model, data_loader=data_loader_train, device=device
            )

            # Evaluate on test loader (named val_* here, but it's the test set in this script).
            val_bal_acc, val_f1_score, val_cm = model_utils.get_acc_metrics(
                model=model, data_loader=data_loader_test, device=device
            )

            # Update best model if test F1 improved.
            if val_f1_score > best_model_f1score:
                best_model_f1score = val_f1_score
                if args.verbose:
                    print(
                        f"best model so far: VersioNo.{num} with test f1 score of {val_f1_score}, balanced accuracy of {val_bal_acc}"
                    )
                    print(f"train scores: bal= {train_bal_acc} f1= {train_f1_score}")

                # Save the best model weights.
                model_save_path = Path(requirements["path_to_model"] / "best_model.pt")
                torch.save(model.state_dict(), model_save_path)

                # Persist which repeat produced the best model.
                best_config_dict["version"] = num

                with open(args.best_config_dict_path, "wb") as file:
                    pickle.dump(best_config_dict, file)

            # Save every model repeat for later inspection / ensembling.
            model_save_path = requirements["path_to_model"] / f"model_{num}.pt"
            torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    main()
