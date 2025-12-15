import pickle
from pathlib import Path
import numpy as np

"""
SHIELD test suite — configuration contract and validation rules
================================================================

This test module validates the *configuration objects* used by SHIELD
(Spatially-enHanced ImmunE Landscape Decoding), a weakly-supervised, 
graph-attention–based pipeline for interpretable cell–cell interaction
modeling. The tests ensure that experiment configs are complete, well-typed,
and self-consistent **before** any training/evaluation runs.

What these tests check
----------------------
1) ``test_all_keys_in_req``:
   - Loads a *requirements* object (dict or pickle file) and verifies that all
     **mandatory** fields exist and have the correct types.
   - Fills in any **optional** fields with sensible defaults (see
     ``default_dict``) and re-validates types/values.
   - Enforces consistent types for paths, integers, strings, lists of strings,
     lists of numbers, and booleans used by the SHIELD data/loader/training
     pipeline.

2) ``test_best_config``:
   - Loads a *best configuration* object (dict or pickle file) produced by a
     hyperparameter search and verifies that all **required** model/feature
     selections are present and correctly typed.

Mandatory keys in the *requirements* object
-------------------------------------------
The following keys **must** be present (see ``must_have_keys``). Types and
semantics are enforced by ``test_all_keys_in_req``:

- ``path_raw_data`` (str | pathlib.Path)
    Filesystem location of the raw per-cell/per-sample data used to build graphs.

All of the following keys need to be the column names within the raw data file:

- ``cell_type_names`` (list[str])
    Canonical cell type labels present in the dataset (e.g., from annotation).

- ``markers`` (list[str])
    Marker/features used to build node attributes (e.g., protein channels).

- ``label_column`` (str)
    Column name containing the target label(s) used for supervision.

- ``label_dict`` (dict-like)
    Mapping from raw labels to canonical/encoded labels used by the model.
    i.e. {cancer: 1, normal: 0}

- ``eval_columns`` (list[str])
    Names of evaluation metrics/columns expected downstream (patient ID, Cell type etc.).
    must contain at least one of the following: CellType

- ``validation_split_column`` (str)
    Column that denotes the split identifier (e.g., CV fold or patient group).

- ``number_validation_splits`` (list[int|float])
    Provides a List of the fold identifiers used for validation splits.
    Therefore, if 5-fold cross-validation is used, this should be [1,2,3,4], with [5] for testing

- ``test_set_fold_number`` (list[int|float])
    Identifier(s) for the held-out test fold(s).
    Therefore, if 5-fold cross-validation is used, this should be [5].

- ``measument_sample_name`` (str)
    Name of the sample identifier column. **Note:** spelled exactly as
    ``measument_sample_name``.
    Sample can be a patient ID, biopsy ID, or other grouping variable.

- ``X_col_name`` (str)
    Column name for the x-coordinate (used for spatial graph construction).

- ``Y_col_name`` (str)
    Column name for the y-coordinate.

The Following key defines model architecture:

- ``sampleing`` (list[float|int] or config-like)
    Sampling definition for sub-samples/tiles (kept as provided; validated as list
    of numbers if applicable). **Note:** the key name is spelled *exactly* as
    ``sampleing``.
- ``output_layer`` (int)
    Size of the model’s output layer (e.g., number of classes).

Optional keys and default behavior
----------------------------------
If any of the following are omitted, the test injects defaults from
``default_dict`` and logs a message:

- Paths:
  - ``path_to_data_set`` (Path): dataset root (default: ``cwd/data_sets/running_example``)
  - ``path_training_results`` (Path): where training outputs are written
  - ``path_to_model`` (Path): where the trained model is saved
  - ``path_to_interaction_plots`` (Path): where SHIELD interaction plots are saved

- Experiment bookkeeping:
  - ``col_of_interest`` (list[str]): columns kept in result tracking
  - ``col_of_variables`` (list[str]): tunable variables recorded per run

- Data/graph building:
  - ``minimum_number_cells`` (int): minimum cells per (sub)sample (graph)
  - ``radius_distance_all`` (list[number]): candidate neighborhood radii
  - ``anker_value_all`` (list[number]): candidate anchor distances
  - ``anker_cell_selction_type`` (str): either ``"%"`` or ``"absolut"`` (exact spelling)
  - ``multiple_labels_per_subSample`` (bool): allow multi-label subsamples
  - ``voro_neighbours`` (int): Voronoi-based neighbor cap (if used)

- Training:
  - ``batch_size`` (int)
  - ``learning_rate`` (float)
  - ``layer_1`` (int): hidden layer width for the SHIELD baseline/ablation models
  - ``comment_norm`` (str)
  - ``databased_norm`` (str|None)
  - ``droupout_rate`` (list[number] or float): dropout range/value (**key name intentionally ``droupout_rate``**)
  - ``augmentation_number`` (int): number of augmentations
  - ``attr_bool`` (bool): whether to compute attribution/interaction maps

Type rules enforced by the tests
--------------------------------
- **Paths**: any key whose name starts with ``"path"`` must be ``str`` or
  ``pathlib.Path``.
- **Integers**: ``minimum_number_cells``, ``batch_size``, ``input_layer``,
  ``layer_1``, ``output_layer``, ``augmentation_number`` must be integer types.
- **Strings**: ``label_column``, ``anker_cell_selction_type``,
  ``validation_split_column``, ``X_col_name``, ``Y_col_name``,
  ``measument_sample_name`` must be strings.
- **List[str]**: ``cell_type_names``, ``markers``, ``eval_columns``,
  ``col_of_interest``, ``col_of_variables``.
- **List[number]**: ``radius_distance_all``, ``anker_value_all``,
  ``droupout_rate``, ``number_validation_splits``, ``test_set_fold_number``.
- **Booleans**: ``filter_cells``, ``multiple_labels_per_subSample``, ``attr_bool``.
- **Conditional**: if ``filter_cells`` is ``True``, both
  ``filter_column`` (list) and ``filter_value`` (int|float|bool) must be present.

Required keys in the *best_config* object
-----------------------------------------
``test_best_config`` enforces the following fields (and types) for a selected
hyperparameter setting:

- **Integers**: ``layer_1``, ``input_layer``, ``output_layer``, ``version``
- **Floats**: ``droupout_rate`` (note the intentional spelling)
- **Booleans**: ``attr_bool``
- **Numbers (float or int)**: ``anker_value``, ``radius_distance``

These correspond to the minimal model specification needed to re-instantiate the
trained SHIELD variant and reproduce the reported performance/interaction maps.

Notes & pitfalls
----------------
- Several key names preserve historical spellings (e.g., ``droupout_rate``,
  ``sampleing``, ``measument_sample_name``). The tests expect these **exact**
  strings to avoid breaking existing experiment artifacts.
- Paths can be provided as strings or ``Path`` objects; they are not resolved
  here, only type-checked.
- Lists that represent folds/splits (``number_validation_splits``,
  ``test_set_fold_number``) are validated as lists of numbers to allow explicit
  identifiers, not just counts.

Minimal examples
----------------
**Requirements (dict)**:
    requirements = {
        "path_raw_data": "/data/IMC/raw.parquet",
        "cell_type_names": ["T cells", "B cells", "Macrophages"],
        "markers": ["CD3", "CD8", "CD68"],
        "sampleing": [0.5],  # kept as provided; validated if numeric
        "label_column": "response_label",
        "label_dict": {"R": 1, "NR": 0},
        "eval_columns": ["f1", "bacc", "auc"],
        "validation_split_column": "cv_fold",
        "number_validation_splits": [0,1,2,3,4],
        "test_set_fold_number": [4],
        "measument_sample_name": "sample_id",
        "X_col_name": "x",
        "Y_col_name": "y",
        "output_layer": 2,
        # optional keys may be omitted and will be filled from default_dict
    }

**Best config (dict)**:
    best_config = {
        "layer_1": 32,
        "input_layer": 24,
        "droupout_rate": 0.3,
        "output_layer": 2,
        "attr_bool": False,
        "anker_value": 200,
        "radius_distance": 250,
        "version": 1,
    }

Place this docstring at the top of the test module so contributors know exactly
what the configuration contract is and why the assertions exist.
"""
must_have_keys = set(
    [
        "path_raw_data",
        "cell_type_names",
        "markers",
        "sampleing",
        "label_column",
        "label_dict",
        "eval_columns",
        "validation_split_column",
        "number_validation_splits",
        "test_set_fold_number",
        "measument_sample_name",
        "X_col_name",
        "Y_col_name",
        "output_layer",
    ]
)

default_dict = {
    "path_to_data_set": Path.cwd() / "data_sets" / "running_example",
    "path_training_results": Path.cwd() / "running_example" / "training_results",
    "path_to_model": Path.cwd() / "running_example" / "model",
    "path_to_interaction_plots": Path.cwd() / "running_example" / "cTc_interactions",
    "col_of_interest": [
        "anker_value",
        "radius_distance",
        "droupout_rate",
        "comment",
        "comment_norm",
        "model_no",
        "split_number",
    ],
    "col_of_variables": ["droupout_rate", "anker_value", "radius_distance"],
    "minimum_number_cells": 25,
    "radius_distance_all": [250, 530],
    "anker_value_all": [200, 500],
    "anker_cell_selction_type": "absolut",  # either % or absolut
    "multiple_labels_per_subSample": True,
    "batch_size": 150,
    "learning_rate": 1e-2,
    "layer_1": 23,
    "comment_norm": "no_norm",
    "databased_norm": None,
    "droupout_rate": [0.2, 0.8],
    "attr_bool": False,
    "augmentation_number": 5,
    "voro_neighbours": 50,
}


def validate_all_keys_in_req(req_file):
    """
    Test to check if the requirements file contains all required keys
    and if the values are in the correct format.

    Args:
        req_file (str): Path to the requirements file or requirements dict.
    """
    if isinstance(req_file, dict):
        requirements = req_file
    elif isinstance(req_file, (str, Path)):
        with open(req_file, "rb") as file:
            requirements = pickle.load(file)
    else:
        raise TypeError("req_file should be a string, Path object or dict.")

    key_set = set(requirements.keys())

    # Check if all required keys are present
    key_set = set(requirements.keys())
    missing_Mandatory_keys = must_have_keys - key_set
    if len(missing_Mandatory_keys) > 0:
        raise AssertionError(f"Missing keys in requirements: {missing_Mandatory_keys}")

    optional_keys = set(default_dict.keys())
    other_keys = optional_keys - key_set
    for opt_key in other_keys:
        requirements[opt_key] = default_dict[opt_key]
        print(
            f"{opt_key} not set in requirements, setting to default: {default_dict[opt_key]}"
        )

    # Check for correct format of the keys:
    # Paths:
    all_paths = {item for item in key_set if item.lower().startswith("path")}
    for path in all_paths:
        if not isinstance(requirements[path], (str, Path)):
            raise AssertionError(f"Path {path} is not a string or a Path object.")

    # int:
    all_ints = [
        "minimum_number_cells",
        "batch_size",
        "input_layer",
        "layer_1",
        "output_layer",
        "augmentation_number",
    ]
    for item in all_ints:
        if not isinstance(requirements[item], (int, np.integer)):
            raise AssertionError(f"Item {item} is not an integer.")

    # string:
    # label_column
    all_strings = [
        "label_column",
        "anker_cell_selction_type",
        "validation_split_column",
        "X_col_name",
        "Y_col_name",
        "measument_sample_name",
    ]
    for item in all_strings:
        if not isinstance(requirements[item], str):
            raise AssertionError(f"Item {item} is not a string.")

    # list of strings:
    all_lists_str = [
        "cell_type_names",
        "markers",
        "eval_columns",
        "col_of_interest",
        "col_of_variables",
    ]
    for list_str in all_lists_str:
        if not isinstance(requirements[list_str], list) or not all(
            isinstance(i, str) for i in requirements[list_str]
        ):
            raise AssertionError(f"Item {list_str} is not a list of strings.")

    # list of floats or ints:
    all_lists_numbers = [
        "radius_distance_all",
        "anker_value_all",
        "droupout_rate",
        "number_validation_splits",
        "test_set_fold_number",
    ]

    for list_float in all_lists_numbers:
        if not isinstance(requirements[list_float], list) or not all(
            isinstance(i, (float, int, np.integer, np.floating))
            for i in requirements[list_float]
        ):
            raise AssertionError(f"Item {list_float} is not a list of floats.")

    # boolians:
    all_bools = ["filter_cells", "multiple_labels_per_subSample", "attr_bool"]
    for item in all_bools:
        if not isinstance(requirements[item], bool):
            raise AssertionError(f"Item {item} is not a bool.")

    # optional keys:
    if requirements["filter_cells"]:
        if not {"filter_column", "filter_value"}.issubset(key_set):
            raise AssertionError(
                "If filter_cells is True, filter_column and filter_value must be present."
            )

        if not isinstance(requirements["filter_column"], list):
            raise AssertionError(
                "filter_column should be a list containing the column to filter by."
            )
        if not isinstance(
            requirements["filter_value"], (int, float, bool, np.integer, np.floating)
        ):
            raise AssertionError("filter_value should be an int, float or bool.")

    return requirements


def validate_best_config(config_file):
    """
    Test to check if the best configuration file exists and is in the correct format.
        Args:
        config_file (str,dict): Path to the requirements file or requirements dict.
    """
    if isinstance(config_file, dict):
        best_config_dict = config_file
    else:
        if isinstance(config_file, (str, Path)):
            if not Path(config_file).exists():
                raise FileNotFoundError(
                    f"Best configuration file {config_file} does not exist."
                )

            with open(Path(config_file), "rb") as file:
                best_config_dict = pickle.load(file)
        else:
            raise TypeError("config_file should be a string or Path object.")

    required_keys = set(
        [
            "layer_1",
            "input_layer",
            "droupout_rate",
            "output_layer",
            "attr_bool",
            "anker_value",
            "radius_distance",
            "version",
        ]
    )

    key_set = set(best_config_dict.keys())

    # Check if all required keys are present
    missing_keys = required_keys - key_set
    if len(missing_keys) > 0:
        raise AssertionError(f"Missing keys in best config: {missing_keys}")

    # Check for correct format of the keys:
    # int:
    all_ints = ["layer_1", "input_layer", "output_layer", "version"]
    for item in all_ints:
        if not isinstance(best_config_dict[item], (int, np.integer)):
            raise AssertionError(f"Item {item} is not an integer.")

    # float:
    all_floats = ["droupout_rate"]
    for item in all_floats:
        if not isinstance(best_config_dict[item], (float, np.floating)):
            raise AssertionError(f"Item {item} is not a float.")

    # bool:
    all_bools = ["attr_bool"]
    for item in all_bools:
        if not isinstance(best_config_dict[item], bool):
            raise AssertionError(f"Item {item} is not a bool.")

    # float or int:
    all_numbers = ["anker_value", "radius_distance"]
    for item in all_numbers:
        if not isinstance(
            best_config_dict[item], (float, int, np.integer, np.floating)
        ):
            raise AssertionError(f"Item {item} is not a float or int.")
