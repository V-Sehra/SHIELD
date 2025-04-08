import pytest
import pickle
from pathlib import Path


def test_all_keys_in_req(req_file):
    """
    Test to check if the requirements file contains all required keys
    and if the values are in the correct format.

    Args:
        req_file (str): Path to the requirements file or requirements dict.
    """
    if isinstance(req_file,dict):
        requirements = req_file
    elif isinstance(req_file, (str,Path)):
        with open(req_file, 'rb') as file:
            requirements = pickle.load(file)
    else:
        raise TypeError("req_file should be a string, Path object or dict.")

    # Define the required keys
    required_keys = set(['path_raw_data', 'path_training_results',
                        'path_to_model', 'label_column',
                        'cell_type_names', 'label_dict',
                        'eval_columns', 'col_of_interest',
                        'col_of_variables', 'minimum_number_cells',
                        'radius_distance_all', 'fussy_limit_all',
                        'anker_value_all','filter_cells',
                        'anker_cell_selction_type', 'multiple_labels_per_subSample',
                        'batch_size', 'learning_rate',
                        'input_layer', 'layer_1',
                        'out_put_layer', 'droupout_rate', 'attr_bool',
                        'augmentation_number', 'X_col_name', 'Y_col_name',
                        'measument_sample_name',
                        'validation_split_column', 'number_validation_splits',
                        'test_set_fold_number', 'voro_neighbours'])

    key_set = set(requirements.keys())

    if required_keys == key_set:
        print("All required keys are present.")
        return

    # Check if all required keys are present
    missing_keys = required_keys - key_set
    if len(missing_keys) > 0:
        raise AssertionError(f"Missing keys in requirements: {missing_keys}")

    # Check for correct format of the keys:
    # Paths:
    all_paths = {item for item in key_set if item.lower().startswith("path")}
    for path in all_paths:
        if not isinstance(requirements[path], (str, Path)):
            raise AssertionError(f"Path {path} is not a string or a Path object.")

    # int:
    all_ints = ['minimum_number_cells', 'batch_size', 'input_layer', 'layer_1',
                'out_put_layer', 'augmentation_number', 'voro_neighbours']
    for item in all_ints:
        if not isinstance(requirements[item], int):
            raise AssertionError(f"Item {item} is not an integer.")

    # string:
    # label_column
    all_strings = ['label_column', 'anker_cell_selction_type', 'validation_split_column', 'X_col_name',
                   'Y_col_name', 'measument_sample_name']
    for item in all_strings:
        if not isinstance(requirements[item], str):
            raise AssertionError(f"Item {item} is not a string.")

    # list of strings:
    all_lists_str = ['cell_type_names', 'markers', 'eval_columns', 'col_of_interest',
                     'col_of_variables']
    for list_str in all_lists_str:
        if not isinstance(requirements[list_str], list) or not all(isinstance(i, str) for i in requirements[list_str]):
            raise AssertionError(f"Item {list_str} is not a list of strings.")

    # list of floats or ints:
    all_lists_numbers = ['radius_distance_all', 'fussy_limit_all',
                        'anker_value_all', 'droupout_rate',
                        'number_validation_splits','test_set_fold_number']

    for list_float in all_lists_numbers:
        if not isinstance(requirements[list_float], list) or not all(
                isinstance(i, (float,int)) for i in requirements[list_float]):

            raise AssertionError(f"Item {list_float} is not a list of floats.")

    # boolians:
    all_bools = ['filter_cells', 'multiple_labels_per_subSample', 'attr_bool']
    for item in all_bools:
        if not isinstance(requirements[item], bool):
            raise AssertionError(f"Item {item} is not a bool.")

    # optional keys:
    if requirements['filter_cells']:
        if not {'filter_column', 'filter_value'}.issubset(key_set):
            raise AssertionError("If filter_cells is True, filter_column and filter_value must be present.")

        if not isinstance(requirements['filter_column'], list):
            raise AssertionError("filter_column should be a list containing the column to filter by.")
        if not isinstance(requirements['filter_value'], (int, float,bool)):
            raise AssertionError("filter_value should be an int, float or bool.")


def test_best_config(config_file):
    """
    Test to check if the best configuration file exists and is in the correct format.
        Args:
        config_file (str,dict): Path to the requirements file or requirements dict.
    """
    if isinstance(config_file,dict):
        best_config_dict = config_file
    else:
        if isinstance(config_file, (str, Path)):
            if not Path(config_file).exists():
                raise FileNotFoundError(f"Best configuration file {config_file} does not exist.")

            with open(Path(config_file), 'rb') as file:
                best_config_dict = pickle.load(file)
        else:
            raise TypeError("config_file should be a string or Path object.")


    required_keys = set(['layer_one', 'input_dim', 'droup_out_rate', 'final_layer',
                         'attr_bool', 'anker_value', 'radius_distance', 'fussy_limit','version'])

    key_set = set(best_config_dict.keys())


    # Check if all required keys are present
    missing_keys = required_keys - key_set
    if len(missing_keys) > 0:
        raise AssertionError(f"Missing keys in best config: {missing_keys}")

    # Check for correct format of the keys:
    # int:
    all_ints = ['layer_one', 'input_dim', 'final_layer','version']
    for item in all_ints:
        if not isinstance(best_config_dict[item], int):
            raise AssertionError(f"Item {item} is not an integer.")

    # float:
    all_floats = ['droup_out_rate', 'fussy_limit']
    for item in all_floats:
        if not isinstance(best_config_dict[item], float):
            raise AssertionError(f"Item {item} is not a float.")

    # bool:
    all_bools = ['attr_bool']
    for item in all_bools:
        if not isinstance(best_config_dict[item], bool):
            raise AssertionError(f"Item {item} is not a bool.")

    # float or int:
    all_numbers = ['anker_value','radius_distance']
    for item in all_numbers:
        if not isinstance(best_config_dict[item], (float,int)):
            raise AssertionError(f"Item {item} is not a float or int.")