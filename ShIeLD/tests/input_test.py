import pytest
import pickle
from pathlib import Path

def test_all_keys_in_req(req_file):
    """
    Test if all required keys are present in the requirements file.

    Args:
        req_file (str): Path to the requirements file.
    """
    with open(req_file, 'rb') as file:
        requirements = pickle.load(file)

    # Define the required keys
    required_keys = set('path_raw_data', 'path_training_results', 'path_to_model',
                     'cell_type_names', 'label_dict', 'label_column',
                     'eval_columns', 'col_of_interest',
                     'col_of_variables','minimum_number_cells',
                     'radius_distance_all', 'fussy_limit_all',
                     'anker_value_all', 'filter_column', 'filter_value',
                     'anker_cell_selction_type', 'multiple_labels_per_subSample',
                     'batch_size', 'learning_rate',
                     'input_layer', 'layer_1',
                     'out_put_layer', 'droupout_rate', 'attr_bool',
                     'augmentation_number', 'X_col_name', 'Y_col_name',
                    'measument_sample_name', 'filter_cells',
                    'validation_split_column', 'number_validation_splits',
                    'test_set_fold_number', 'voro_neighbours')

    key_set = set(requirements.keys())

    if required_keys == key_set:
        print("All required keys are present.")
        return

    # Check if all required keys are present
    missing_keys = required_keys - key_set
    if len(missing_keys)>0:
        raise AssertionError(f"Missing keys in requirements: {missing_keys}")

    # Check for correct format of the keys:
    # Paths:
    all_paths = {item for item in key_set if item.lower().startswith("path")}
    for path in all_paths:
        if not isinstance(requirements[path], (str,Path)):
            raise AssertionError(f"Path {path} is not a string or a Path object.")

    # int:
    all_ints = ['minimum_number_cells', 'batch_size','input_layer','layer_1',
                'out_put_layer','augmentation_number','voro_neighbours']
    for item in all_ints:
        if not isinstance(requirements[item], int):
            raise AssertionError(f"Item {item} is not an integer.")

    # list of strings:
    all_lists_str = ['cell_type_names','markers', 'eval_columns', 'col_of_interest',
                     'col_of_variables', 'filter_column']
    for list_str in all_lists_str:
        if not isinstance(requirements[list_str], list) or not all(isinstance(i, str) for i in requirements[list_str]):
            raise AssertionError(f"Item {list_str} is not a list of strings.")

    #list of floats:
    all_lists_floats = ['radius_distance_all', 'fussy_limit_all',
                        'anker_value_all','droupout_rate','number_validation_splits'
                        'test_set_fold_number']
    for list_float in all_lists_floats:
        if not isinstance(requirements[list_float], list) or not all(isinstance(i, float) for i in requirements[list_float]):
            raise AssertionError(f"Item {list_float} is not a list of floats.")


    #string:
    #label_column