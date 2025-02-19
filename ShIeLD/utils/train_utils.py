from pathlib import Path
import pandas as pd
from pandas import DataFrame



def get_train_results_csv(requirement_dict: dict, split_number: int) -> DataFrame:
    """
    Retrieves or initializes a CSV file containing training results.

    Parameters:
    - requirement_dict (dict): A dictionary containing various requirements, including the path for saving training results.
    - split_number (int): The identifier for the validation split.

    Returns:
    - pd.DataFrame: A DataFrame containing training results, either loaded from an existing CSV or initialized with default columns.
    """

    # Get the path where the training results CSV should be stored
    path_save_csv_models = requirement_dict['path_training_results']

    # Ensure the directory exists (creates it if necessary)
    path_save_csv_models.mkdir(parents=True, exist_ok=True)

    # Define the expected CSV file path based on the split number
    csv_file = Path(path_save_csv_models / f'validation_split_{split_number}_results_training.csv')

    # If the file exists, load it; otherwise, create an empty DataFrame with predefined columns
    if csv_file.exists():
        training_results_csv = pd.read_csv(csv_file)
    else:
        training_results_csv = pd.DataFrame(columns=[
            'prozent_of_anker_cells',  # Percentage of anchor cells used
            'radius_neibourhood',      # Radius defining the neighborhood
            'fussy_limit',             # Threshold for fuzzy logic application
            'dp',                      # Dropout rate
            'comment',                 # General comments
            'comment_norm',            # Normalized comments
            'model_no'                 # Model number
        ])

    return training_results_csv  # Return the DataFrame



