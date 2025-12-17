from pathlib import Path, PosixPath
import pandas as pd
from pandas import DataFrame
import numpy as np

import pickle
from tqdm import tqdm
import os

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from . import model_utils

from typing import List, Tuple, Optional, Union


def early_stopping(
    loss_epoch: List, patience: Optional[int] = 15, max_epochs: Optional[int] = None
) -> bool:
    """
    This function checks if the training process should be stopped early based on the change in loss over epochs.
    If the change in loss between the last two epochs is less than 0.001 and the number of epochs is greater than the patience threshold, the function returns True, indicating that training should be stopped.
    Otherwise, it returns False, indicating that training should continue.

    Parameters:
    loss_epoch (list): A list of loss values for each epoch.
    patience (int): The number of epochs to wait before stopping training if the change in loss is less than 0.001. Default is 15.
    max_epochs (int, optional): Maximum number of epochs to train. If None, training continues until early stopping / 2*patience is triggered.
    Returns:
    bool: True if training should be stopped, False otherwise.
    """
    if patience is None:
        patience = 8
    if max_epochs is None:
        max_epochs = 2 * patience

    if patience < 8:
        raise ValueError("Patience should be equal to or greater than 8.")

    if len(loss_epoch) >= max_epochs:
        return True

    if len(loss_epoch) >= patience:
        if (np.mean(loss_epoch[-8:-5]) - np.mean(loss_epoch[-3:])) < 0.001:
            return True
        else:
            return False

    return False


def get_train_results_csv(
    requirement_dict: dict, column_names: Optional[list] = None
) -> Tuple[DataFrame, PosixPath]:
    """
    Retrieves or initializes a CSV file containing training results.

    Parameters:
    - requirement_dict (dict): A dictionary containing various requirements, including the path for saving training results.
    - split_number (int): The identifier for the validation split.
    - column_names (list, optional): A list of column names for the DataFrame. If None, default column names are used.

    Returns:
    - pd.DataFrame: A DataFrame containing training results, either loaded from an existing CSV or initialized with default columns.
    """
    if column_names is None:
        column_names = [
            "anker_value",  # Percentage of anchor cells used
            "radius_distance",  # Radius defining the neighborhood
            "fussy_limit",  # Threshold for fuzzy logic application
            "dropout_rate",  # Dropout rate
            "comment",  # General comments
            "comment_norm",  # Normalized comments
            "model_no",  # Model number
            "bal_acc_train",  # Balanced accuracy on the training set
            "train_f1_score",
            "bal_acc_validation",  # Balanced accuracy on the validation set
            "val_f1_score",
            "split_number",  # Validation split number
        ]
    # Get the path where the training results CSV should be stored
    path_save_csv_models = requirement_dict["path_training_results"]

    # Ensure the directory exists (creates it if necessary)
    path_save_csv_models.mkdir(parents=True, exist_ok=True)

    # Define the expected CSV file path based on the split number
    csv_file_path = Path(path_save_csv_models / "validation_split_results_training.csv")

    # If the file exists, load it; otherwise, create an empty DataFrame with predefined columns
    if csv_file_path.exists():
        training_results_csv = pd.read_csv(csv_file_path)

    else:
        training_results_csv = pd.DataFrame(columns=column_names)

    return training_results_csv, csv_file_path  # Return the DataFram


def initialize_loss(
    path: str, device: str, tissue_dict: dict, noise_yLabel: Union[bool, str] = False
) -> nn.CrossEntropyLoss:
    """
    This function initializes the loss function as weighted loss.
    It calculates the class weights based on the number of graphs for each tissue type in the training data.
    The class weights are used to initialize the CrossEntropyLoss function from PyTorch.

    Parameters:
    path (str): The path to the file containing the training file names.
    device (str): The device to which the tensor of class weights should be moved.
    tissue_dict (dict): A dictionary mapping tissue types to integers.
    noise_yLabel (bool | str): If not False ['even','prob']. Will select the noise labels
                                CAUTION: needed to be element of the data set
                                'even': label selected evenly
                                'prob': label selected according to the accurance of the cell types within a voronoi

    Returns:
    nn.CrossEntropyLoss: The initialized loss function with class weights.
    """

    if noise_yLabel == False:  # noqa: E712
        if isinstance(path, str) or isinstance(path, PosixPath):
            with open(path, "rb") as f:
                all_train_file_names = pickle.load(f)
        elif isinstance(path, list):
            all_train_file_names = path

        class_weights = []
        for origin in tissue_dict.keys():
            number_tissue_graphs = len(
                [file_name for file_name in all_train_file_names if origin in file_name]
            )
            class_weights.append(1 - (number_tissue_graphs / len(all_train_file_names)))

        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    elif noise_yLabel is True:
        print("initializing loss with noise labels: No Voronoi")

        class_weights = np.zeros((len(tissue_dict.keys()), 1))

        for files in tqdm(os.listdir(Path(path))):
            try:
                g = torch.load(Path(path) / files)
                class_weights[g["y"]] += 1
            except:  # noqa: E722
                print(
                    "skipping file: ",
                    files,
                    " due to error in loading or missing label",
                )

        class_weights = 1 - (class_weights.T / sum(class_weights))
        class_weights = torch.tensor(class_weights.flatten(), dtype=torch.float32).to(
            device
        )

    else:
        print("initializing loss with noise labels: ", noise_yLabel)

        if noise_yLabel != "even" and noise_yLabel != "prob":
            raise ValueError(
                f"Invalid noise_yLabel: {noise_yLabel}. Must be 'even' or 'prob'."
            )

        class_weights = np.zeros((len(tissue_dict.keys()), 1))

        for files in tqdm(os.listdir(Path(path))):
            try:
                g = torch.load(Path(path) / files)
                if f"y_noise_{noise_yLabel}" in g:
                    class_weights[g[f"y_noise_{noise_yLabel}"]] += 1
                else:
                    class_weights[g["y"]] += 1
            except:  # noqa: E722
                print(
                    "skipping file: ",
                    files,
                    " due to error in loading or missing label",
                )

        class_weights = 1 - (class_weights.T / sum(class_weights))
        class_weights = torch.tensor(class_weights.flatten(), dtype=torch.float32).to(
            device
        )

        print("done!")

    return nn.NLLLoss(weight=class_weights).to(device)


def train_loop_shield(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fkt: nn.CrossEntropyLoss,
    attr_bool: bool,
    device: str,
    patience: Optional[int] = 5,
    noise_yLabel: Union[bool, str] = False,
    max_epochs: Optional[int] = None,
) -> Tuple[torch.nn.Module, List[float]]:
    """
    Trains the SHIELD model using the provided optimizer, data loader, and loss function.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer responsible for updating model parameters.
        model (torch.nn.Module): The neural network model to be trained.
        data_loader (DataLoader): Data loader providing batches of training samples.
        loss_fkt (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function used for training.
        attr_bool (bool): Boolean flag indicating whether edge attributes should be used.
        device (str): The device ('cuda' or 'cpu') where computations are performed.
        patience (int): Number of epochs to wait before stopping training if no improvement is observed.
        noise_yLabel (bool | str): If not False ['even','prob']. Will select the noise labels
                                    CAUTION: needed to be element of the data set
                                    'even': label selected evenly
                                    'prob': label selected according to the accurance of the cell types within a voronoi
        max_epochs (int, optional): Maximum number of epochs to train. If None, training continues until early stopping / 2*patience is triggered.
    Returns:
        Tuple[torch.nn.Module, List[float]]: The trained model and a list of training loss values per epoch.
    """

    # List to store the loss values for each epoch
    train_loss = []

    if max_epochs is not None:
        max_epochs = int(max_epochs)

    # Flag for early stopping condition
    early_stopping_bool = False

    # Continue training until the early stopping condition is met
    while not early_stopping_bool:
        # List to store batch-wise loss values for the current epoch
        loss_batch = []

        # Iterate over batches in the data loader
        for train_sample in tqdm(data_loader):
            # Zero the gradients before the backward pass
            optimizer.zero_grad()

            # Perform a prediction step on the current batch
            prediction, attention, output, y, sample_ids = model_utils.prediction_step(
                batch_sample=train_sample,
                model=model,
                device=device,
                per_patient=False,  # Whether to track patient-level predictions
                noise_yLabel=noise_yLabel,
            )

            # Compute the loss between model output and ground truth labels
            loss = loss_fkt(output, y)

            # Perform backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Store batch loss
            loss_batch.append(loss.item())

        # Compute the average loss for the current epoch
        train_loss.append(np.mean(loss_batch))

        # Check if early stopping should be triggered based on loss history
        early_stopping_bool = early_stopping(
            loss_epoch=train_loss, patience=patience, max_epochs=max_epochs
        )

    # Return the trained model and recorded loss values
    return model, train_loss
