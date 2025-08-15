from pathlib import Path, PosixPath
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import pickle
from tqdm import tqdm
import os

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from . import model_utils

from typing import List, Tuple, Optional, Union, Dict
import torch_geometric
import optuna


def get_eval_metrics(target_y, pred_y):
    """
    Compute evaluation metrics for classification: F1 score, balanced accuracy, and AUC.

    Parameters
    ----------
    target_y : array-like
        Ground truth class labels.
    pred_y : array-like
        Predicted class labels.

    Returns
    -------
    Tuple[float, float, float]
        Weighted F1 score, balanced accuracy, and macro AUC score (if computable).
    """
    # Weighted F1 Score across all classes
    f1 = f1_score(target_y, pred_y, average='weighted')

    # Balanced Accuracy accounts for class imbalance
    bacc = balanced_accuracy_score(target_y, pred_y)

    # Attempt to compute multiclass AUC (One-vs-One)
    try:
        auc = roc_auc_score(target_y, pred_y, multi_class='ovo', average='macro')
    except ValueError:
        # AUC undefined if only one class is present in target
        auc = float('nan')

    return f1, bacc, auc


def eval_model_optuna(
        model: torch.nn.Module,
        args, data_loader: Union[torch.utils.data.DataLoader, torch_geometric.data.DataLoader],
        device: str = 'cpu',

) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a trained model on validation/test data.

    Parameters
    ----------
    path_to_data : Path
        Path to the directory containing the dataset.
    path_to_FileList_val : PosixPath
        Path to the file listing validation/test samples.
    config_file_data : dict
        Dataset-specific configuration (e.g., panel, Timepoint...).
    config_file_train : dict
        Training-specific configuration (e.g., batch size).
    model : torch.nn.Module
        Trained model to be evaluated.
    args : Namespace
        Argument parser or config object containing model options like input_shape.
    device : str, optional
        Target device ('cpu' or 'cuda'), by default 'cpu'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Ground truth labels and predicted labels as numpy arrays.
    """

    # Load validation/test data

    model.eval()
    pred_val = []
    target_val = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            predictions = model(data)
            predicted_labels = predictions.argmax(dim=1)

            pred_val.extend(predicted_labels.cpu().numpy())
            target_val.extend(data.y.cpu().numpy())

    return np.array(target_val), np.array(pred_val)


def train_loop_optuna(config_file: Dict, model,
                      data_loader: Union[torch.utils.data.DataLoader, torch_geometric.data.DataLoader],
                      optimizer: torch.optim.Optimizer,
                      criterion: torch.nn.Module,
                      args,
                      device='cpu') -> Tuple[torch.nn.Module, List]:
    '''
    Function to train a model with early stopping criteria based on training loss plateau.
    config_file (dict): Configuration file containing training parameters.
    model (torch.nn.Module): The model to be trained.
    data_loader (torch.utils.data.DataLoader | torch_geometric.data import Dataset): DataLoader for training data.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    criterion (torch.nn.Module): Loss function for training.
    args (argparse.Namespace): Command line arguments containing input shape and other parameters.
    device (str): Device to run the training on ('cpu' or 'cuda').
    '''

    min_epochs = config_file['epochs']
    if min_epochs < 10:
        min_epochs = 10
        print("Minimum epochs set to 10 for early stopping criteria.")
    train_stop = False
    train_losses = []

    while train_stop is False:
        model.train()
        epoch_losses = []
        if args.input_shape != 'graph':
            for x_batch, y_batch in data_loader:
                x_batch = torch.flatten(x_batch, start_dim=1).to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
        else:
            for data in data_loader:
                y_batch = data.y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data.to(device)), y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)

        if len(train_losses) > min_epochs:
            recent_3 = np.mean(train_losses[-3:])
            previous_5 = np.mean(train_losses[-8:-3])  # 5 epochs before the last 3
            if (recent_3 > previous_5):
                print(f"[Early Stopping] Train loss plateaued (last 3: {recent_3:.4f} > prev 5: {previous_5:.4f}).")
                break
            if len(train_losses) >= min_epochs + 10:
                print(f"[Training stop] training stopped after {len(train_losses)} epochs. Last loss: {avg_loss:.4f}")
                break
    return model, train_losses


def objective(
        trial: optuna.Trial,
        model_type: str,
        data_loader_train: torch.utils.data.DataLoader,
        data_loader_test: torch.utils.data.DataLoader,
        output_dim: int,
        input_dim: int,
        config_file_train: Dict,
        args,
        device: torch.device
) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): Current Optuna trial.
        model_type (str): Model architecture key (e.g., 'GAT', 'GNN').
        data_loader (DataLoader): Training DataLoader.
        output_dim (int): Number of output classes.
        input_dim (int): Input feature dimension.

        config_file_train (dict): Training configuration.
        args (Any): Additional runtime arguments (e.g., argparse.Namespace).
        device (torch.device): Device to train on.

    Returns:
        float: Score based on selected `args.scoring` metric.
    """

    # --- Hyperparameter Search Space ---
    if args.model_type == 'MLP':
        params = {
            'dropout': trial.suggest_float("dropout", *config_file_train['dropout']),
            'lr': trial.suggest_float("lr", *config_file_train['lr'], log=True),
            'num_layers': trial.suggest_int("num_layers", *config_file_train['hidden_layers_number']),
            'weight_decay': trial.suggest_float("weight_decay", *config_file_train['weight_decay'], log=True),
            'batch_norm_bool': trial.suggest_categorical("batch_norm_bool", config_file_train['batch_norm_bool']),
            'layer_norm_bool': trial.suggest_categorical("layer_norm_bool", config_file_train['layer_norm_bool']),
            'activation': trial.suggest_categorical("activation", config_file_train['activation']),
            'output_dim': output_dim,
            'input_dim': input_dim
        }
    elif args.model_type == 'GNN' or args.model_type == 'GAT':
        params = {
            'dropout': trial.suggest_float("dropout", *config_file_train['dropout']),
            'lr': trial.suggest_float("lr", *config_file_train['lr'], log=True),
            'num_layers': trial.suggest_int("num_layers", *config_file_train['hidden_layers_number']),
            'weight_decay': trial.suggest_float("weight_decay", *config_file_train['weight_decay'], log=True),
            'batch_norm_bool': trial.suggest_categorical("batch_norm_bool", config_file_train['batch_norm_bool']),
            'graph_norm_bool': trial.suggest_categorical("graph_norm_bool", config_file_train['graph_norm_bool']),
            'layer_norm_bool': trial.suggest_categorical("layer_norm_bool", config_file_train['layer_norm_bool']),
            'similarity_typ': trial.suggest_categorical("similarity_typ", config_file_train['similarity_typ']),
            'activation': trial.suggest_categorical("activation", config_file_train['activation']),
            'output_dim': output_dim,
            'input_dim': input_dim
        }
        # --- Logical Constraint ---
        if params['batch_norm_bool'] and params['graph_norm_bool']:
            raise optuna.exceptions.TrialPruned("Cannot use both batch norm and graph norm simultaneously.")

    # --- Hidden Dimensions Per Layer ---
    hidden_dims = [
        trial.suggest_categorical(f"hidden_dim_{i}", [16, 32, 48, 64, 128])
        for i in range(params['num_layers'])
    ]
    params['hidden_dims'] = hidden_dims

    # --- Model Initialization ---
    model = model_utils.load_model(model_type=model_type, params=params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    print(model)
    print("Training:")

    # --- Training ---
    model, train_losses = train_loop_optuna(
        config_file=config_file_train,
        model=model,
        data_loader=data_loader_train,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        args=args
    )

    print("Validating:")

    # --- Validation ---
    target_val, pred_val = eval_model_optuna(
        data_loader=data_loader_test,
        model=model,
        args=args,
        device=device
    )

    # --- Evaluation Metrics ---
    f1, bacc, auc = get_eval_metrics(target_y=target_val, pred_y=pred_val)

    # --- Return Selected Metric ---
    metric_to_optimise = {
        'f1_weighted': f1,
        'balanced_accuracy': bacc,
        'roc_auc_score': auc
    }[args.scoring]

    return metric_to_optimise


def early_stopping(loss_epoch: List, patience: Optional[int] = 15) -> bool:
    """
    This function checks if the training process should be stopped early based on the change in loss over epochs.
    If the change in loss between the last two epochs is less than 0.001 and the number of epochs is greater than the patience threshold, the function returns True, indicating that training should be stopped.
    Otherwise, it returns False, indicating that training should continue.

    Parameters:
    loss_epoch (list): A list of loss values for each epoch.
    patience (int): The number of epochs to wait before stopping training if the change in loss is less than 0.001. Default is 15.

    Returns:
    bool: True if training should be stopped, False otherwise.
    """
    if patience is None:
        patience = 5

    if patience < 2:
        raise ValueError("Patience should be greater than 1.")

    if len(loss_epoch) >= patience:

        if (loss_epoch[-2] - loss_epoch[-1]) < 0.001:
            return True
        else:
            return False
    else:
        return False


def get_train_results_csv(requirement_dict: dict) -> Tuple[DataFrame, PosixPath]:
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
    csv_file_path = Path(path_save_csv_models / f'validation_split_results_training.csv')

    # If the file exists, load it; otherwise, create an empty DataFrame with predefined columns
    if csv_file_path.exists():
        training_results_csv = pd.read_csv(csv_file_path)

    else:
        training_results_csv = pd.DataFrame(columns=[
            'anker_value',  # Percentage of anchor cells used
            'radius_distance',  # Radius defining the neighborhood
            'fussy_limit',  # Threshold for fuzzy logic application
            'droupout_rate',  # Dropout rate
            'comment',  # General comments
            'comment_norm',  # Normalized comments
            'model_no',  # Model number
            'bal_acc_train',  # Balanced accuracy on the training set
            'train_f1_score',
            'bal_acc_validation',  # Balanced accuracy on the validation set
            'val_f1_score',
            'split_number',  # Validation split number
        ])

    return training_results_csv, csv_file_path  # Return the DataFram


def initiaize_loss(path: str, device: str, tissue_dict: dict,
                   noise_yLabel: Union[bool, str] = False) -> nn.CrossEntropyLoss:
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

    if noise_yLabel == False:
        if type(path) == str or type(path) == PosixPath:
            with open(path, 'rb') as f:
                all_train_file_names = pickle.load(f)
        elif type(path) == list:
            all_train_file_names = path

        class_weights = []
        for origin in tissue_dict.keys():
            number_tissue_graphs = len([file_name for file_name in all_train_file_names
                                        if origin in file_name])
            class_weights.append(1 - (number_tissue_graphs / len(all_train_file_names)))

        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    elif noise_yLabel is True:
        print('initializing loss with noise labels: No Voronoi')

        class_weights = np.zeros((len(tissue_dict.keys()), 1))

        for files in tqdm(os.listdir(Path(path))):
            try:
                g = torch.load(Path(path) / files)
                class_weights[g['y']] += 1
            except:
                print('skipping file: ', files, ' due to error in loading or missing label')

        class_weights = 1 - (class_weights.T / sum(class_weights))
        class_weights = torch.tensor(class_weights.flatten(), dtype=torch.float32).to(device)

    else:
        print('initializing loss with noise labels: ', noise_yLabel)

        if noise_yLabel != 'even' and noise_yLabel != 'prob':
            raise ValueError(f"Invalid noise_yLabel: {noise_yLabel}. Must be 'even' or 'prob'.")

        class_weights = np.zeros((len(tissue_dict.keys()), 1))

        for files in tqdm(os.listdir(Path(path))):
            try:
                g = torch.load(Path(path) / files)
                if f'y_noise_{noise_yLabel}' in g:
                    class_weights[g[f'y_noise_{noise_yLabel}']] += 1
                else:
                    class_weights[g['y']] += 1
            except:
                print('skipping file: ', files, ' due to error in loading or missing label')

        class_weights = 1 - (class_weights.T / sum(class_weights))
        class_weights = torch.tensor(class_weights.flatten(), dtype=torch.float32).to(device)

        print('done!')

    return nn.CrossEntropyLoss(weight=class_weights).to(device)


def train_loop_shield(
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fkt: nn.CrossEntropyLoss,
        attr_bool: bool,
        device: str,
        patience: Optional[int] = 5,
        noise_yLabel: Union[bool, str] = False
) -> Tuple[torch.nn.Module, List[float]]:
    """
    Trains the ShIeLD model using the provided optimizer, data loader, and loss function.

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
    Returns:
        Tuple[torch.nn.Module, List[float]]: The trained model and a list of training loss values per epoch.
    """

    # List to store the loss values for each epoch
    train_loss = []

    # Flag for early stopping condition
    early_stopping_bool = False

    print('Start training')

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
                noise_yLabel=noise_yLabel
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
        early_stopping_bool = early_stopping(loss_epoch=train_loss, patience=patience)

    # Return the trained model and recorded loss values
    return model, train_loss
