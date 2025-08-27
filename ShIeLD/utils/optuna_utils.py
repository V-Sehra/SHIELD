from pathlib import Path, PosixPath

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from . import model_utils, data_class

from typing import List, Tuple, Union, Dict
import torch_geometric
import optuna


def load_data_loader(model_type: str, path_to_graphs: Path, split_number: int,
                     requirements: Dict, fix_nodeSize: int = 132):
    train_folds = requirements['number_validation_splits'].copy()
    train_folds.remove(split_number)
    if model_type == 'GNN' or model_type == 'GAT':
        from torch_geometric.loader import DataLoader
        data_loader_train = DataLoader(
            data_class.graph_dataset(
                root=str(path_to_graphs / 'train' / 'graphs'),
                path_to_graphs=path_to_graphs,
                fold_ids=train_folds,
                requirements_dict=requirements,
                graph_file_names=f'train_set_validation_split_{split_number}_file_names.pkl',
            ),
            batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
        )

        data_loader_validation = DataLoader(
            data_class.graph_dataset(
                root=str(path_to_graphs / 'train' / 'graphs'),
                path_to_graphs=path_to_graphs,
                fold_ids=[split_number],
                requirements_dict=requirements,
                graph_file_names=f'validation_validation_split_{split_number}_file_names.pkl',
            ),
            batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
        )
    else:
        from torch.utils.data import DataLoader
        data_loader_train = DataLoader(
            data_class.fixed_dataset(
                root=str(path_to_graphs / 'train' / 'graphs'),
                path_to_graphs=path_to_graphs,
                fold_ids=train_folds,
                requirements_dict=requirements,
                graph_file_names=f'train_set_validation_split_{split_number}_file_names.pkl',
                fix_nodeSize=fix_nodeSize
            ),
            batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
        )
        data_loader_validation = DataLoader(
            data_class.fixed_dataset(
                root=str(path_to_graphs / 'train' / 'graphs'),
                path_to_graphs=path_to_graphs,
                fold_ids=[split_number],
                requirements_dict=requirements,
                graph_file_names=f'validation_validation_split_{split_number}_file_names.pkl',
                fix_nodeSize=fix_nodeSize
            ),
            batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
        )
    return data_loader_train, data_loader_validation


def load_model(model_type: str, params: Dict[str, Union[str, float, int, List]]) -> nn.Module:
    """
    Load and initialize a GNN model based on the specified type and parameters.

    Args:
        model_type (str): Type of the model to load. Expected: 'GAT' or 'GNN'.
        params (dict): Dictionary of model parameters, must contain:
            - input_dim (int)
            - hidden_dims (list[int])
            - output_dim (int)
            - dropout (float)
            - batch_norm_bool (bool)
            - layer_norm_bool (bool)
            - graph_norm_bool (bool)
            - similarity_typ (str)
            - activation (str)

    Returns:
        nn.Module: Instantiated model.

    Raises:
        ValueError: If the `model_type` is not recognized.
    """

    if model_type == 'GAT':
        model = model_utils.GAT(input_dim=params['input_dim'],
                                hidden_dims=params['hidden_dims'],
                                output_dim=params['output_dim'],
                                dropout=params['dropout'],
                                batch_norm_bool=params['batch_norm_bool'],
                                layer_norm_bool=params['layer_norm_bool'],
                                graph_norm_bool=params['graph_norm_bool'],
                                similarity_typ=params['similarity_typ'],
                                activation_function=params['activation']
                                )
    elif model_type == 'GNN':
        model = model_utils.GCN(
            input_dim=params['input_dim'],
            hidden_dims=params['hidden_dims'],
            output_dim=params['output_dim'],
            dropout=params['dropout'],
            batch_norm_bool=params['batch_norm_bool'],
            layer_norm_bool=params['layer_norm_bool'],
            graph_norm_bool=params['graph_norm_bool'],
            similarity_typ=params['similarity_typ'],
            activation_function=params['activation']
        )
    elif model_type == 'MLP':
        model = model_utils.MLP(input_dim=params['input_dim'],
                                hidden_dims=params['hidden_dims'],
                                output_dim=params['output_dim'],
                                dropout=params['dropout'],
                                batch_norm_bool=params['batch_norm_bool'],
                                layer_norm_bool=params['layer_norm_bool'],
                                activation_function=params['activation']
                                )

    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Must be 'GAT', 'GNN' or MLP.")

    return model


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
        model_type:str,
        model: torch.nn.Module, data_loader: Union[torch.utils.data.DataLoader, torch_geometric.data.DataLoader],
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
        if model_type == 'MLP':
            for x_batch, y_batch in data_loader:
                predictions = model(x_batch.to(device))
                predicted_labels = predictions.argmax(dim=1)
                
                pred_val.extend(predicted_labels.cpu().numpy())
                target_val.extend(y_batch.flatten().cpu().numpy())
        else:
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
                      model_type,
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
        if model_type == 'MLP':
            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch.to(device)), y_batch.flatten().to(device))
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
    model = load_model(model_type=model_type, params=params).to(device)
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
        model_type=model_type,
    )

    print("Validating:")

    # --- Validation ---
    target_val, pred_val = eval_model_optuna(
        data_loader=data_loader_test,
        model=model,
        device=device,
        model_type = model_type
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
