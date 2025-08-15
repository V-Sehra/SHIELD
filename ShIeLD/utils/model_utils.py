#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:18:15 2024

@author: Vivek
"""

import torch
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import GATv2Conv, GCNConv
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm

from typing import List, Tuple, Optional, Union
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from torch_geometric.nn import global_mean_pool, GraphNorm
from torch.nn import BatchNorm1d


def get_acc_metrics(model: torch.nn.Module, data_loader: DataLoader,
                    device: Union[torch.device, str], noise_yLabel: Union[bool, str] = False) -> Tuple[float, float]:
    """
    Computes the balanced accuracy and weighted F1 score for a model on the given dataset.

    Parameters:
    - model (torch.nn.Module): The trained PyTorch model used for evaluation.
    - data_loader (DataLoader): A PyTorch DataLoader providing batches of graph data.
    - device (torch.device or str): The device (CPU or GPU) to perform computations on.
    - noise_yLabel (bool or str): If False, uses true labels. If 'even' or 'prob',
      uses noisy labels provided in the dataset accordingly.

    Returns:
    - tuple: A tuple containing:
        - balanced_accuracy (float): The balanced accuracy score.
        - f1 (float): The weighted F1 score.
    """
    model_prediction = []
    true_label = []

    for sample in data_loader:
        prediction, attention, output, y, sample_ids = prediction_step(
            batch_sample=sample,
            model=model,
            device=device,
            per_patient=False,  # Whether to track patient-level predictions
            noise_yLabel=noise_yLabel
        )

        _, value_pred = torch.max(output, dim=1)

        model_prediction.extend(value_pred.cpu())
        true_label.extend(y.cpu())

    return balanced_accuracy_score(true_label, model_prediction), f1_score(true_label, model_prediction,
                                                                           average='weighted'), confusion_matrix(
        y_true=true_label, y_pred=model_prediction, normalize='true')


def move_to_device(data: Union[torch.tensor, List], device: Union[torch.device, str]):
    """
    Moves the input data to the specified device (CPU or GPU).

    Parameters:
    - data (torch.Tensor or list): The data to be moved.
    - device (torch.device): The device to which the data should be moved.

    Returns:
    - torch.Tensor or list: The data moved to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [item.to(device) for item in data]
    else:
        raise TypeError("Unsupported data type for moving to device.")


def prediction_step(batch_sample: List,
                    model: torch.nn.Module,
                    device: str,
                    per_patient: bool = False,
                    noise_yLabel: Union[bool, str] = False) -> Tuple[
    List, List, torch.Tensor, torch.Tensor, Optional[List]]:
    """
    Performs a prediction step using a given model on a batch of samples.

    Parameters:
    - batch_sample (list): A batch of samples containing graph data.
    - model (torch.nn.Module): The trained model used for predictions.
    - attr_bool (bool): If True, includes edge attributes (e.g., Euclidean distance).
    - device (torch.device): The device (CPU or GPU) where computations are performed.
    - per_patient (bool, optional): If True, returns patient/sample IDs for further analysis. Default is False.

    Returns:
    - prediction (list): Model predictions for each sample.
    - attention (list): Attention weights from the model (if applicable).
    - output (torch.Tensor): Stacked predictions across the batch.
    - y (torch.Tensor): Ground truth labels.
    - sample_ids (list or None): List of patient/sample IDs if `per_patient` is True, otherwise None.
    """
    batch_sample = move_to_device(data=batch_sample, device=device)

    prediction, attention = model(data_list=batch_sample)

    # Stack predictions into a single tensor
    output = torch.vstack(prediction)

    # Extract ground truth labels and move them to the same device as output
    if noise_yLabel == False or noise_yLabel == True:
        y = torch.tensor([sample.y for sample in batch_sample]).to(output.device)
    else:
        if noise_yLabel != 'even' and noise_yLabel != 'prob':
            raise ValueError(f"Invalid noise_yLabel: {noise_yLabel}. Must be 'even' or 'prob'.")

        y = []
        for sample in batch_sample:
            if f'y_noise_{noise_yLabel}' in sample:
                y.append(sample[f'y_noise_{noise_yLabel}'])
            else:
                y.append(sample[f'y'])
        y = torch.tensor(y).long().to(output.device)
    # Retrieve sample IDs if per_patient is True
    if per_patient:
        try:
            sample_ids = [sample.ids for sample in batch_sample]
        except AttributeError:  # Fallback if `ids` attribute is not available
            sample_ids = [sample.image for sample in batch_sample]
    else:
        sample_ids = None

    return prediction, attention, output, y, sample_ids


# Baseline Models

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout, activation_function,
                 batch_norm_bool, layer_norm_bool):
        super().__init__()

        self.activation_function = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU()
        }[activation_function]

        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.norm_layers.append(nn.LayerNorm(hidden_dims[0]))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.norm_layers.append(nn.LayerNorm(hidden_dims[i]))

        # Final output layer
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)

        self.batch_norm_bool = batch_norm_bool
        self.batch_norm = BatchNorm1d(input_dim)
        self.layer_norm_bool = layer_norm_bool

    def forward(self, x):
        if self.batch_norm_bool:
            x = self.batch_norm(x)

        for layer, norm in zip(self.layers, self.norm_layers):
            x = layer(x)
            if norm:
                x = norm(x)
            x = self.activation_function(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.classifier(x)
        return F.softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout, activation_function,
                 batch_norm_bool, layer_norm_bool, graph_norm_bool, similarity_typ):
        super().__init__()

        self.activation_function = {'relu': nn.ReLU(),
                                    'tanh': nn.Tanh(),
                                    'leaky_relu': nn.LeakyReLU(),
                                    'gelu': nn.GELU()
                                    }[activation_function]

        self.convs = torch.nn.ModuleList()
        self.layer_norm_list = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))

        self.layer_norm_list.append(GraphNorm(hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.convs.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))
            self.layer_norm_list.append(GraphNorm(hidden_dims[i]))

        # Final classifier
        self.classifier = torch.nn.Linear(hidden_dims[-1], output_dim)
        # Pre-normalization (on input)

        self.batch_norm_bool = batch_norm_bool
        self.batch_norm = BatchNorm(input_dim)

        self.graph_norm_bool = graph_norm_bool
        self.graph_norm = GraphNorm(input_dim)

        self.layer_norm_bool = layer_norm_bool

        self.similarity_typ = similarity_typ

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.similarity_typ == 'euc_dist':
            edge_attr = data.euclid.float()
        elif self.similarity_typ == 'cosine':
            edge_attr = data.cosine.float()
        elif self.similarity_typ == 'inv_euc_dist':
            edge_attr = 1 / data.euclid.float()
        elif self.similarity_typ is None:
            edge_attr = None
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_typ}")

        if self.batch_norm_bool:
            x = self.batch_norm(x)

        for conv, layer_norm in zip(self.convs, self.layer_norm_list):
            x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_attr))

            if self.layer_norm_bool:
                x = layer_norm(x, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


# Example GNN model
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout, activation_function,
                 batch_norm_bool, layer_norm_bool, graph_norm_bool, similarity_typ, edge_dim=1):
        super().__init__()

        self.activation_function = {'relu': nn.ReLU(),
                                    'tanh': nn.Tanh(),
                                    'leaky_relu': nn.LeakyReLU(),
                                    'gelu': nn.GELU()
                                    }[activation_function]

        self.convs = torch.nn.ModuleList()
        self.layer_norm_list = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(GATv2Conv(input_dim, hidden_dims[0], edge_dim=edge_dim))

        self.layer_norm_list.append(GraphNorm(hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.convs.append(GATv2Conv(hidden_dims[i - 1], hidden_dims[i], edge_dim=edge_dim))
            self.layer_norm_list.append(GraphNorm(hidden_dims[i]))

        # Final classifier
        self.classifier = torch.nn.Linear(hidden_dims[-1], output_dim)
        # Pre-normalization (on input)

        self.batch_norm_bool = batch_norm_bool
        self.batch_norm = BatchNorm(input_dim)

        self.graph_norm_bool = graph_norm_bool
        self.graph_norm = GraphNorm(input_dim)

        self.layer_norm_bool = layer_norm_bool

        self.similarity_typ = similarity_typ

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.similarity_typ == 'euc_dist':
            edge_attr = data.euclid.float()
        elif self.similarity_typ == 'cosine':
            edge_attr = data.cosine.float()
        elif self.similarity_typ == 'inv_euc_dist':
            edge_attr = 1 / data.euclid.float()
        elif self.similarity_typ is None:
            edge_attr = None
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_typ}")

        if self.batch_norm_bool:
            x = self.batch_norm(x)

        for conv, layer_norm in zip(self.convs, self.layer_norm_list):
            x = self.activation_function(conv(x=x, edge_index=edge_index, edge_attr=edge_attr))

            if self.layer_norm_bool:
                x = layer_norm(x, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class simple_1l_GNN(nn.Module):
    """
    A simple onlayer Graph Neural Network (GNN) using three GCNConv layers, batch normalization,
    dropout, and a final linear layer for classification.

    Attributes:
        conv1 (GCNConv): First graph convolutional layer.
        lin (nn.Linear): Final linear layer to map to output classes.
        dp (float): Dropout probability for regularization.
        Mean_agg (MeanAggregation): Aggregation function for node embeddings.
    """

    def __init__(self, num_of_feat, f_3, dp, noisy_edge: Union[bool, str] = False, f_final=2):
        """
        Initializes the GNN model.

        Args:
            num_of_feat (int): Number of input node features.
            f_1 (int): Number of features in the first GCN layer.
            dp (float): Dropout probability for regularization.
            f_final (int): Number of output classes (default: 2).

        """
        super(simple_1l_GNN, self).__init__()

        self.conv1 = GCNConv(num_of_feat, f_3)

        self.lin = Linear(f_3, f_final)

        self.dp = dp
        self.Mean_agg = MeanAggregation()  # Aggregates node embeddings into a graph representation
        self.noisy_edge = noisy_edge

    def forward(self, data_list):
        """
        Forward pass for processing multiple graph samples.

        Args:
            data_list (list[torch.Tensor]): List of the Dataobjects i.e:
                                            -node feature tensors, each corresponding to a graph.
                                            -edge index tensors defining graph connectivity.
                                            -edge attribute tensors [optional].

        Returns:
            tuple:
                - list[torch.Tensor]: List of softmax predictions for each input graph.
                - list[torch.Tensor]: List of attention scores from the GAT layer.
        """

        prediction = []  # List to store predictions for each graph

        sample_number = len(data_list)  # Number of graphs in the batch

        for idx in range(sample_number):
            x = data_list[idx].x.float()  # Convert node features to float

            if self.noisy_edge == False:
                edge_index = data_list[idx].edge_index_plate.long()  # Convert edge indices to long tensor
            else:
                edge_index = data_list[idx][
                    f'edge_index_plate_{self.noisy_edge}'].long()

            # Apply GAT convolution, with or without edge attributes
            x = self.conv1(x=x, edge_index=edge_index)

            x = F.relu(x)  # Apply ReLU activation
            x = F.dropout(x, p=self.dp, training=self.training)  # Apply dropout

            x = self.Mean_agg(x, dim=0)  # Aggregate node embeddings into a graph representation
            x = self.lin(x)  # Final linear transformation

            prediction.append(F.softmax(x, dim=1))  # Softmax activation for classification

        return prediction, []
