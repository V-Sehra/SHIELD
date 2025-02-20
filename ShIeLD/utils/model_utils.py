#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:18:15 2024

@author: Vivek
"""



import torch
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import GATv2Conv,GCNConv
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm

from typing import List, Tuple, Optional

from sklearn.metrics import balanced_accuracy_score


def get_balance_acc(model, data_loader,attr_bool,device):

    model_prediction = []
    true_label = []

    for sample in data_loader:
        prediction, attention, output, y, sample_ids = model_utils.prediction_step(
            batch_sample=sample,
            model=model,
            attr_bool=attr_bool,
            device=device,
            per_patient=False  # Whether to track patient-level predictions
        )

        _, value_pred = torch.max(output, dim=1)


        model_prediction.extend(value_pred.cpu())
        true_label.extend(y.cpu())

    return balanced_accuracy_score(true_label,model_prediction)




def prediction_step(batch_sample: List, model: torch.nn.Module, attr_bool: bool, device: str, per_patient: bool=False)  -> Tuple[List, List, torch.Tensor, torch.Tensor, Optional[List]]:
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

    # Move node features (x) and edge indices to the specified device
    sample_x = [sample.x.to(device) for sample in batch_sample]
    sample_edge = [sample.edge_index_plate.to(device) for sample in batch_sample]

    # If edge attributes are used, extract and pass them to the model
    if attr_bool:
        sample_att = [sample.plate_euc.to(device) for sample in batch_sample]
        prediction, attention = model(sample_x, sample_edge, sample_att)
    else:
        prediction, attention = model(sample_x, sample_edge)

    # Stack predictions into a single tensor
    output = torch.vstack(prediction)

    # Extract ground truth labels and move them to the same device as output
    y = torch.tensor([sample.y for sample in batch_sample]).to(output.device)

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

class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with three hidden layers, optional batch normalization,
    dropout regularization, and a softmax output activation.

    Attributes:
        layer_1 (nn.Linear): First fully connected layer.
        layer_2 (nn.Linear): Second fully connected layer.
        layer_3 (nn.Linear): Third fully connected layer.
        outputLayer (nn.Linear): Output layer.
        BatchNorm (nn.BatchNorm1d): Batch normalization layer applied to the input.
        Pre_norm (bool): Whether to apply batch normalization before passing through the network.
        dp (float): Dropout probability for regularization.
    """

    def __init__(self, input_dim, layer1, layer2, layer3, output_dim, dp, Pre_norm):
        """
        Initializes the neural network.

        Args:
            input_dim (int): Number of input features.
            layer1 (int): Number of neurons in the first hidden layer.
            layer2 (int): Number of neurons in the second hidden layer.
            layer3 (int): Number of neurons in the third hidden layer.
            output_dim (int): Number of output neurons (e.g., number of classes in classification).
            dp (float): Dropout probability (0-1) for regularization.
            Pre_norm (bool): If True, applies batch normalization to the input before passing through the layers.
        """
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, layer1)
        self.layer_2 = nn.Linear(layer1, layer2)
        self.layer_3 = nn.Linear(layer2, layer3)
        self.outputLayer = nn.Linear(layer3, output_dim)
        self.BatchNorm = nn.BatchNorm1d(input_dim)  # Normalize input features if Pre_norm is enabled
        self.Pre_norm = Pre_norm
        self.dp = dp  # Dropout probability

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the network, with softmax activation.
        """
        if self.Pre_norm:
            x = self.BatchNorm(x)  # Apply batch normalization to input if enabled

        # First hidden layer with ReLU activation and dropout
        x = F.relu(self.layer_1(x))
        x = F.dropout(x, p=self.dp, training=self.training)

        # Second hidden layer with ReLU activation and dropout
        x = F.relu(self.layer_2(x))
        x = F.dropout(x, p=self.dp, training=self.training)

        # Third hidden layer with ReLU activation and dropout
        x = F.relu(self.layer_3(x))
        x = F.dropout(x, p=self.dp, training=self.training)

        # Output layer with softmax activation (for multi-class classification)
        x = F.softmax(self.outputLayer(x), dim=1)

        return x



class simple_GAT(nn.Module):
    """
    A simple Graph Attention Network (GAT) using three GATv2 convolutional layers,
    batch normalization, dropout, and a final linear layer for classification.

    Attributes:
        conv1 (GATv2Conv): First graph attention convolutional layer.
        conv2 (GATv2Conv): Second graph attention convolutional layer.
        conv3 (GATv2Conv): Third graph attention convolutional layer.
        lin (nn.Linear): Final linear layer to map to output classes.
        similarity_typ (str): Type of similarity metric used (default: 'euclide').
        dp (float): Dropout probability for regularization.
        Pre_norm (bool): Whether to apply batch normalization before processing.
        BatchNorm (BatchNorm1d): Batch normalization layer for input features.
        Mean_agg (MeanAggregation): Aggregation function for node embeddings.
    """

    def __init__(self, num_of_feat, f_1, f_2, f_3,
                 dp, Pre_norm, f_final=2, edge_dim=1, similarity_typ='euclide'):
        """
        Initializes the GAT model.

        Args:
            num_of_feat (int): Number of input node features.
            f_1 (int): Number of features in the first GAT layer.
            f_2 (int): Number of features in the second GAT layer.
            f_3 (int): Number of features in the third GAT layer.
            dp (float): Dropout probability for regularization.
            Pre_norm (bool): Whether to apply batch normalization to input features.
            f_final (int): Number of output classes (default: 2).
            edge_dim (int): Dimension of edge attributes.
            similarity_typ (str): Type of similarity function to use (default: 'euclide').
        """
        super(simple_GAT, self).__init__()

        self.conv1 = GATv2Conv(num_of_feat, f_1, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(f_1, f_2, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(f_2, f_3, edge_dim=edge_dim)
        self.lin = Linear(f_3, f_final)
        self.similarity_typ = similarity_typ

        self.dp = dp
        self.Pre_norm = Pre_norm
        self.BatchNorm = BatchNorm(num_of_feat)  # Batch normalization for input features
        self.Mean_agg = MeanAggregation()  # Aggregates node embeddings into a graph representation

    def forward(self, node_list, edge_list, edge_att=None):
        """
        Forward pass for processing multiple graph samples.

        Args:
            node_list (list[torch.Tensor]): List of node feature tensors, each corresponding to a graph.
            edge_list (list[torch.Tensor]): List of edge index tensors defining graph connectivity.
            edge_att (list[torch.Tensor], optional): List of edge attribute tensors.

        Returns:
            list[torch.Tensor]: List of softmax predictions for each input graph.
        """
        prediction = []
        sample_number = len(node_list)  # Number of graphs in the batch

        for idx in range(sample_number):
            x = node_list[idx].float()  # Convert node features to float
            edge_attr = edge_att[idx].float()  # Convert edge attributes to float
            edge_index = edge_list[idx].long()  # Convert edge indices to long tensor

            if self.Pre_norm:
                x = self.BatchNorm(x)  # Apply batch normalization if enabled

            # Pass through the three GAT layers with ReLU activations and dropout
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dp, training=self.training)

            x = self.conv2(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dp, training=self.training)

            x = self.conv3(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dp, training=self.training)

            x = self.Mean_agg(x, dim=0)  # Aggregate node embeddings into a graph representation
            x = self.lin(x)  # Final linear layer
            prediction.append(F.softmax(x, dim=1))  # Softmax activation for classification

        return prediction

class simple_GNN(nn.Module):
    """
    A simple Graph Neural Network (GNN) using three GCNConv layers, batch normalization,
    dropout, and a final linear layer for classification.

    Attributes:
        conv1 (GCNConv): First graph convolutional layer.
        conv2 (GCNConv): Second graph convolutional layer.
        conv3 (GCNConv): Third graph convolutional layer.
        lin (nn.Linear): Final linear layer to map to output classes.
        similarity_typ (str): Type of similarity metric used (default: 'euclide').
        dp (float): Dropout probability for regularization.
        Pre_norm (bool): Whether to apply batch normalization before processing.
        BatchNorm (BatchNorm1d): Batch normalization layer for input features.
        Mean_agg (MeanAggregation): Aggregation function for node embeddings.
    """

    def __init__(self, num_of_feat, f_1, f_2, f_3,
                 dp, Pre_norm, f_final=2, similarity_typ='euclide'):
        """
        Initializes the GNN model.

        Args:
            num_of_feat (int): Number of input node features.
            f_1 (int): Number of features in the first GCN layer.
            f_2 (int): Number of features in the second GCN layer.
            f_3 (int): Number of features in the third GCN layer.
            dp (float): Dropout probability for regularization.
            Pre_norm (bool): Whether to apply batch normalization to input features.
            f_final (int): Number of output classes (default: 2).
            similarity_typ (str): Type of similarity function to use (default: 'euclide').
        """
        super(simple_GNN, self).__init__()

        self.conv1 = GCNConv(num_of_feat, f_1)
        self.conv2 = GCNConv(f_1, f_2)
        self.conv3 = GCNConv(f_2, f_3)

        self.lin = Linear(f_3, f_final)
        self.similarity_typ = similarity_typ

        self.dp = dp
        self.Pre_norm = Pre_norm
        self.BatchNorm = BatchNorm(num_of_feat)  # Batch normalization for input features
        self.Mean_agg = MeanAggregation()  # Aggregates node embeddings into a graph representation

    def forward(self, node_list, edge_list):
        """
        Forward pass for processing multiple graph samples.

        Args:
            node_list (list[torch.Tensor]): List of node feature tensors, each corresponding to a graph.
            edge_list (list[torch.Tensor]): List of edge index tensors defining graph connectivity.

        Returns:
            list[torch.Tensor]: List of softmax predictions for each input graph.
        """
        prediction = []
        sample_number = len(node_list)  # Number of graphs in the batch

        for idx in range(sample_number):
            x = node_list[idx].float()  # Convert node features to float
            edge_index = edge_list[idx].long()  # Convert edge indices to long tensor

            if self.Pre_norm:
                x = self.BatchNorm(x)  # Apply batch normalization if enabled

            # Pass through the three GCN layers with ReLU activations and dropout
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dp, training=self.training)

            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dp, training=self.training)

            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dp, training=self.training)

            x = self.Mean_agg(x, dim=0)  # Aggregate node embeddings into a graph representation
            x = self.lin(x)  # Final linear layer
            prediction.append(F.softmax(x, dim=1))  # Softmax activation for classification

        return prediction