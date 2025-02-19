#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:18:15 2024

@author: Vivek
"""
import numpy as np
import pickle
import pandas as pd
from typing import List, Tuple,Optional

from sklearn.neighbors import NearestNeighbors

import torch
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import GATv2Conv,GCNConv
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

def calc_edge_mat(mat: np.array, dist_bool: bool = True, radius: float = 50) -> np.array:
    """
    This function calculates the edge matrix for a given matrix of data points.
    It uses the NearestNeighbors algorithm from sklearn to find the nearest neighbors within a given radius.
    It then constructs the edge matrix by creating source and destination nodes based on the number of connections each cell has.
    Self-connections are removed from the edge matrix.

    Parameters:
    mat (np.array): The input matrix of data points.
    dist_bool (bool): If True, the function also returns the distances between the nodes. Default is True.
    radius (int): The radius within which to find the nearest neighbors. Default is 265.

    Returns:
    np.array: The edge matrix. If dist_bool is True, it also returns the distances between the nodes.

    """

    neigh = NearestNeighbors(radius=radius).fit(mat)
    neigh_dist, neigh_ind = neigh.radius_neighbors(mat)

    # check the number of conections each cell has
    total_number_conections = [len(conections) for conections in neigh_ind]

    # create the source node indices based on the number of conections a cell has
    edge_scr = np.concatenate([np.repeat(idx, total_number_conections[idx])
                               for idx in range(len(total_number_conections))])

    # concat all idx calculated by sklearn as the dest node
    edge_dest = np.concatenate(neigh_ind)

    # remove the self connection
    remove_self_node = np.zeros(len(total_number_conections), dtype=np.dtype('int'))

    idx_counter = 0
    for idx in range(len(total_number_conections)):
        remove_self_node[idx] = int(idx_counter)
        idx_counter += total_number_conections[idx]

    edge_scr = np.delete(edge_scr, remove_self_node)
    edge_dest = np.delete(edge_dest, remove_self_node)

    edge = np.array([edge_scr, edge_dest])

    if dist_bool:
        return edge, np.delete(np.concatenate(neigh_dist), remove_self_node)
    else:
        return edge


def early_stopping(loss_epoch: list, patience: int = 15) -> bool:
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

    if len(loss_epoch) > patience:
        if (loss_epoch[-2] - loss_epoch[-1]) < 0.001:
            return True
        else:
            return False
    else:
        return False


def get_p2p_att_score(sample: list, cell_phenotypes_sample: np.array, all_phenotypes: np.array, node_attention_scores: list) -> Tuple[List, List, List]:
    """
    This function calculates the attention score for each cell phenotype to all other phenotypes.
    It uses the end attention score p2p (phenotype to phenotype).
    It also uses the word "node" instead of cell.

    Parameters:
    sample (list): A list of data samples.
    cell_phenotypes_sample (np.array): An array of cell phenotype names for each sample.
    all_phenotypes (np.array): An array of all phenotype names.
    node_attention_scores (list): A list of attention scores for each node.

    Returns:
    raw_att_p2p (list): A list of raw attention scores for each phenotype to all other phenotypes.
    normalisation_factor_edge_number(list): A list of normalization factors based on the raw count of edges between two phenotypes.
    normalised_p2p(list): A list of normalized attention scores for each phenotype to all other phenotypes.

    """

    raw_att_p2p = []
    normalisation_factor_edge_number = []
    normalised_p2p = []

    # Find the cell type (phenotype) for each cell/node
    scr_node = [cell_phenotypes_sample[sample_idx][sample[sample_idx].edge_index_plate[0]] for sample_idx in range(len(sample))]
    dst_node = [cell_phenotypes_sample[sample_idx][sample[sample_idx].edge_index_plate[1]] for sample_idx in range(len(sample))]

    for sample_idx in range(len(scr_node)):
        df_att = pd.DataFrame(data={'src': scr_node[sample_idx],
                                    'dst': dst_node[sample_idx],
                                    'att': node_attention_scores[sample_idx].flatten()})
        # Have the same DataFrame containing all phenotypes
        # If a connection is not present in the sample, fill it with NaN
        att_df = data_utils.fill_missing_row_and_col_withNaN(
            data_frame=df_att.groupby(['src', 'dst'])['att'].sum().reset_index().pivot(index='src', columns='dst', values='att'),
            cell_types_names=all_phenotypes)
        # Unnormalized attention score
        raw_att_p2p.append(att_df)

        edge_df = data_utils.fill_missing_row_and_col_withNaN(
            data_frame=df_att.groupby(['src', 'dst']).count().reset_index().pivot(index='src', columns='dst', values='att'),
            cell_types_names=all_phenotypes)

        # Normalize the p2p based on the raw count of edges between these two phenotypes
        normalisation_factor_edge_number.append(edge_df)
        normalised_p2p.append(att_df / edge_df)

    return raw_att_p2p, normalisation_factor_edge_number, normalised_p2p


def initiaize_loss(path: str, device: str, tissue_dict: Optional[dict]=None) -> nn.CrossEntropyLoss:
    """
    This function initializes the loss function as weighted loss.
    It calculates the class weights based on the number of graphs for each tissue type in the training data.
    The class weights are used to initialize the CrossEntropyLoss function from PyTorch.

    Parameters:
    path (str): The path to the file containing the training file names.
    device (str): The device to which the tensor of class weights should be moved.
    tissue_dict (dict): A dictionary mapping tissue types to integers. If None, a default dictionary is used. Default is None.

    Returns:
    nn.CrossEntropyLoss: The initialized loss function with class weights.
    """

    if tissue_dict is None:
        tissue_dict = {'normalLiver': 0,
                       'core': 1,
                       'rim': 2}
    class_weights = []
    # Collect all file names and prevent non-graphs to count
    with open(path, 'rb') as f:
        all_train_file_names = pickle.load(f)

    for origin in tissue_dict.keys():
        number_tissue_graphs = len([file_name for file_name in all_train_file_names
                                    if file_name.find(origin) != -1])
        class_weights.append(1 - (number_tissue_graphs / len(all_train_file_names)))

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    return criterion


def remove_zero_distances(edge: np.array, dist=np.array, limit: float = 0, return_dist: bool = False):
    """
    This function removes edges with distances less than or equal to a given limit from a given edge matrix.
    It also optionally removes the corresponding distances from a given distance array.

    Parameters:
    edge (np.array): The input edge matrix.
    dist (np.array): The array of distances corresponding to the edges in the edge matrix.
    limit (float): The distance limit. Edges with distances less than or equal to this limit are removed. Default is 0.
    return_dist (bool): If True, the function also returns the updated distance array. Default is False.

    Returns:
    np.array: The updated edge matrix. If return_dist is True, it also returns the updated distance array.
    """

    zero_index = np.where(dist <= limit)[0]

    if len(zero_index) != 0:
        edge = np.delete(edge, zero_index, 1)

    if return_dist:
        dist = np.delete(dist, zero_index, 0)
        return (edge, dist)

    else:
        return edge



def turn_data_list_into_batch(data_sample: list, device: str, attr_bool: bool = False, id_bool: bool = False):
    """
    This function processes a list of data samples by moving each attribute of the samples to a specified device.
    It returns lists of x attributes, edge indices, Euclidean distances, and IDs of the samples.

    Parameters:
    data_sample (list): A list of data samples. Each sample is an object with attributes 'x', 'edge_index_plate', 'plate_euc', and 'ids'.
    device (str): The device to which the attributes of the samples should be moved.
    attr_bool (bool): If True, the function also processes and returns the Euclidean distances of the samples. Default is False.
    id_bool (bool): If True, the function also processes and returns the IDs of the samples. Default is False.

    Returns:
    tuple: A tuple containing lists of
    x attributes,
    edge indices,
    Euclidean distances (if attr_bool is True),
    IDs (if id_bool is True) of the samples.
    """

    sample_x = [sample.x.to(device) for sample in data_sample]
    sample_edge = [sample.edge_index_plate.to(device) for sample in data_sample]
    sample_att = [sample.plate_euc.to(device) for sample in data_sample] if attr_bool else None
    ids_list = [sample.ids for sample in data_sample] if id_bool else None

    return sample_x, sample_edge, sample_att, ids_list

def train_loop(data_loader: DataLoader, model: nn.Module, optimizer: Optimizer, loss_fkt: nn.CrossEntropyLoss,
               attr_bool: bool = False, device: str ='cuda', loss_batch: list=[])-> Tuple[nn.Module, list]:

    """
    This function performs a training loop for a given model and data loader.
    It iterates over the data loader, processes each sample, makes predictions using the model, calculates the loss, and updates the model parameters.
    It also optionally processes and uses the Euclidean distances of the samples.

    Parameters:
    data_loader (DataLoader): The data loader to iterate over.
    model (nn.Module): The model to train.
    optimizer (Optimizer): The optimizer to use for updating the model parameters.
    loss_fkt (nn.Module): The loss function to use for calculating the loss.
    attr_bool (bool): If True, the function also processes and uses the Euclidean distances of the samples. Default is False.
    device (str): The device to which the samples should be moved. Default is 'cuda'.
    loss_batch (list): A list to which the loss for each batch should be appended. Default is an empty list.

    Returns:
    nn.Module: The trained model.
    list: The list of losses for each batch.
    """

    for train_sample in data_loader:
        optimizer.zero_grad()
        sample_x, sample_edge, sample_att, ids_list = turn_data_list_into_batch(data_sample=train_sample,
                                                                                device=device,
                                                                                attr_bool=attr_bool,
                                                                                id_bool=False)
        if attr_bool:
            prediction, attention = model(sample_x, sample_edge, sample_att)
        else:
            prediction, attention = model(sample_x, sample_edge)

        output = torch.vstack(prediction)
        y = torch.tensor([sample.y for sample in train_sample]).to(output.device)

        loss = loss_fkt(output, y)

        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())

    return model, loss_batch




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