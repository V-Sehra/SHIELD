#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import torch
from torch_geometric.data import Data
import random as rnd
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from typing import Optional, List, Union
from pathlib import Path, PosixPath
import random


def assign_label_from_distribution(labels_in_graph: pd.Series, node_prob: bool = False) -> str:
    '''

    labels_in_graph(pd.Series): labels_in_graph:
    node_prob(bool):
    :return: str with label type
    '''
    labels = labels_in_graph.index.tolist()

    if node_prob:
        probs = (labels_in_graph / labels_in_graph.sum()).tolist()
    else:
        probs = [0.5, 0.5]

    return random.choices(labels, weights=probs, k=1)[0]


def bool_passer(argument):
    if argument == 'True' or argument == 'true' or argument == '1' or argument == 1 or argument == True:
        value = True
    else:
        value = False
    return (value)


def create_graph_and_save(vornoi_id: int, radius_neibourhood: float,
                          whole_data: pd.DataFrame, voronoi_list: List, sub_sample: str,
                          requiremets_dict: dict, save_path_folder: Union[str, PosixPath],
                          repeat_id: int, skip_existing: bool = False,
                          noisy_labeling: bool = False, node_prob: bool = False):
    """
    Creates a graph from spatial data and saves it as a PyTorch geometric Data object.

    Args:
        vornoi_id (int): ID of the Voronoi region.
        radius_neibourhood (float): Radius for neighborhood search.
        whole_data (DataFrame): The entire dataset.
        voronoi_list (list): List of Voronoi cell indices.
        sub_sample (str): Sample identifier.
        requiremets_dict (dict): Dictionary containing dataset requirements.
        save_path_folder (str): Directory path where graphs are saved.
        repeat_id (int): Counter for repeated samples.
        noisy_labeling (bool): If True, applies noisy labeling to the graph to check for robustness.
        node_prob (bool): If True, uses node probabilities for label assignment.

    Returns:
        None
    """

    tissue_dict = requiremets_dict['label_dict']
    # List of evaluation column names.
    eval_col_name = requiremets_dict['eval_columns']
    # Column names for gene expression features.
    gene_col_name = requiremets_dict['markers']

    cosine = torch.nn.CosineSimilarity(dim=1)

    # Extract data for the current Voronoi region
    graph_data = whole_data.iloc[voronoi_list[vornoi_id]].copy()

    # Determine the most frequent tissue type in this region
    count_tissue_type = graph_data[requiremets_dict['label_column']].value_counts()
    if noisy_labeling and (len(count_tissue_type) > 1):
        dominating_tissue_type = assign_label_from_distribution(labels_in_graph=count_tissue_type,
                                                                node_prob=node_prob)
    else:
        dominating_tissue_type = count_tissue_type.idxmax()

    file_name = f'graph_subSample_{sub_sample}_{dominating_tissue_type}_{(len(voronoi_list) * repeat_id) + vornoi_id}.pt'

    if skip_existing:
        if Path(f'{save_path_folder}', file_name).exists():
            return

    # Convert gene expression features into a tensor
    node_data = torch.tensor(graph_data[gene_col_name].to_numpy()).float()

    # Extract spatial coordinates (X, Y)
    coord = graph_data[[requiremets_dict['X_col_name'], requiremets_dict['Y_col_name']]]

    # Compute the edge index using a utility function
    edge_mat = get_edge_index(mat=coord, dist_bool=True, radius=radius_neibourhood)

    # Compute cosine similarity between connected nodes
    plate_cosine_sim = cosine(node_data[edge_mat[0][0]], node_data[edge_mat[0][1]]).cpu()

    # Create a PyTorch Geometric data object
    data = Data(
        x=node_data,
        edge_index_plate=torch.tensor(edge_mat[0]).long(),
        plate_euc=torch.tensor(1 / edge_mat[1]),  # Inverse of Euclidean distance
        plate_cosine_sim=plate_cosine_sim,
        fold_id=graph_data[requiremets_dict['validation_split_column']].unique(),
        orginal_cord=torch.tensor(coord.to_numpy()),
        eval=graph_data[eval_col_name].to_numpy(dtype=np.dtype('str')),
        eval_col_names=eval_col_name,
        sub_sample=sub_sample,
        y=torch.tensor(tissue_dict[dominating_tissue_type]).flatten(),
        y_true=count_tissue_type.idxmax()
    ).cpu()

    # Save the processed graph data
    # torch.save(data, Path(f'{save_path_folder}', file_name))

    return


def fill_missing_row_and_col_withNaN(data_frame: pd.DataFrame, cell_types_names: np.array) -> pd.DataFrame:
    """
    This function fills missing rows and columns in a given DataFrame with NaN values.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame to be processed.
    cell_types_names (np.array): An array of cell type names.

    Returns:
    pd.DataFrame: The processed DataFrame with missing rows and columns filled with NaN values.

    """

    # Identify columns in cell_types_names that are not in the DataFrame's columns
    missing_cols = cell_types_names[~np.isin(cell_types_names, data_frame.columns)]
    # Fill missing columns with NaN values
    data_frame[missing_cols] = np.full((len(data_frame), len(missing_cols)), np.nan)

    # Identify rows in cell_types_names that are not in the DataFrame's index
    missing_rows = cell_types_names[~np.isin(cell_types_names, data_frame.index)]

    # Concatenate the DataFrame with a new DataFrame that contains the missing rows filled with NaN values
    data_frame = pd.concat([data_frame,
                            pd.DataFrame(np.full((len(missing_rows), data_frame.shape[1]), np.nan),
                                         columns=data_frame.columns,
                                         index=cell_types_names[~np.isin(cell_types_names, data_frame.index)])])

    # Sort the DataFrame by index and columns
    data_frame = data_frame.sort_index()[sorted(data_frame.columns)]

    return data_frame


def get_edge_index(mat: np.array, dist_bool: bool = True, radius: float = 265):
    """
    Computes the edge index for a graph based on spatial proximity using Nearest Neighbors.

    Args:
        mat (np.ndarray): A matrix (NxD) where each row represents a point in D-dimensional space.
        dist_bool (bool, optional): Whether to return edge distances along with edge indices. Default is True.
        radius (float, optional): The radius within which points are considered neighbors. Default is 265.

    Returns:
        np.ndarray: An array of shape (2, E) representing source and destination node indices.
        np.ndarray (optional): An array of distances corresponding to the edges if dist_bool is True.
    """

    # Fit a NearestNeighbors model to find neighbors within the given radius
    neigh = NearestNeighbors(radius=radius).fit(mat)

    # Get the distances and indices of neighbors for each point
    neigh_distance, neighours_per_cell = neigh.radius_neighbors(mat)

    # Compute the number of connections each cell (node) has
    total_number_connections = [len(connections) for connections in neighours_per_cell]

    # Create the source node indices based on the number of connections each node has
    edge_src = np.concatenate([
        np.repeat(idx, total_number_connections[idx]) for idx in range(len(total_number_connections))
    ])

    # Flatten the neighbor indices to get the destination nodes
    edge_dest = np.concatenate(neighours_per_cell)

    # Flatten the distances array
    distances = np.concatenate(neigh_distance)

    # Identify and remove self-loops (where source and destination are the same)
    remove_self_node = distances == 0
    edge_src = np.delete(edge_src, remove_self_node)
    edge_dest = np.delete(edge_dest, remove_self_node)

    # Construct the edge index array
    edge = np.array([edge_src, edge_dest])

    # Return edge indices with or without distances based on dist_bool
    if dist_bool:
        return edge, np.delete(distances, remove_self_node)
    else:
        return edge


def get_median_with_threshold(series: pd.Series, threshold: float) -> Optional[pd.Series]:
    if series.count() >= threshold:
        return series.median()
    else:
        return np.nan


def get_voronoi_id(data_set: DataFrame,
                   requiremets_dict: dict,
                   anker_cell: DataFrame,
                   fussy_limit: Optional[float] = None,
                   centroid_bool: bool = False) -> np.array:
    """
    Function to assign each data point to a Voronoi cell.

    Parameters:
    data_set (DataFrame): DataFrame containing the data points.
    requiremets_dict (dict): Dictionary containing dataset requirements.
    anker_cell (DataFrame): DataFrame containing the anchor points for Voronoi cells.
    boarder_number (int): Number of nearest neighbors to consider for each data point.
    fussy_limit (float): Threshold for fuzzy assignment of data points to Voronoi cells.
    centroid_bool (bool): If True, use the centroid of Voronoi cells for assignment.

    Returns:
    ndarray: List of Voronoi cells with assigned data points or array of nearest anchor indices.
    """
    boarder_number = requiremets_dict['voro_neighbours']
    x_col_name = requiremets_dict['X_col_name']
    y_col_name = requiremets_dict['Y_col_name']

    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(anker_cell[[x_col_name, y_col_name]])

    # Find the closest anchor for each point in 'data_set'
    dist, indices = tree.query(data_set[[x_col_name, y_col_name]], k=boarder_number)

    # If 'fussy_limit' is specified, adjust the assignment of data points to Voronoi cells
    if fussy_limit is not None:

        # If 'centroid_bool' is True, use the centroid of Voronoi cells for assignment
        if centroid_bool:
            copy_data = data_set.copy()
            copy_data['voronoi_id'] = indices[:, 0]
            centroid_data = copy_data.groupby('voronoi_id')[[x_col_name, y_col_name]].mean().reset_index()
            tree_centroid = cKDTree(centroid_data[[x_col_name, y_col_name]])
            dist_centroid, _ = tree_centroid.query(data_set[[x_col_name, y_col_name]], k=boarder_number)
            proximity_to_border = [dist_centroid[:, 0] / dist_centroid[:, i] for i in range(0, boarder_number)]
        else:

            proximity_to_border = [dist[:, 0] / dist[:, i] for i in range(0, boarder_number)]

        first_assignment = indices[:, 0].copy()
        voronoi_collection = []

        # Adjust the assignment of data points to Voronoi cells based on 'fussy_limit'
        for idx_voro in range(indices[:, 0].max() + 1):
            voronois_list = np.where(first_assignment == idx_voro)[0]
            for next_anker in range(1, len(proximity_to_border)):
                included_cells = proximity_to_border[next_anker] > fussy_limit
                if any(included_cells):
                    bordering_cells = (indices[:, next_anker].copy() == idx_voro) & (included_cells)
                    voronois_list = np.append(voronois_list, np.where(bordering_cells)[0])
            voronoi_collection.append(np.unique(voronois_list))

        return np.array(voronoi_collection, dtype=object)
    else:
        # If 'fussy_limit' is not specified, return the nearest anchor indices
        return indices


# Old functions
# _______________________________
def create_test_train_split(args):
    """
    This function creates training and testing sets from a raw CSV file.
    It first converts the raw CSV file into a graph CSV file.
    It then gets a list of unique patient IDs from the raw data.
    If the 'new_test_split_bool' argument is True, it shuffles the patient list and splits it into training and testing sets.
    Otherwise, it uses a predefined list of patient IDs for the testing set.
    It then creates directories for saving the training and testing sets and saves the sets as CSV files.

    Parameters:
    args (Namespace): The command-line arguments. It should include 'path_to_raw_csv', 'new_test_split_bool', and 'path_to_save_data'.

    Returns:
    None
    """
    # Convert the raw CSV file into a graph CSV file
    raw_data = turn_raw_csv_to_graph_csv(pd.read_csv(args.path_to_raw_csv))

    # Get a list of unique patient IDs from the raw data
    patent_list = raw_data['Patient'].unique()

    # If the 'new_test_split_bool' argument is True, shuffle the patient list and split it into training and testing sets
    # Otherwise, use a predefined list of patient IDs for the testing set
    if bool_passer(args.new_test_split_bool):
        rnd.shuffle(patent_list)
        train_patients = patent_list[0:int(len(patent_list) * 0.8)]
        test_patients = patent_list[int(len(patent_list) * 0.8):]

        train_csv = raw_data[raw_data['Patient'].isin(train_patients)]
        test_csv = raw_data[raw_data['Patient'].isin(test_patients)]
    else:
        test_patient_ids = ['LHCC51', 'LHCC53', 'Pat53', 'LHCC45', 'Pat52']
        train_csv = raw_data[~raw_data['Patient'].isin(test_patient_ids)]
        test_csv = raw_data[raw_data['Patient'].isin(test_patient_ids)]

    # Create directories for saving the training and testing sets
    os.system(f'mkdir -p {args.path_to_save_data}/train_set')
    os.system(f'mkdir -p {args.path_to_save_data}/test_set')

    # Save the training and testing sets as CSV files
    train_csv.to_csv(os.path.join(f'{args.path_to_save_data}', 'train_set', 'train_cells.csv'))
    test_csv.to_csv(os.path.join(f'{args.path_to_save_data}', 'test_set', 'test_cells.csv'))

    return


def replace_celltype(mat, celltype_list_to_replace):
    # Substring to find
    for substring_to_replace in celltype_list_to_replace:
        # Replace strings containing the substring
        for i, row in enumerate(mat):
            for j, word in enumerate(row):
                if substring_to_replace in word:
                    mat[i][j] = substring_to_replace

    return mat


def turn_raw_csv_to_graph_csv(raw_csv):
    """
    This function transforms a raw CSV file into a graph CSV file, which contains all nessesary coloumns for the next steps.
    It calculates the X and Y values as the average of 'XMax' and 'XMin', and 'YMax' and 'YMin', respectively.
    It then filters the columns of the DataFrame to only include those that contain certain substrings.

    Parameters:
    raw_csv (pd.DataFrame): The raw CSV file as a DataFrame.

    Returns:
    pd.DataFrame: The transformed DataFrame ready to be used for graph creation.
    """

    # Calculate the X and Y values as the average of 'XMax' and 'XMin', and 'YMax' and 'YMin', respectively
    raw_csv['X_value'] = (raw_csv['XMax'] + raw_csv['XMin']) / 2
    raw_csv['Y_value'] = (raw_csv['YMax'] + raw_csv['YMin']) / 2

    # Filter the columns of the DataFrame to only include those that contain certain substrings
    raw_csv = raw_csv.loc[:, raw_csv.columns.str.contains(
        'Tissue|Patient|X_value|Y_value|.Intensity|Class|Class0|CD45.Positive.Classification|Celltype',
        na=False)]
    return raw_csv


def turn_pixel_to_meter(pixel_radius):
    pixel_to_miliMeter_factor = 2649.291339
    mycro_meter_radius = pixel_radius * (10 ** 3 / pixel_to_miliMeter_factor)
    return round(mycro_meter_radius)


def combine_cell_types(original_array, string_list, retrun_org_matrix=False):
    """
    This function combines cell types in a given array based on a list of substrings.
    If a substring from the list is found in an element of the array, that element is replaced with the substring.
    After all replacements, the function removes duplicates from the array.

    Parameters:
    original_array (np.array): The original array containing cell type names.
    string_list (list): A list of substrings to search for in the array elements.

    Returns:
    np.array: The processed array with combined cell types and without duplicates.
    """

    for substring_to_replace in string_list:
        # Find indices where the substring occurs
        indices = np.core.defchararray.find(original_array.astype(str), substring_to_replace)

        # Replace words containing the substring with 'substring_to_replace'
        original_array[indices != -1] = substring_to_replace

        unique_elements, unique_indices, inverse_indices = np.unique(original_array, return_index=True,
                                                                     return_inverse=True)

    if retrun_org_matrix:
        return unique_elements, original_array
    # Remove duplicates and return the processed array
    return original_array
