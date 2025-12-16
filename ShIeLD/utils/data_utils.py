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


def get_random_edges(
    original_edge_mat: np.ndarray, number_nodes: int, percent_number_cells: float = 0.1
):
    """
    Generate synthetic edge matrices with controlled connectivity and randomness.

    Parameters
    ----------
    original_edge_mat : np.ndarray
        The original edge matrix of shape (2, N), where edges are defined as pairs (source, destination).
    percent_number_cells : float, optional (default=0.1)
        Percentage (0–1) of the number of nodes to use when generating additional random edges.
    number_nodes : int
        number of nodes within the graph
    Returns
    -------
    edge_mat_same_connectivity : np.ndarray
        A new edge matrix where each node has approximately the same average degree as in the original graph.
    edge_mat_random_percentage : np.ndarray
        A new edge matrix generated randomly using the specified percentage of total nodes.
    """
    # Get unique source nodes and their degree (connectivity)
    if len(original_edge_mat[0]) > 0:
        unique, connectivity = np.unique(original_edge_mat[0], return_counts=True)

        # Generate a synthetic edge matrix with same average connectivity
        sample_number = round(np.mean(connectivity))
        scr_same_connect = np.repeat(np.arange(number_nodes), sample_number)

        # Sample exactly len(scr) destinations
        dst_same_connect = np.random.choice(number_nodes, size=len(scr_same_connect))
        edge_mat_same_connectivity = np.array([scr_same_connect, dst_same_connect])
    else:
        edge_mat_same_connectivity = np.array([[], []])

    # Generate a synthetic edge matrix with random edges based on percentage of nodes
    random_repeats = round(number_nodes * percent_number_cells)
    src_random_percentage = np.repeat(np.arange(number_nodes), random_repeats)
    dst_random_percentage = np.random.choice(
        number_nodes, size=len(src_random_percentage)
    )
    edge_mat_random_percentage = np.array(
        [src_random_percentage, dst_random_percentage]
    )

    return edge_mat_same_connectivity, edge_mat_random_percentage


def assign_label_from_distribution(
    labels_in_graph: pd.Series, node_prob: Union[str, bool] = False
) -> str:
    """

    labels_in_graph(pd.Series): labels_in_graph:
    node_prob(bool):
    :return: str with label type
    """
    labels = labels_in_graph.index.tolist()

    if node_prob == False or node_prob == "even":  # noqa: E712
        probs = [1 / len(labels_in_graph) for _ in range(len(labels_in_graph))]

        return rnd.choices(labels, weights=probs, k=1)[0]

    elif node_prob == True or node_prob == "prob":  # noqa: E712
        probs = (labels_in_graph / labels_in_graph.sum()).tolist()
        return rnd.choices(labels, weights=probs, k=1)[0]
    elif node_prob == "both":
        probs = (labels_in_graph / labels_in_graph.sum()).tolist()
        even_probs = [1 / len(labels_in_graph) for _ in range(len(labels_in_graph))]
        return rnd.choices(labels, weights=probs, k=1)[0], rnd.choices(
            labels, weights=even_probs, k=1
        )[0]


def bool_passer(argument):
    """Convert various representations of truth values to a boolean.
    Parameters
    ----------
    argument : str, int, bool
        The input value to be converted to a boolean.
        Acceptable string values are "True", "true", "1" for True,
        and "False", "false", "0" for False.
        Integer values 1 and 0 are also accepted.
        Boolean values True and False are returned as is.
    """
    if (
        argument == "True"
        or argument == "true"
        or argument == "1"
        or argument == 1
        or argument == True  # noqa: E712
    ):
        value = True
    elif (
        argument == "False"
        or argument == "false"
        or argument == "0"
        or argument == 0
        or argument == False  # noqa: E712
    ):
        value = False
    else:
        value = argument
    return value


def compute_connection_matrix(src, dst, phenotype_names, absolute_bool):
    """
    Compute a cell-cell interaction matrix between source and destination cell types.

    Parameters
    ----------
    src : array-like
        List or array of source cell phenotypes (categorical labels).
    dst : array-like
        List or array of destination cell phenotypes (categorical labels), same length as `src`.
    phenotype_names : list of str
        List of all known phenotype (cell type) names to define matrix dimensions and ordering.
    absolute_bool : bool
        If True, return absolute interaction counts.
        If False, return row-normalized percentages (i.e., conditional distributions of dst per src).

    Returns
    -------
    pd.DataFrame
        A square DataFrame where rows represent source phenotypes and columns represent destination phenotypes.
        Values are either absolute counts or row-wise percentages depending on `absolute_bool`.
        Rows and columns are aligned with the `phenotype_names` order.

    Notes
    -----
    This function is typically used to analyze neighborhood-based cell-cell interactions in spatial omics data.
    """

    # Convert inputs to numpy arrays for efficient processing
    src = np.array(src)
    dst = np.array(dst)

    # Get sorted list of unique phenotype names for consistent ordering
    phenotypes = sorted(set(phenotype_names))

    # Create a DataFrame from source-destination pairings
    df_conn = pd.DataFrame({"src": src, "dst": dst})

    # Count how often each (src, dst) pair occurs; pivot into a square matrix
    conn_counts = df_conn.groupby(["src", "dst"]).size().unstack(fill_value=0)

    # Ensure all phenotype combinations are represented, even if absent in the data
    conn_counts = conn_counts.reindex(
        index=phenotypes, columns=phenotypes, fill_value=0
    )

    if absolute_bool:
        # Return raw counts of connections
        return conn_counts
    else:
        # Normalize each row to percentage (conditional on source phenotype)
        conn_percent = (
            conn_counts.div(conn_counts.sum(axis=1).replace(0, 1), axis=0) * 100
        )
        return conn_percent


def create_graph_and_save(
    vornoi_id: int,
    radius_neibourhood: float,
    whole_data: pd.DataFrame,
    voronoi_list: List,
    sub_sample: str,
    requiremets_dict: dict,
    save_path_folder: Union[str, PosixPath],
    repeat_id: int,
    skip_existing: bool = False,
    noisy_labeling: bool = False,
    node_prob: Union[str, bool] = False,
    randomise_edges: bool = False,
    percent_number_cells: float = 0.1,
    segmentation: str = "voronoi",
):
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
        node_prob (bool,str): If True, uses node probabilities for label assignment.
                              if both then return 50/50 chance and true prob
                              if even only 50/50, if True or prob then the true dist
        randomise_edges (bool): are you interested in randomsied edges for baselining
        segmentation:str = 'voronoi'

    Returns:
        None
    """

    tissue_dict = requiremets_dict["label_dict"]
    # List of evaluation column names.
    eval_col_name = requiremets_dict["eval_columns"]
    # Column names for gene expression features.
    gene_col_name = requiremets_dict["markers"]

    cosine = torch.nn.CosineSimilarity(dim=1)

    # Extract data for the current Voronoi region
    # Extract data for the current Voronoi region
    if segmentation.lower() == "voronoi":
        graph_data = whole_data.iloc[voronoi_list[vornoi_id]].copy()
    elif segmentation.lower() == "random":
        if not isinstance(whole_data, list):
            raise ValueError(
                "if segmentation is random then the provided whole set must be a list of DataFrames"
            )

        graph_data = whole_data[vornoi_id].copy()

    # Determine the most frequent tissue type in this region
    count_tissue_type = graph_data[requiremets_dict["label_column"]].value_counts()

    file_name = f"graph_subSample_{sub_sample}_{count_tissue_type.idxmax()}_RepNo{repeat_id}_VoroID{vornoi_id}.pt"

    if skip_existing:
        if Path(f"{save_path_folder}", file_name).exists():
            print(f"Skipping existing file: {file_name}")
            return

    # Convert gene expression features into a tensor
    node_data = torch.tensor(graph_data[gene_col_name].to_numpy()).float()

    # Extract spatial coordinates (X, Y)
    coord = graph_data[[requiremets_dict["X_col_name"], requiremets_dict["Y_col_name"]]]

    # Compute the edge index using a utility function
    edge_mat = get_edge_index(mat=coord, dist_bool=True, radius=radius_neibourhood)

    # Compute cosine similarity between connected nodes
    plate_cosine_sim = cosine(
        node_data[edge_mat[0][0]], node_data[edge_mat[0][1]]
    ).cpu()

    # Create a PyTorch Geometric data object
    data = Data(
        x=node_data,
        edge_index_plate=torch.tensor(edge_mat[0]).long(),
        plate_euc=torch.tensor(1 / edge_mat[1]),  # Inverse of Euclidean distance
        plate_cosine_sim=plate_cosine_sim,
        fold_id=graph_data[requiremets_dict["validation_split_column"]].unique(),
        orginal_cord=torch.tensor(coord.to_numpy()),
        eval=graph_data[eval_col_name].to_numpy(dtype=np.dtype("str")),
        eval_col_names=eval_col_name,
        sub_sample=sub_sample,
        y=torch.tensor(tissue_dict[count_tissue_type.idxmax()]).flatten(),
    ).cpu()

    if noisy_labeling and (len(count_tissue_type) > 1):
        if node_prob == False or node_prob == "even":  # noqa: E712
            even_label = assign_label_from_distribution(
                labels_in_graph=count_tissue_type, node_prob=node_prob
            )
            data.y_noise_even = torch.tensor(tissue_dict[even_label]).flatten()

        elif node_prob == True or node_prob == "prob":  # noqa: E712
            prob_label = assign_label_from_distribution(
                labels_in_graph=count_tissue_type, node_prob=node_prob
            )
            data.y_noise_prob = torch.tensor(tissue_dict[prob_label]).flatten()

        elif node_prob == "both":
            prob_label, even_label = assign_label_from_distribution(
                labels_in_graph=count_tissue_type, node_prob=node_prob
            )
            data.y_noise_prob = torch.tensor(tissue_dict[prob_label]).flatten()
            data.y_noise_even = torch.tensor(tissue_dict[even_label]).flatten()

    if randomise_edges:
        edge_mat_same_connectivity, edge_mat_random_percentage = get_random_edges(
            original_edge_mat=edge_mat[0],
            percent_number_cells=percent_number_cells,
            number_nodes=node_data.shape[0],
        )

        data.edge_index_plate_sameCon = torch.tensor(edge_mat_same_connectivity).long()
        data.edge_index_plate_percent = torch.tensor(edge_mat_random_percentage).long()

    # Save the processed graph data
    torch.save(data, Path(f"{save_path_folder}", file_name))

    return


def reducePopulation(
    df: pd.DataFrame, columnName: str, cellTypeName: str, downsampleRatio: float = 0.3
):
    """
    This function reduces the population of a specific cell type in a DataFrame by downsampling it.
    df (pd.DataFrame): original sample
    columnName (str): which column name contrains the phenotypes
    cellTypeName (str): which cell type should be reduced
    downsampleRatio (float): the ratio of the cell type to be kept in the sample, default is 0.3 (30%)
    :return df (pd.DataFrame): DataFrame with reduced population of the specified cell type
    """
    # print("reduceing population of cell type:", cellTypeName)
    # Example filter mask
    mask = df[columnName].str.contains(cellTypeName, case=False, na=False)

    # Rows containing "Macrophages"
    cellTypeName_rows = df[mask]

    # Rows not containing "Macrophages"
    non_cellTypeName_rows = df[~mask]

    # Subsample 30% of the "Macrophages" rows
    cellTypeName_subsample = cellTypeName_rows.sample(
        frac=downsampleRatio, random_state=42
    )

    # Combine them back
    df = pd.concat([non_cellTypeName_rows, cellTypeName_subsample]).sort_index()

    return df


def fill_missing_row_and_col_withNaN(
    data_frame: pd.DataFrame, cell_types_names: np.array
) -> pd.DataFrame:
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
    data_frame = pd.concat(
        [
            data_frame,
            pd.DataFrame(
                np.full((len(missing_rows), data_frame.shape[1]), np.nan),
                columns=data_frame.columns,
                index=cell_types_names[~np.isin(cell_types_names, data_frame.index)],
            ),
        ]
    )

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
    edge_src = np.concatenate(
        [
            np.repeat(idx, total_number_connections[idx])
            for idx in range(len(total_number_connections))
        ]
    )

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


def get_median_with_threshold(
    series: pd.Series, threshold: float
) -> Optional[pd.Series]:
    if series.count() >= threshold:
        return series.median()
    else:
        return np.nan


def get_voronoi_id(
    data_set: DataFrame,
    requiremets_dict: dict,
    anker_cell: DataFrame,
    fussy_limit: Optional[float] = None,
    centroid_bool: bool = False,
    eps: float = 1e-12,
) -> np.array:
    """
    Function to assign each data point to a Voronoi cell.

    Parameters:
    data_set (DataFrame): DataFrame containing the data points.
    requiremets_dict (dict): Dictionary containing dataset requirements.
    anker_cell (DataFrame): DataFrame containing the anchor points for Voronoi cells.
    boarder_number (int): Number of nearest neighbors to consider for each data point.
    fussy_limit (float): Threshold for fuzzy assignment of data points to Voronoi cells.
    centroid_bool (bool): If True, use the centroid of Voronoi cells for assignment.
    eps (float): Add a small number to prevent div(0)
    Returns:
    ndarray: List of Voronoi cells with assigned data points or array of nearest anchor indices.
    """
    boarder_number = requiremets_dict["voro_neighbours"]
    x_col_name = requiremets_dict["X_col_name"]
    y_col_name = requiremets_dict["Y_col_name"]

    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(anker_cell[[x_col_name, y_col_name]])

    # Find the closest anchor for each point in 'data_set'
    dist, indices = tree.query(data_set[[x_col_name, y_col_name]], k=boarder_number)

    # If 'fussy_limit' is specified, adjust the assignment of data points to Voronoi cells
    if fussy_limit is not None:
        # If 'centroid_bool' is True, use the centroid of Voronoi cells for assignment
        if centroid_bool:
            copy_data = data_set.copy()
            copy_data["voronoi_id"] = indices[:, 0]
            centroid_data = (
                copy_data.groupby("voronoi_id")[[x_col_name, y_col_name]]
                .mean()
                .reset_index()
            )
            tree_centroid = cKDTree(centroid_data[[x_col_name, y_col_name]])
            dist_centroid, _ = tree_centroid.query(
                data_set[[x_col_name, y_col_name]], k=boarder_number
            )
            num = np.nan_to_num(
                dist_centroid[:, 0].astype(float), nan=0.0, posinf=0.0, neginf=0.0
            )
            proximity_to_border = [
                num
                / np.maximum(
                    np.nan_to_num(
                        dist_centroid[:, i].astype(float),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ),
                    eps,
                )
                for i in range(boarder_number)
            ]

        else:
            num = np.nan_to_num(
                dist[:, 0].astype(float), nan=0.0, posinf=0.0, neginf=0.0
            )

            proximity_to_border = [
                num
                / np.maximum(
                    np.nan_to_num(
                        dist[:, i].astype(float), nan=0.0, posinf=0.0, neginf=0.0
                    ),
                    eps,
                )
                for i in range(boarder_number)
            ]

        first_assignment = indices[:, 0].copy()
        voronoi_collection = []

        # Adjust the assignment of data points to Voronoi cells based on 'fussy_limit'
        for idx_voro in range(indices[:, 0].max() + 1):
            voronois_list = np.where(first_assignment == idx_voro)[0]
            for next_anker in range(1, len(proximity_to_border)):
                included_cells = proximity_to_border[next_anker] > fussy_limit
                if any(included_cells):
                    bordering_cells = (indices[:, next_anker].copy() == idx_voro) & (
                        included_cells
                    )
                    voronois_list = np.append(
                        voronois_list, np.where(bordering_cells)[0]
                    )
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
    patent_list = raw_data["Patient"].unique()

    # If the 'new_test_split_bool' argument is True, shuffle the patient list and split it into training and testing sets
    # Otherwise, use a predefined list of patient IDs for the testing set
    if bool_passer(args.new_test_split_bool):
        rnd.shuffle(patent_list)
        train_patients = patent_list[0 : int(len(patent_list) * 0.8)]
        test_patients = patent_list[int(len(patent_list) * 0.8) :]

        train_csv = raw_data[raw_data["Patient"].isin(train_patients)]
        test_csv = raw_data[raw_data["Patient"].isin(test_patients)]
    else:
        test_patient_ids = ["LHCC51", "LHCC53", "Pat53", "LHCC45", "Pat52"]
        train_csv = raw_data[~raw_data["Patient"].isin(test_patient_ids)]
        test_csv = raw_data[raw_data["Patient"].isin(test_patient_ids)]

    # Create directories for saving the training and testing sets
    os.system(f"mkdir -p {args.path_to_save_data}/train_set")
    os.system(f"mkdir -p {args.path_to_save_data}/test_set")

    # Save the training and testing sets as CSV files
    train_csv.to_csv(
        os.path.join(f"{args.path_to_save_data}", "train_set", "train_cells.csv")
    )
    test_csv.to_csv(
        os.path.join(f"{args.path_to_save_data}", "test_set", "test_cells.csv")
    )

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
    raw_csv["X_value"] = (raw_csv["XMax"] + raw_csv["XMin"]) / 2
    raw_csv["Y_value"] = (raw_csv["YMax"] + raw_csv["YMin"]) / 2

    # Filter the columns of the DataFrame to only include those that contain certain substrings
    raw_csv = raw_csv.loc[
        :,
        raw_csv.columns.str.contains(
            "Tissue|Patient|X_value|Y_value|.Intensity|Class|Class0|CD45.Positive.Classification|Celltype",
            na=False,
        ),
    ]
    return raw_csv


def turn_pixel_to_meter(pixel_radius):
    pixel_to_miliMeter_factor = 2649.291339
    mycro_meter_radius = pixel_radius * (10**3 / pixel_to_miliMeter_factor)
    return round(mycro_meter_radius)


def combine_cell_types(original_array, string_list, retrun_adj_matrix=False):
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
    adjusted_array = original_array.copy()

    for substring_to_replace in string_list:
        # Find indices where the substring occurs
        indices = np.core.defchararray.find(
            adjusted_array.astype(str), substring_to_replace
        )

        # Replace words containing the substring with 'substring_to_replace'
        adjusted_array[indices != -1] = substring_to_replace

    unique_elements, unique_indices, inverse_indices = np.unique(
        adjusted_array, return_index=True, return_inverse=True
    )

    if retrun_adj_matrix:
        return unique_elements, adjusted_array
    # Remove duplicates and return the processed array
    return unique_elements
