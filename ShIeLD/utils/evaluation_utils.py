#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""

import train_utils
import model_utils

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader


from pathlib import PosixPath
from tqdm import tqdm
import pickle
from typing import Dict, Optional

import seaborn as sns
import matplotlib.pyplot as plt




def get_cell_to_cell_interaction_dict(
        requirements_dict: Dict,
        data_loader: DataLoader,
        model: torch.nn.Module,
        device: str,
        save_dict_path: Optional[PosixPath] = None
    )-> Dict:
    """
    Computes cell-to-cell interaction metrics from a trained model's predictions and attention scores.

    Parameters:
    - requirements_dict (dict): A dictionary containing experimental setup details (e.g., cell type names).
    - data_loader: A PyTorch DataLoader object for iterating over graph-based cell data.
    - model: The trained neural network model used for prediction.
    - attr_bool (bool): Whether to use attribution-based attention scores.
    - device: The computation device (e.g., "cpu" or "cuda").
    - save_dict_path (Optional[PosixPath]): Path to save/load the output dictionary. If None, results are not saved.

    Returns:
    - dict: A dictionary containing:
        - 'fals_pred': Boolean array indicating incorrect predictions.
        - 'correct_predicted': Boolean array indicating correct predictions.
        - 'original_prediction': Array of model-predicted labels.
        - 'true_labels_train': Array of ground-truth labels.
        - 'sub_sample_list': List of sample identifiers.
        - 'normed_p2p': List of normalized phenotype-to-phenotype attention scores.
        - 'raw_att_p2p': List of raw attention scores.
        - 'normalisation_factor_edge_number': List of normalization factors for edge numbers.
    """

    # Lists to store collected information
    fals_pred = []
    correct_predicted = []
    true_labels_train = []
    sub_sample_list = []

    raw_att_p2p = []
    normalisation_factor_edge_number = []
    normed_p2p = []
    prediction_model = []

    # Extract cell type information from requirements_dict
    cell_types = np.array(requirements_dict['cell_type_names'])

    # Find the index of the 'CellType' column in the evaluation data
    try:
        cell_type_eval_index = np.where(np.array(requirements_dict['eval_columns']) == 'CellType')[0][0]
    except IndexError:
        raise ValueError("Column 'CellType' not found in requirements_dict['eval_columns'].")

    # Iterate over all samples in the data loader
    for data_sample in tqdm(data_loader, desc="Processing samples"):
        # Run a prediction step using the model
        prediction, attention, output, y, sample_ids = model_utils.prediction_step(
            batch_sample=data_sample,
            model=model,
            attr_bool=requirements_dict['attr_bool'],
            device=device,
            per_patient=False
        )

        # Get predicted class labels
        _, value_pred = torch.max(output, dim=1)

        # Collect performance information
        fals_pred.extend((value_pred != y).flatten().cpu().detach().numpy())
        correct_predicted.extend((value_pred == y).flatten().cpu().detach().numpy())
        true_labels_train.extend(y.flatten().cpu().detach().numpy())
        prediction_model.extend(value_pred.flatten().cpu().detach().numpy())

        # Collect sample identifiers
        sub_sample_list.extend([sample.sub_sample for sample in data_sample])

        # Extract node-level attention scores
        node_level_attention_scores = [att_val[1].cpu().detach().numpy() for att_val in attention]

        # Extract cell type information for the given sample
        cell_type_names = [sample.eval[:, cell_type_eval_index] for sample in data_sample]

        # Compute phenotype-to-phenotype attention scores
        phenotype_attention_matrix = get_p2p_att_score(
            sample=data_sample,
            cell_phenotypes_sample=cell_type_names,
            all_phenotypes=cell_types,
            node_attention_scores=node_level_attention_scores
        )

        # Store phenotype attention metrics
        raw_att_p2p.extend(phenotype_attention_matrix[0])
        normalisation_factor_edge_number.extend(phenotype_attention_matrix[1])
        normed_p2p.extend(phenotype_attention_matrix[2])

    # Compile all collected information into a dictionary
    dict_all_info = {
        'fals_pred': np.array(fals_pred),
        'correct_predicted': np.array(correct_predicted),
        'original_prediction': np.array(prediction_model),
        'true_labels_train': np.array(true_labels_train),
        'image_list': np.array(sub_sample_list),
        'normed_p2p': normed_p2p,
        'raw_att_p2p': raw_att_p2p,
        'normalisation_factor_edge_number': normalisation_factor_edge_number,
    }

    # If a save path is provided, save the dictionary to a file
    if save_dict_path is not None:
        with open(save_dict_path, 'wb') as f:
            pickle.dump(dict_all_info, f)

    return dict_all_info


def get_hypersear_results(requirements_dict: dict):
    """
    Runs a hyperparameter search analysis by aggregating model performance metrics.

    Parameters:
    - requirements_dict (dict): A dictionary specifying training requirements.

    Returns:
    - pd.DataFrame: A DataFrame containing the mean balanced accuracy and the count of unique data splits
      for each combination of col_of_variables.
    """

    # Retrieve hyperparameter search results from training utilities.
    hyper_search_results = train_utils.get_train_results_csv(requirement_dict=requirements_dict)

    # First-level grouping: Aggregate mean balanced accuracy per col_of_interest
    model_grouped = hyper_search_results.groupby(requirements_dict['col_of_interest']).agg(
        total_acc_balanced_mean=('bal_acc_validation', 'mean')  # Compute mean validation accuracy
    ).reset_index()

    # Second-level grouping: Compute the mean accuracy and count unique splits per col_of_variables
    hyper_grouped = model_grouped.groupby(requirements_dict['col_of_variables']).agg(
        total_acc_balanced_mean=('total_acc_balanced_mean', 'mean'),  # Mean across models in col_of_variables
        count=('split_number', 'nunique')  # Count unique data splits used
    ).reset_index()

    return hyper_grouped


def get_p2p_att_score(sample, cell_phenotypes_sample, all_phenotypes, node_attention_scores):
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
    list: A list of raw attention scores for each phenotype to all other phenotypes.
    list: A list of normalization factors based on the raw count of edges between two phenotypes.
    list: A list of normalized attention scores for each phenotype to all other phenotypes.
    """

    raw_att_p2p = []
    normalisation_factor_edge_number = []
    normalised_p2p = []

    # Find the cell type (phenotype) for each cell/node
    scr_node = [cell_phenotypes_sample[sample_idx][sample[sample_idx].edge_index_plate[0]] for sample_idx in
                range(len(sample))]
    dst_node = [cell_phenotypes_sample[sample_idx][sample[sample_idx].edge_index_plate[1]] for sample_idx in
                range(len(sample))]

    for sample_idx in range(len(scr_node)):
        df_att = pd.DataFrame(data={'src': scr_node[sample_idx],
                                    'dst': dst_node[sample_idx],
                                    'att': node_attention_scores[sample_idx].flatten()})
        # Have the same DataFrame containing all phenotypes
        # If a connection is not present in the sample, fill it with NaN
        att_df = data_utils.fill_missing_row_and_col_withNaN(
            data_frame=df_att.groupby(['src', 'dst'])['att'].sum().reset_index().pivot(index='src', columns='dst',
                                                                                       values='att'),
            cell_types_names=all_phenotypes)

        # Unnormalized attention score
        raw_att_p2p.append(att_df)

        edge_df = data_utils.fill_missing_row_and_col_withNaN(
            data_frame=df_att.groupby(['src', 'dst']).count().reset_index().pivot(index='src', columns='dst',
                                                                                  values='att'),
            cell_types_names=all_phenotypes)

        # Normalize the p2p based on the raw count of edges between these two phenotypes
        normalisation_factor_edge_number.append(edge_df)
        normalised_p2p.append(att_df / edge_df)

    return raw_att_p2p, normalisation_factor_edge_number, normalised_p2p

def create_parameter_influence_plots(df: pd.DataFrame, observed_variable:str, save_path: Optional[PosixPath] =None):
    """
    Creates and displays a boxplot to visualize the influence of a specific hyperparameter
    on model accuracy.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing hyperparameter values and corresponding accuracy metrics.
    - observed_variable (str): The name of the hyperparameter to analyze.
    - save_path (str, optional): Path to save the generated plot. If None, the plot is not saved.

    Returns:
    - None (displays the plot and optionally saves it).
    """

    # Set figure size for better visualization
    plt.figure(figsize=(10, 6))

    # Filter dataset to only include rows where 'hyperparameter' matches the observed variable
    data_filtered = df[df['hyperparameter'] == observed_variable]

    # Create a boxplot showing the distribution of accuracy scores for different hyperparameter values
    sns.boxplot(x='value', y='total_acc_balanced_mean', data=data_filtered)

    # Set plot title and axis labels
    plt.title(f'Effect of {observed_variable} on Accuracy')
    plt.xlabel('Hyperparameter Value')
    plt.ylabel('Balanced Accuracy')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path)

