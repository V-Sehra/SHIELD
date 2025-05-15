#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""

from . import train_utils
from . import model_utils
from . import data_utils

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from pathlib import PosixPath
from tqdm import tqdm
import pickle
from typing import Dict, Optional,Tuple
from itertools import compress
from scipy.stats import mannwhitneyu

import seaborn as sns
import matplotlib.pyplot as plt


def break_title(title, max_len=10):
    if len(title) <= max_len:
        return title
    # Find the last space before the max_len limit
    break_idx = title.rfind(' ', 0, max_len)
    if break_idx == -1:
        # No space found before max_len, try finding the next one
        break_idx = title.find(' ', max_len)
    if break_idx != -1:
        return title[:break_idx] + '\n' + title[break_idx + 1:]
    return title  # No space found, return as is


def get_best_config_dict(hyper_search_results ,requirements_dict):
    """
    Extracts the best configuration from hyperparameter search results.

    Parameters:
    - hyper_search_results (pd.DataFrame): DataFrame containing hyperparameter search results.
    - requirements_dict (dict): Dictionary containing requirements for the model.

    Returns:
    - dict: The best configuration dictionary.
    """
    must_have_columns = ['layer_1', 'input_layer', 'droupout_rate', 'output_layer',
                         'attr_bool', 'anker_value', 'radius_distance', 'fussy_limit']

    # Sort the results by balanced accuracy and select the top entry
    best_config_dict = {}
    for column in hyper_search_results.columns:
        best_config_dict[column] = hyper_search_results[column].iloc[0]

    # Create a dictionary with the best configuration
    missing_keys = [key for key in must_have_columns if key not in best_config_dict]
    for key in missing_keys:
        if type(requirements_dict[key]) is list:
            best_config_dict[key] = requirements_dict[key][0]
        else:
            best_config_dict[key] = requirements_dict[key]

    return best_config_dict



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
    true_labels = []
    sub_sample_list = []

    raw_att_p2p = []
    normalisation_factor_edge_number = []
    normed_p2p = []
    prediction_model = []

    # Extract cell type information from requirements_dict
    cell_types = np.array(requirements_dict['cell_type_names'])

    # Find the index of the 'CellType' column in the evaluation data
    try:
        cell_type_eval_index = np.where(
            np.char.lower(np.array(requirements_dict['eval_columns'])) == 'celltype'.lower()
        )[0][0]
    except IndexError:
        raise ValueError(f"Column 'CellType' not found in {requirements_dict['eval_columns']}.")

    # Iterate over all samples in the data loader
    for data_sample in tqdm(data_loader, desc="Processing samples"):
        # Run a prediction step using the model
        prediction, attention, output, y, sample_ids = model_utils.prediction_step(
            batch_sample=data_sample,
            model=model,
            device=device,
            per_patient=False
        )

        # Get predicted class labels
        _, value_pred = torch.max(output, dim=1)

        # Collect performance information
        fals_pred.extend((value_pred != y).flatten().cpu().detach().numpy())
        correct_predicted.extend((value_pred == y).flatten().cpu().detach().numpy())
        true_labels.extend(y.flatten().cpu().detach().numpy())
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
        'false_pred': np.array(fals_pred),
        'correct_predicted': np.array(correct_predicted),
        'original_prediction': np.array(prediction_model),
        'true_labels': np.array(true_labels),
        'sub_sample_list': np.array(sub_sample_list),
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
    hyper_search_results,csv_file_path = train_utils.get_train_results_csv(requirement_dict=requirements_dict)

    # First-level grouping: Aggregate mean balanced accuracy per col_of_interest
    model_grouped = hyper_search_results.groupby(requirements_dict['col_of_interest']).agg(
        total_acc_balanced_mean=('bal_acc_validation', 'mean')  # Compute mean validation accuracy
    ).reset_index()

    col_of_variables = []
    for column in requirements_dict['col_of_variables']:
        if len(model_grouped[column].unique()) > 1:
            col_of_variables.append(column)
        else:
            print(
                f'will not report evaluation of {column} as the HyperSearch only invertigated one value {model_grouped[column].unique()}')

    # Second-level grouping: Compute the mean accuracy and count unique splits per col_of_variables
    hyper_grouped = model_grouped.groupby(col_of_variables).agg(
        total_acc_balanced_mean=('total_acc_balanced_mean', 'mean'),  # Mean across models in col_of_variables
        count=('split_number', 'nunique')  # Count unique data splits used
    ).reset_index()

    return hyper_grouped



def get_interaction_DataFrame(
        tissue_id: str,
        interaction_dict: Dict[str, any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts and processes interaction data for a given tissue.

    Args:
        tissue_id (str): The tissue identifier for which interaction data is extracted.
        interaction_dict (dict): Dictionary containing:
            - 'true_labels': Array indicating tissue IDs.
            - 'normed_p2p': List of normalized point-to-point interaction DataFrames.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Raw interaction DataFrame for the specified tissue.
            - Aggregated mean interaction DataFrame with a threshold filter.
    """

    # Select samples corresponding to the given tissue_id
    sample_mask = interaction_dict['true_labels'] == tissue_id
    selected_dfs = list(compress(interaction_dict['normed_p2p'], sample_mask))

    # Define threshold for filtering interactions
    threshold = len(selected_dfs) * 0.01

    # Combine selected interaction DataFrames
    interaction_df = pd.concat(selected_dfs)

    # Compute median values with threshold filtering and sort columns
    mean_interaction_df = (
        interaction_df
        .groupby(interaction_df.index)
        .agg(lambda x: data_utils.get_median_with_threshold(x, threshold))
        .sort_index()[sorted(interaction_df.columns)]
    )

    return interaction_df, mean_interaction_df


def get_top_interaction_per_celltype(interaction_limit:int,
                                     all_interaction_mean_df:pd.DataFrame,
                                     all_interaction_df:pd.DataFrame) -> Dict:
    """
    Identifies the top interactions for each cell type based on interaction strength.

    Args:
        interaction_limit (int): The maximum number of top interactions to retrieve per cell type.
        all_interaction_mean_df (DataFrame): DataFrame containing the mean interaction strengths between cell types.
        all_interaction_df (DataFrame): DataFrame containing detailed interaction values between cell types.

    Returns:
        dict: A dictionary where keys are source cell types, and values are lists of tuples
              (destination cell type, interaction value) representing the strongest interactions.
    """

    top_connections = {}
    cell_types = all_interaction_mean_df.columns

    # Loop over each cell type
    for src_cell in cell_types:
        # Get the indices of the top connections for the source cell type
        top_dst_cells = (
            all_interaction_mean_df[src_cell]
            [~np.isnan(all_interaction_mean_df[src_cell])]  # Remove NaN values
            .sort_values(ascending=False)[:interaction_limit]  # Select top interactions
            .index
        )

        values = []

        # Loop over each top connection and store its value
        for dst_cell in top_dst_cells:
            values.append((dst_cell, all_interaction_df[src_cell][dst_cell]))

        # Store the strongest connections in the dictionary
        top_connections[src_cell] = values

    return top_connections


def get_p2p_att_score(sample:list, cell_phenotypes_sample: np.array, all_phenotypes: np.array, node_attention_scores:list) -> Tuple[list, list, list]:
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
    data_filtered = data_filtered.fillna(0)
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



def plot_cell_cell_interaction_boxplots(
    significance_1: float,
    significance_2: float,
    interaction_limit: int,
    all_interaction_mean_df: pd.DataFrame,
    top_connections: dict,
    save_path: Optional[PosixPath] = None,
    # plotting parameteres
    log_y:bool = False,
    star_size:int = 2000,
    line_width:int = 5,
    costum_fig_size: Optional[Tuple[int,int]] = None,
    costum_star_shift: Optional[float] = None
):
    """
    Plots boxplots of cell-cell interactions and performs statistical significance tests.

    Args:
        significance_1 (float): The p-value threshold for marking significant interactions with one star.
        significance_2 (float): The p-value threshold for marking highly significant interactions with two stars.
        interaction_limit (int): The number of top interactions to display per cell type.
        all_interaction_df (pd.DataFrame): DataFrame containing interaction values between cell types.
        top_connections (dict): Dictionary with cell types as keys and lists of (destination cell type, interaction values) as values.
        save_path (Optional[PosixPath]): Path to save the plot. If None, the plot is not saved.

    Returns:
        None
    """

    # Set plot font size
    plt.rcParams.update({'font.size': 50})

    cell_types = all_interaction_mean_df.columns
    num_cells = len(cell_types)

    # Configure figure size and double star shift based on interaction limit
    if costum_fig_size is not None:
        fig_size = costum_fig_size
    else:
        fig_size = (170, 60) if interaction_limit == 16 else (130, 60)

    if costum_star_shift is not None:
        double_star_shift = costum_star_shift
    else:
        double_star_shift = 0.16 if interaction_limit == 16 else 0.1

    # Create subplots (2 rows, num_cells/2 columns)
    fig, axs = plt.subplots(
        nrows=2,
        ncols=int(num_cells / 2),
        figsize=fig_size,
        sharey=True,
        layout='constrained'
    )



    # Iterate over subplots
    for row in range(2):
        for idx in range(int(num_cells / 2)):
            # Get the source cell type
            src_cell = cell_types[idx + int(num_cells / 2) * row]

            # Extract names and values of top interactions
            names = [dst_cell[0] for dst_cell in top_connections[src_cell]]
            values = [dst_cell[1][~np.isnan(dst_cell[1])] for dst_cell in top_connections[src_cell] if len(dst_cell[1]) > 2]

            # Create boxplot
            axs[row, idx].set_title(break_title(src_cell))
            axs[row, idx].grid(True, linewidth=line_width)
            box = axs[row, idx].boxplot(values)

            # Customize plot aesthetics
            for element in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
                plt.setp(box[element], linewidth=line_width)

            # Thicken subplot axis spines
            for spine in axs[row, idx].spines.values():
                spine.set_linewidth(line_width)

            # Perform Mann-Whitney U tests for each destination cell type
            for dst_cell_idx, dst_name in enumerate(names):
                # Convert NaN values to zero for statistical testing
                data_1 = np.nan_to_num(all_interaction_mean_df[src_cell].to_numpy())
                data_2 = np.nan_to_num(values[dst_cell_idx].to_numpy())

                # Conduct Mann-Whitney U test
                _, p = mannwhitneyu(data_1, data_2, alternative='less')

                # Mark significance on the plot
                if significance_2 < p < significance_1:
                    axs[row, idx].scatter(dst_cell_idx + 1, 1.05, marker='*', color='black', s=star_size)
                elif p < significance_2:
                    axs[row, idx].scatter(dst_cell_idx + 1 + double_star_shift, 1.05, marker='*', color='black', s=star_size)
                    axs[row, idx].scatter(dst_cell_idx + 1 - double_star_shift, 1.05, marker='*', color='black', s=star_size)

            if values:
                # Ensure x-ticks match available names
                if len(names) < interaction_limit:
                    missing_names = cell_types[~cell_types.isin(names)].to_list()
                    names.extend(missing_names)

                axs[row, idx].set_xticks(np.arange(1, len(values) + 1), names[:len(values)], rotation=90)

            axs[row, idx].grid(False)

            # Compute statistical summaries (median, quartiles)
            mat = np.nan_to_num(all_interaction_mean_df[src_cell].to_numpy())
            median = np.median(mat)
            lower_q, upper_q = np.quantile(mat, [0.25, 0.75])

            # Highlight interquartile range and median line
            axs[row, idx].axhspan(lower_q, upper_q, facecolor='green', alpha=0.2)
            axs[row, idx].axhline(y=median, color='red', linestyle='--', linewidth=line_width)
            if log_y:
                axs[row, idx].set_yscale('log')

    # Save figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150)

