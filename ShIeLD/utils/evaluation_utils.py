#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""

from . import train_utils
from . import model_utils
from . import data_utils
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from pathlib import PosixPath
from tqdm import tqdm
import pickle
from typing import Dict, Optional, Tuple, Union
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


def get_best_config_dict(hyper_search_results, requirements_dict):
    """
    Extracts the best configuration from hyperparameter search results.

    Parameters:
    - hyper_search_results (pd.DataFrame): DataFrame containing hyperparameter search results.
    - requirements_dict (dict): Dictionary containing requirements for the model.

    Returns:
    - dict: The best configuration dictionary.
    """
    if 'sampleing' in requirements_dict.keys():
        if requirements_dict['sampleing'] == 'random':
            requirements_dict['fussy_limit'] = 'random_sampling'

    must_have_columns = ['layer_1', 'input_layer', 'droupout_rate', 'output_layer', 'comment_norm',
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
    best_config_dict['output_layer'] = requirements_dict['output_layer']

    return best_config_dict


def get_cell_to_cell_interaction_dict(
        requirements_dict: Dict,
        data_loader: DataLoader,
        model: torch.nn.Module,
        device: str,
        save_dict_path: Optional[PosixPath] = None,
        column_celltype_name: str = 'CellType',
        edge_noise: Union[bool, str] = False,
) -> Dict:
    """
    Computes cell-to-cell interaction metrics from a trained model's predictions and attention scores.

    Parameters:
    - requirements_dict (dict): A dictionary containing experimental setup details (e.g., cell type names).
    - data_loader: A PyTorch DataLoader object for iterating over graph-based cell data.
    - model: The trained neural network model used for prediction.
    - attr_bool (bool): Whether to use attribution-based attention scores.
    - device: The computation device (e.g., "cpu" or "cuda").
    - save_dict_path (Optional[PosixPath]): Path to save/load the output dictionary. If None, results are not saved.
    - column_celltype_name (str): which column in the evaluation data contains cell type information.

    -edge_noise (bool,str): If an alternative edge index is used this parameter describes it

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

    shorten_celltypeNames = False

    # Extract cell type information from requirements_dict
    if 'combine_cellPhenotypes' in requirements_dict.keys():
        if requirements_dict['combine_cellPhenotypes'] is not None:
            cell_types = data_utils.combine_cell_types(original_array=np.array(requirements_dict['cell_type_names']),
                                                       string_list=requirements_dict['combine_cellPhenotypes'],
                                                       retrun_adj_matrix=False)
            shorten_celltypeNames = True
        else:
            print(f'cannot use {requirements_dict["combine_cellPhenotypes"]} to combine the cell types will use all')
    else:
        cell_types = np.array(requirements_dict['cell_type_names'])

    # Find the index of the 'CellType' column in the evaluation data
    try:
        cell_type_eval_index = np.where(
            np.char.lower(np.array(requirements_dict['eval_columns'])) == column_celltype_name.lower()
        )[0][0]
    except IndexError:
        raise ValueError(f"Column {column_celltype_name} not found in {requirements_dict['eval_columns']}.")

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
        if shorten_celltypeNames:
            cell_type_names = []
            for sample in data_sample:
                cell_type_names.append(data_utils.combine_cell_types(
                    original_array=sample.eval[:, cell_type_eval_index],
                    string_list=requirements_dict['combine_cellPhenotypes'],
                    retrun_adj_matrix=True)[1])
        else:
            cell_type_names = [sample.eval[:, cell_type_eval_index] for sample in data_sample]

        # Compute phenotype-to-phenotype attention scores
        phenotype_attention_matrix = get_p2p_att_score(
            sample=data_sample,
            cell_phenotypes_sample=cell_type_names,
            all_phenotypes=cell_types,
            node_attention_scores=node_level_attention_scores,
            edge_noise=edge_noise
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
    hyper_search_results, csv_file_path = train_utils.get_train_results_csv(requirement_dict=requirements_dict)

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


def get_top_interaction_per_celltype(interaction_limit: int,
                                     all_interaction_mean_df: pd.DataFrame,
                                     all_interaction_df: pd.DataFrame) -> Dict:
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


def get_p2p_att_score(sample: list, cell_phenotypes_sample: np.array, all_phenotypes: np.array,
                      node_attention_scores: list, edge_noise: Union[bool, str]) -> Tuple[list, list, list]:
    """
    This function calculates the attention score for each cell phenotype to all other phenotypes.
    It uses the end attention score p2p (phenotype to phenotype).
    It also uses the word "node" instead of cell.

    Parameters:
    sample (list): A list of data samples.
    cell_phenotypes_sample (np.array): An array of cell phenotype names for each sample.
    all_phenotypes (np.array): An array of all phenotype names.
    node_attention_scores (list): A list of attention scores for each node.
    edge_noise (bool,str): If an alternative edge index is used this parameter describes it

    Returns:
    list: A list of raw attention scores for each phenotype to all other phenotypes.
    list: A list of normalization factors based on the raw count of edges between two phenotypes.
    list: A list of normalized attention scores for each phenotype to all other phenotypes.
    """

    raw_att_p2p = []
    normalisation_factor_edge_number = []
    normalised_p2p = []

    if edge_noise is False:
        edge_index_name = 'edge_index_plate'
    else:
        edge_index_name = f'edge_index_plate_{edge_noise}'

    # Find the cell type (phenotype) for each cell/node
    scr_node = [cell_phenotypes_sample[sample_idx][sample[sample_idx][edge_index_name][0].cpu()] for sample_idx in
                range(len(sample))]
    dst_node = [cell_phenotypes_sample[sample_idx][sample[sample_idx][edge_index_name][1].cpu()] for sample_idx in
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


def plot_confusion_with_std(mean_cm: np.array, std_cm: np.array, class_names: list,
                            title='Mean Confusion Matrix ± STD (%)'):
    """
    Plots a confusion matrix with values shown as mean ± std in each cell.

    Parameters:
    - mean_cm (ndarray): Mean confusion matrix (in %), shape (C, C)
    - std_cm (ndarray): Standard deviation matrix, shape (C, C)
    - class_names (list of str): Names of the classes
    - title (str): Plot title
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.Blues
    im = ax.imshow(mean_cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('%', rotation=270, labelpad=15)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotate each cell with "mean±std"
    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            cell_text = f"{mean_cm[i, j]:.2f}±{std_cm[i, j]:.2f}"
            ax.text(j, i, cell_text, ha='center', va='center', color='black', fontsize=15)

    plt.tight_layout()
    plt.show()


def create_parameter_influence_plots(df: pd.DataFrame, observed_variable: str, save_path: Optional[PosixPath] = None):
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
        interaction_limit: int,
        all_interaction_mean_df: pd.DataFrame,
        top_connections: dict,
        save_path: Optional[PosixPath] = None,
        *,
        log_y: bool = False,
        star_size: int = 2000,
        line_width: int = 5,
        custom_fig_size: Optional[Tuple[int, int]] = None,
        custom_star_shift: Optional[float] = None,
        nan_to_zero: bool = False,
        stars: bool = False,
        significance_1: Optional[float] = None,
        significance_2: Optional[float] = None,
):
    """
    Plots boxplots of cell-cell interactions with optional statistical testing and significance annotations.

    Args:
        interaction_limit: Number of top interactions to display per cell type.
        all_interaction_mean_df: DataFrame with interaction values between cell types.
        top_connections: Dictionary {cell type: list of (destination cell type, values)}.
        save_path: Optional path to save the figure and p-value CSVs.
        log_y: If True, apply log scale to y-axis.
        star_size: Size of stars used for significance.
        line_width: Line width for plot elements.
        custom_fig_size: Custom figure size.
        custom_star_shift: Shift distance for double-star placement.
        nan_to_zero: If True, converts NaNs to zeros before statistical testing.
        stars: Whether to annotate significance with stars.
        significance_1: Threshold for single-star significance.
        significance_2: Threshold for double-star significance.
    """
    if stars and (significance_1 is None or significance_2 is None):
        raise ValueError("Both significance_1 and significance_2 must be set when stars=True")

    plt.rcParams.update({'font.size': 50})

    cell_types = all_interaction_mean_df.columns
    num_cells = len(cell_types)

    fig_size = custom_fig_size or ((170, 60) if interaction_limit > 16 else (130, 60))
    double_star_shift = custom_star_shift if custom_star_shift is not None else (
        0.16 if interaction_limit > 16 else 0.1)

    fig, axs = plt.subplots(nrows=2, ncols=num_cells // 2, figsize=fig_size, sharey=True, layout='constrained')

    p_val_scores = []
    fdr_scores = []
    for row in range(2):
        for idx in range(num_cells // 2):
            ax = axs[row, idx]
            src_cell = cell_types[idx + (num_cells // 2) * row]
            top_conns = top_connections.get(src_cell, [])

            names = [dst_cell[0] for dst_cell in top_conns]
            values = [dst_cell[1][~np.isnan(dst_cell[1])] for dst_cell in top_conns if len(dst_cell[1]) > 2]

            ax.set_title(src_cell)
            ax.grid(True, linewidth=line_width)

            box = ax.boxplot(values, patch_artist=True)

            # Make boxplots white (opaque)
            for patch in box['boxes']:
                patch.set_facecolor('white')

            # Line styling
            for element in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
                plt.setp(box[element], linewidth=line_width)

            for spine in ax.spines.values():
                spine.set_linewidth(line_width)

            for dst_cell_idx, dst_name in enumerate(names):
                if nan_to_zero:
                    data_1 = np.nan_to_num(all_interaction_mean_df[src_cell].to_numpy())
                    data_2 = np.nan_to_num(values[dst_cell_idx])
                else:
                    d1 = all_interaction_mean_df[src_cell].to_numpy()
                    d2 = values[dst_cell_idx]
                    data_1 = d1[~np.isnan(d1)]
                    data_2 = d2[~np.isnan(d2)]

                _, p_val = mannwhitneyu(data_1, data_2, alternative='less')
                p_val_scores.append([src_cell, dst_name, p_val])

                if stars:
                    if significance_2 < p_val < significance_1:
                        ax.scatter(dst_cell_idx + 1, 1.05, marker='*', color='black', s=star_size)
                    elif p_val < significance_2:
                        ax.scatter(dst_cell_idx + 1 + double_star_shift, 1.05, marker='*', color='black', s=star_size)
                        ax.scatter(dst_cell_idx + 1 - double_star_shift, 1.05, marker='*', color='black', s=star_size)

            _, p_adj, _, _ = multipletests([x[2] for x in p_val_scores], method='fdr_bh')
            adjusted_log10 = np.log10(p_adj)

            for dst_cell_idx, dst_name in enumerate(names):
                fdr_scores.append([src_cell, dst_name, adjusted_log10[dst_cell_idx]])
                sig_text = f"{adjusted_log10[dst_cell_idx]:.2f}"
                ax.text(dst_cell_idx + 1, 1.01, sig_text, color='black', fontsize=30, ha='center')

            if values:
                if len(names) < interaction_limit:
                    missing = cell_types[~cell_types.isin(names)].to_list()
                    names.extend(missing)

                ax.set_xticks(np.arange(1, len(values) + 1))
                ax.set_xticklabels(names[:len(values)], rotation=90)

            ax.grid(True, axis='y')

            full_data = np.nan_to_num(all_interaction_mean_df[src_cell].to_numpy())
            median = np.median(full_data)
            q1, q3 = np.quantile(full_data, [0.25, 0.75])
            ax.axhspan(q1, q3, facecolor='green', alpha=0.2)
            ax.axhline(y=median, color='red', linestyle='--', linewidth=line_width)

            if log_y:
                ax.set_yscale('log')

    # Build and export p-value matrix
    log_p_matrix = pd.DataFrame(p_val_scores, columns=['src', 'dst', 'p']).pivot(index='src', columns='dst', values='p')
    log_FDR_matrix = pd.DataFrame(fdr_scores, columns=['src', 'dst', 'p']).pivot(index='src', columns='dst', values='p')

    if save_path is not None:
        plot_path = save_path.with_name(save_path.stem + ('_log' if log_y else '') + save_path.suffix)
        log_p_matrix.to_csv(plot_path.with_name(plot_path.stem + '_p_values.csv'))
        log_FDR_matrix.to_csv(plot_path.with_name(plot_path.stem + '_FDR_values.csv'))
        plt.savefig(plot_path, dpi=250)
        plot_top_k_log_p_values_per_row(log_pval_matrix=log_FDR_matrix,
                                        k=interaction_limit,
                                        font_size=50,
                                        figsize=(20, 10),
                                        rename_labels=False,
                                        save_path=plot_path.with_name(plot_path.stem + '_FDR_values_per_row.png'))

    plt.show()


def plot_top_k_log_p_values_per_row(
        log_pval_matrix: pd.DataFrame,
        k: int = 8,
        font_size: int = 12,
        figsize: tuple = (8, 6),
        rename_labels: bool = True,
        save_path: Optional[PosixPath] = None
):
    """
    Plots the top-k -log10(p) values for each row in a p-value matrix (e.g., FDR-corrected p-values).
    Each row corresponds to a source cell type and columns to destination cell types.
    For each row, an individual scatter plot is generated showing the most significant interactions.

    Args:
        log_pval_matrix (pd.DataFrame): DataFrame with 'src' column (row labels) and -log10(p) values in other columns.
        k (int): Number of top -log10(p) values to plot per row (default: 8).
        font_size (int): Font size used in plots (default: 12).
        figsize (tuple): Size of each individual plot (default: (8, 6)).
        rename_labels (bool): If True, abbreviates 'M2 Macrophages PD-L1+' to 'MAC +' and 'PD-L1-' to 'MAC -'.

    Returns:
        None
    """
    # Extract row labels and clean data matrix
    src_labels = log_pval_matrix['src']
    df_values = log_pval_matrix.drop(columns=log_pval_matrix.columns[0])  # Drop 'src'

    plt.rcParams.update({'font.size': font_size})

    # Loop through each row (i.e., each source cell type)
    for row in range(len(df_values)):
        values = df_values.iloc[row]
        mask = ~values.isna()  # Only keep non-NaN values
        vals_clean = values[mask]
        labels_clean = np.array(values.index)[mask]

        # Compute -log10(p) from log10(p)
        neg_log_vals = -vals_clean

        # Sort and get top-k
        sorted_idx = np.argsort(neg_log_vals)[::-1]
        top_idx = sorted_idx[:k]
        top_vals = neg_log_vals.iloc[top_idx]
        top_labels = labels_clean[top_idx]
        x_vals = np.arange(len(top_vals))

        # Create the plot
        plt.figure(figsize=figsize)
        plt.scatter(x_vals, top_vals, color='tab:blue', s=10)

        # Annotate points with cell type names
        for j, txt in enumerate(top_labels):
            if rename_labels:
                if txt == 'M2 Macrophages PD-L1+':
                    txt = 'MAC +'
                elif txt == 'M2 Macrophages PD-L1-':
                    txt = 'MAC -'

            plt.annotate(txt, (x_vals[j], top_vals.iloc[j]),
                         fontsize=10, rotation=90, ha='left', va='bottom',
                         clip_on=False, xytext=(0, 2), textcoords='offset points')

        # Plot configuration
        plt.title(src_labels[row], fontsize=font_size)
        plt.ylabel('-log10(p)', fontsize=font_size)
        plt.xticks([])
        plt.ylim([0, max(neg_log_vals) * 1.2])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(save_path, dpi=250)


def plot_confusion_with_std(mean_cm, std_cm, class_names, title='Mean Confusion Matrix ± STD (%)'):
    """
    Plots a confusion matrix with values shown as mean ± std in each cell.

    Parameters:
    - mean_cm (ndarray): Mean confusion matrix (in %), shape (C, C)
    - std_cm (ndarray): Standard deviation matrix, shape (C, C)
    - class_names (list of str): Names of the classes
    - title (str): Plot title
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.Blues
    im = ax.imshow(mean_cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('%', rotation=270, labelpad=15)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotate each cell with "mean±std"
    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            cell_text = f"{mean_cm[i, j]:.2f}±{std_cm[i, j]:.2f}"
            ax.text(j, i, cell_text, ha='center', va='center', color='black', fontsize=15)

    plt.tight_layout()
    plt.show()
