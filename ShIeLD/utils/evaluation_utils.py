#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""
import seaborn as sns
import matplotlib.pyplot as plt
import train_utils
from pandas import DataFrame
from typing import Optional
from pathlib import PosixPath


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


def create_parameter_influence_plots(df: DataFrame, observed_variable:str, save_path: Optional[PosixPath] =None):
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

