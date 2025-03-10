#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""
import torch
from torch_geometric.data import Dataset
from pathlib import Path
import os
import sys
import pickle
import pandas as pd

cwd = os.getcwd()
sys.path.append(cwd + "/utils/")


class diabetis_dataset(Dataset):
    """
    Custom PyTorch Geometric dataset for loading diabetes-related graph data.

    This dataset loads preprocessed graph files from a directory. If a file containing
    the list of graph filenames does not exist, it filters graph files based on a CSV
    and saves the list for future use.

    Attributes:
        csv_file (pd.DataFrame): DataFrame containing image names from the CSV file.
        graph_file_names (list): List of filtered graph filenames.
        root (str): Root directory containing graph files.
    """


    def __init__(self, root, fold_ids,requirements_dict, graph_file_names,path_to_graphs):
        """
        Initializes the diabetis_dataset.

        Args:
            root (str): Path to the directory containing the graph files.
            csv_file (str): Path to the CSV file containing image names.
            graph_file_names (str): Path to store/retrieve the list of graph filenames.
        """

        # If the file storing graph filenames does not exist, create it
        if not Path.exists(path_to_graphs/graph_file_names):
            print('Creating file name list:')

            csv_file = pd.read_csv(requirements_dict['path_raw_data'])  # Load CSV file containing image names
            csv_file = csv_file[requirements_dict['measument_sample_name']][
                csv_file[requirements_dict['validation_split_column']].isin(fold_ids)]
            # List all files in the root directory
            df = pd.DataFrame(os.listdir(root), columns=['file_name'])

            # Create a regex pattern from the image names in the CSV file
            regex_pattern = '|'.join(csv_file.unique())

            # Filter filenames in the root directory that match the CSV images
            filtered_df = df[df['file_name'].str.contains(regex_pattern)]
            self.graph_file_names = filtered_df['file_name'].tolist()

            # Save the filtered filenames to avoid redundant processing in future runs
            with open(path_to_graphs/graph_file_names, 'wb') as f:
                pickle.dump(self.graph_file_names, f)

        else:
            print('Load the previously stored list of graph filenames')

            # Load the previously stored list of graph filenames
            with open(Path(path_to_graphs/graph_file_names), 'rb') as f:
                self.graph_file_names = pickle.load(f)

        super().__init__(root)

    @property
    def processed_dir(self):
        """
        Returns the root directory where processed graph files are stored.
        """
        return self.root

    @property
    def processed_file_names(self):
        """
        Returns the list of processed graph filenames.
        """
        return self.graph_file_names

    def process(self):
        """
        Dummy function. Required by PyG's Dataset class but not implemented.
        """
        pass

    def len(self):
        """
        Returns the total number of graphs in the dataset.

        Returns:
            int: Number of processed graph files.
        """
        return len(self.processed_file_names)

    def get(self, idx):
        """
        Retrieves a graph data object by index.

        Args:
            idx (int): Index of the graph file to retrieve.

        Returns:
            torch_geometric.data.Data: The loaded graph data.
        """
        data = torch.load(f'{self.processed_dir}/{self.processed_file_names[idx]}')
        return data

class local_immune_graph_dataset(Dataset):
    def __init__(self, root, path_to_name_file):

        # since the number of graph vary from patient and region, we need to save the names of the graphs
        # in a seperate file and load this for the training
        self.path_to_name_file = path_to_name_file


        super().__init__(root)

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        with open(self.path_to_name_file, 'rb') as f:
            graph_file_names = pickle.load(f)

        return graph_file_names


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        try:
            data = torch.load(f'{self.processed_dir}/{self.processed_file_names[idx]}')

        except:
            print(f'Error in loading graph {self.processed_file_names[idx]}')
            print('Please check the graph file and the path to the graph file')
            print('Trouble shoot: Rerun data_processing.py and check if the graph file is created')

        return data
