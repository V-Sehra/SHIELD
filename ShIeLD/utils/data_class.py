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
from tqdm import tqdm
cwd = os.getcwd()
sys.path.append(cwd + "/utils/")


class graph_dataset(Dataset):
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


    def __init__(self, root, fold_ids,requirements_dict, graph_file_names,path_to_graphs,
                 normalize=None,normalizer_filename=None) :
        """
        Initializes the diabetis_dataset.

        Args:
            root (str): Path to the directory containing the graph files.
            csv_file (str): Path to the CSV file containing image names.
            graph_file_names (str): Path to store/retrieve the list of graph filenames.
            normalize (bool): If True, normalize the graph data.
        """


        # If the file storing graph filenames does not exist, create it
        if not Path.exists(path_to_graphs/graph_file_names):
            print('Creating file name list:')

            csv_file = pd.read_csv(requirements_dict['path_raw_data'])
            image_ids = csv_file[requirements_dict['measument_sample_name']][
                csv_file[requirements_dict['validation_split_column']].isin(fold_ids)
            ].unique()

            # List files in root
            df = pd.DataFrame(os.listdir(root), columns=['file_name'])

            # Extract the "XX_B" or "XX_A" pattern from each filename using regex
            df['identifier'] = df['file_name'].str.extract(r'graph_subSample_(\d+_[AB])_')[0]

            # Filter exact matches
            filtered_df = df[df['identifier'].isin(image_ids)]

            self.graph_file_names = filtered_df['file_name'].tolist()

            # Save the filtered filenames to avoid redundant processing in future runs
            with open(path_to_graphs/graph_file_names, 'wb') as f:
                pickle.dump(self.graph_file_names, f)

        else:
            print('Load the previously stored list of graph filenames')

            # Load the previously stored list of graph filenames
            with open(Path(path_to_graphs/graph_file_names), 'rb') as f:
                self.graph_file_names = pickle.load(f)

        self.normalize = normalize
        if self.normalize is not None:

            if 'global_std' in self.normalize:
                if not Path.exists(path_to_graphs / normalizer_filename):
                    print('need to create the standardizer values')
                    sum_all = 0
                    total_number = 0
                    for file in tqdm(self.graph_file_names):
                        data = torch.load(f'{root}/{file}')
                        sum_all += torch.sum(data.x, dim=0)
                        total_number += data.x.shape[0]

                    self.total_mean = sum_all / total_number

                    sig2 = 0
                    for file in tqdm(self.graph_file_names):
                        data = torch.load(f'{root}/{file}')
                        sig2 += torch.sum(((data.x - self.total_mean) ** 2), dim=0)

                    self.sig = torch.sqrt(sig2 / total_number)

                    normalising_factors = [self.total_mean, self.sig]

                    with open(path_to_graphs / normalizer_filename, 'wb') as f:
                        pickle.dump(normalising_factors, f)
                else:
                    print('load the standardizer values')
                    with open(path_to_graphs / normalizer_filename, 'rb') as f:
                        self.total_mean, self.sig = pickle.load(f)

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
        if self.normalize is not None:
            if 'global_std' in self.normalize:
                data.x = (data.x - self.total_mean) / self.sig
        return data

