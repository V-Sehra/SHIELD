
import torch
from torch_geometric.data import Dataset
import os
import sys
import pickle

cwd = os.getcwd()
sys.path.append(cwd + "/utils/")


cosine = torch.nn.CosineSimilarity(dim=1)


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
