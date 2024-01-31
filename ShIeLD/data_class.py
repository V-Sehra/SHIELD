import numpy as np
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import os
import sys
import pickle

cwd = os.getcwd()
sys.path.append(cwd + "/utils/")
import model_utils
import copy

cosine = torch.nn.CosineSimilarity(dim=1)


class local_immune_graph_dataset(Dataset):
    def __init__(self, root, path_to_name_file,
                 path_csv_file,
                 radius_neighbourhood=199,
                 minimum_number_cells=50,
                 number_of_samples=20):

        # since the number of graph vary from patient and region, we need to save the names of the graphs
        # in a seperate file and load this for the training
        self.path_to_name_file = path_to_name_file
        # the orignal data path
        self.path_csv_file = path_csv_file

        # the parameters for the graph creation
        # number of samples per y and x-axis
        self.number_of_samples = number_of_samples
        # minimum number of cells per graph
        self.minimum_number_cells = minimum_number_cells
        # radius of the neighbourhood
        self.radius_neighbourhood = radius_neighbourhood

        self.cwd = os.getcwd()
        # load the original data set
        self.input_data = pd.read_csv(f'{self.path_csv_file}')
        self.patient_id_list = self.input_data['Patient'].unique()
        #tissue typed/ Labels
        self.tissue_dict = {'normalLiver': 0,
                            'core': 1,
                            'rim': 2}

        super().__init__(root)

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        with open(self.path_to_name_file, 'rb') as f:
            graph_file_names = pickle.load(f)

        return graph_file_names

    def process(self):
        self.input_data = self.input_data[self.input_data['CD45.Positive.Classification'] == 1]

        column_indicator_gene = self.input_data.columns.str.contains('.Intensity')
        column_indicator_eval_information = self.input_data.columns.isin(['Class', 'Class0', 'Celltype'])
        column_indicator_cd45 = self.input_data.columns.isin(['CD45.Positive.Classification'])

        os.system(f'mkdir -p {self.root}')

        graph_name_list = []

        for patient_id in self.patient_id_list:

            # Select the data from the Patient
            whole_data_pat = self.input_data[self.input_data['Patient'] == patient_id]
            for origin in self.tissue_dict.keys():

                #segment the whole image into the desired tissue type
                tissue_segmented_data = whole_data_pat.loc[whole_data_pat['Tissue'] == origin]

                #get the x and y range of the tissue
                x_0 = tissue_segmented_data['X_value'].min()
                x_max = tissue_segmented_data['X_value'].max()

                y_0 = tissue_segmented_data['Y_value'].min()
                y_max = tissue_segmented_data['Y_value'].max()

                # increase the x baseline range to get a more symetric increase
                delta_x = ((x_max - x_0) / self.number_of_samples) * 3
                padding_x = delta_x * 0.1

                # get the y step distance
                delta_y = (y_max - y_0) / self.number_of_samples
                padding_y = delta_y * 0.1

                batch_counter = 0
                # propergate through the image and create the graphs
                for x_step in range(self.number_of_samples):

                    x_step_0 = x_0 + delta_x * x_step
                    x_step_max = x_0 + delta_x * (1 + x_step)

                    sub_cell_y = tissue_segmented_data[((tissue_segmented_data['X_value'] > (x_step_0 - padding_x)) &
                                                        (tissue_segmented_data['X_value'] < (x_step_max + padding_x)))]

                    y_step_0 = sub_cell_y['Y_value'].min()
                    y_step_max = copy.copy(y_step_0) + delta_y

                    # increase the size of the y step if the number of cells is too small
                    while y_step_max < sub_cell_y['Y_value'].max():

                        sub_sample_cells = sub_cell_y[((sub_cell_y['Y_value'] > (y_step_0 - padding_y)) &
                                                       (sub_cell_y['Y_value'] < (y_step_max + padding_y)))]

                        if len(sub_sample_cells) < self.minimum_number_cells:
                            while (len(sub_sample_cells) < self.minimum_number_cells) and (
                                    y_step_max < sub_cell_y['Y_value'].max()):
                                y_step_max = y_step_max + delta_y
                                sub_sample_cells = sub_cell_y[((sub_cell_y['Y_value'] > (y_step_0 - padding_y)) &
                                                               (sub_cell_y['Y_value'] < (y_step_max + padding_y)))]

                            y_step_0 = copy.copy(y_step_max)
                        else:
                            y_step_0 = y_step_0 + delta_y

                        y_step_max = y_step_max + delta_y

                        immu_mask = np.concatenate(
                            np.array(sub_sample_cells.loc[:, column_indicator_cd45], dtype='bool'))

                        # check if enough cells have been selected (boarder region might be too small)
                        if len(sub_sample_cells) > self.minimum_number_cells:
                            sub_sample_cells = sub_sample_cells[immu_mask]

                            real_cordinates = sub_sample_cells[['Y_value', 'X_value']]
                            plate_edge, plat_att = model_utils.calc_edge_mat(real_cordinates,
                                                                                dist_bool=True,
                                                                                radius=self.radius_neighbourhood)

                            # remove all "overlapping" cells
                            plate_edge = model_utils.remove_zero_distances(edge = plate_edge,
                                                                         dist = plat_att,
                                                                         dist_return= False)

                            # select the desired features for the nodes
                            cells = torch.tensor((sub_sample_cells.loc[:, column_indicator_gene]).to_numpy()).float()

                            data = Data(x=cells,

                                        edge_index_plate=torch.tensor(plate_edge).long(),
                                        orginal_cord=torch.tensor(real_cordinates.to_numpy()),

                                        eval=sub_sample_cells.loc[:, column_indicator_eval_information]
                                        .to_numpy(dtype=np.dtype('str')),

                                        eval_col_names=sub_sample_cells.loc[:, column_indicator_eval_information]
                                        .columns.to_numpy(dtype=np.dtype('str')),

                                        ids = patient_id,

                                        y=torch.tensor(self.tissue_dict[origin]).flatten()).cpu()

                            torch.save(data, os.path.join(f'{self.root}',
                                                          f'graph_pat_{patient_id}_{origin}_{batch_counter}.pt'))
                            graph_name_list.append(f'graph_pat_{patient_id}_{origin}_{batch_counter}.pt')
                            torch.cuda.empty_cache()
                            batch_counter += 1

        # save the graph names
        with open(os.path.join(self.path_to_name_file), 'wb') as f:
            pickle.dump(graph_name_list, f)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(f'{self.processed_dir}/{self.processed_file_names[idx]}')

        return data
