#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:18:15 2024

@author: Vivek
"""

import os
import pandas as pd
import copy
import argparse
import sys
import torch
import pickle
cwd = os.getcwd()
sys.path.append(os.path.join(f'{cwd}', 'utils'))

idx_folder = cwd.index('ShIeLD') + len('ShIeLD')
shield_dir = os.path.join(f'{cwd[:idx_folder]}')

import data_utils

tissue_type_name = ['normalLiver', 'core', 'rim']
tissue_dict = {'normalLiver': 0,
               'core': 1,
               'rim': 2}


def create_data_set():
    parser = argparse.ArgumentParser()

    parser.add_argument("-path_to_raw_csv", "--path_to_raw_csv", type=str, default=os.path.join(f'{shield_dir}','data','raw_data.csv'))
    parser.add_argument("-path_to_save_data", "--path_to_save_data", type=str, default=os.path.join(f'{shield_dir}','data'))
    parser.add_argument("-new_test_split_bool", "--new_test_split_bool", type=bool, default='False',choices=['True', 'False'])

    parser.add_argument("-r_s", "--number_steps_region_subsampleing", type=int, default=100)
    parser.add_argument("-radius", "--radius", type=int, default=50)
    parser.add_argument("-min_cells", "--minimum_number_cells", type=str, default=50)
    parser.add_argument("-marker_list", "--marker_list", type=str, default=['CD56','CD16','CD163','CD68','CD39','CD11c','CD38','CD11b',
                                                                              'CD15','CD66b','CD40','HLA-DR','CD62L','CD19','CD45RA','CD57',
                                                                              'PD-L1','CD25','FoxP3','CD45RO','IL18Rα','CD4','CD3','ICOS','PD-1','CD69',
                                                                              'CD8','CD161','TCRVα 7.2'])

    # Parse the command-line arguments
    args = parser.parse_args()

    if (not os.path.exists(os.path.join(f'{args.path_to_save_data}', 'train_set', 'train_cells.csv')))\
            or (not os.path.exists(os.path.join(f'{args.path_to_save_data}', 'test_set', 'test_cells.csv'))):
        print('initalizing the data set')
        data_utils.create_test_train_split(args)

    for data_type in ['train', 'test']:
        print(f'Creating the graphs for  {data_type} set')
        """
        This loop iterates over the data types 'train' and 'test'.
        For each data type, it performs the following steps:
        - Defines the directory paths for the data set and the graphs.
        - Reads the CSV file for the data set.
        - Filters the data to include only cells with a positive CD45 classification.
        - Defines indicators for the columns that contain gene intensities, evaluation information, and CD45 classification.
        - Initializes an empty list for the names of the graphs.
        - Iterates over the unique patient IDs in the data.
            - For each patient ID, it selects the data for that patient and iterates over the keys in the tissue dictionary.
                - For each tissue type, it segments the data to include only that tissue type and gets the X and Y range of the tissue.
                - It calculates the X and Y step distances and initializes a counter for the batches.
                - It then iterates over the number of samples, segmenting the data based on the X and Y step distances and increasing the size of the Y step if the number of cells is too small.
                - If enough cells have been selected, it selects the desired features for the nodes, creates a Data object, and saves it as a graph.
                - It also appends the name of the graph to the list of graph names and increments the batch counter.
        """


        folder_dir = os.path.join(f'{args.path_to_save_data}', f'{data_type}_set')
        save_path_graphs = os.path.join(f'{folder_dir}', 'graphs')
        os.system(f'mkdir -p {save_path_graphs}')

        input_data = pd.read_csv(os.path.join(f'{folder_dir}', f'{data_type}_cells.csv'))

        input_data = input_data[input_data['CD45.Positive.Classification'] == 1]

        column_indicator_gene = input_data.columns.str.contains('|'.join(args.marker_list))
        column_indicator_eval_information = input_data.columns.isin(['Class', 'Class0', 'Celltype'])

        graph_name_list = []

        for patient_id in input_data['Patient'].unique():

            # Select the data from the Patient
            whole_data_pat = input_data[input_data['Patient'] == patient_id]
            for origin in tissue_dict.keys():

                # segment the whole image into the desired tissue type
                tissue_segmented_data = whole_data_pat.loc[whole_data_pat['Tissue'] == origin]

                # get the x and y range of the tissue
                x_0 = tissue_segmented_data['X_value'].min()
                x_max = tissue_segmented_data['X_value'].max()

                y_0 = tissue_segmented_data['Y_value'].min()
                y_max = tissue_segmented_data['Y_value'].max()

                # increase the x baseline range to get a more symetric increase
                delta_x = ((x_max - x_0) / args.number_steps_region_subsampleing) * 3
                padding_x = delta_x * 0.1

                # get the y step distance
                delta_y = (y_max - y_0) / args.number_steps_region_subsampleing
                padding_y = delta_y * 0.1

                batch_counter = 0
                # propergate through the image and create the graphs
                for x_step in range(args.number_steps_region_subsampleing):

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

                        if len(sub_sample_cells) < args.minimum_number_cells:
                            while (len(sub_sample_cells) < args.minimum_number_cells) and (
                                    y_step_max < sub_cell_y['Y_value'].max()):
                                y_step_max = y_step_max + delta_y
                                sub_sample_cells = sub_cell_y[((sub_cell_y['Y_value'] > (y_step_0 - padding_y)) &
                                                               (sub_cell_y['Y_value'] < (y_step_max + padding_y)))]

                            y_step_0 = copy.copy(y_step_max)
                        else:
                            y_step_0 = y_step_0 + delta_y

                        y_step_max = y_step_max + delta_y


                        # check if enough cells have been selected (boarder region might be too small)
                        if len(sub_sample_cells) > args.minimum_number_cells:
                            file_name = data_utils.create_graphs(mat = sub_sample_cells, radius = args.radius, gene_indicator = column_indicator_gene,
                                                                 eval_indicator = column_indicator_eval_information,
                                                                 patient_id = patient_id, origin = origin,
                                                                batch_counter = batch_counter, save_path_graphs = save_path_graphs)
                            graph_name_list.append(file_name)
                            torch.cuda.empty_cache()
                            batch_counter += 1

        # save the graph names
        with open(os.path.join(folder_dir,'graph_name_list.pt'), 'wb') as f:
            pickle.dump(graph_name_list, f)

if __name__ == '__main__':
    create_data_set()