import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import Data
import random as rnd
import model_utils

cwd = os.getcwd()
idx_folder = cwd.index('ShIeLD') + len('ShIeLD')
shield_dir = os.path.join(f'{cwd[:idx_folder]}')

tissue_type_name = ['normalLiver', 'core', 'rim']
tissue_dict = {'normalLiver': 0,
               'core': 1,
               'rim': 2}


def bool_passer(argument):
    if argument == 'True':
        value = True
    else:
        value = False
    return (value)

def combine_cell_types(original_array, string_list):
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

    for substring_to_replace in string_list:
        # Find indices where the substring occurs
        indices = np.core.defchararray.find(original_array.astype(str), substring_to_replace)

        # Replace words containing the substring with 'substring_to_replace'
        original_array[indices != -1] = substring_to_replace

        unique_elements, unique_indices, inverse_indices = np.unique(original_array, return_index=True, return_inverse=True)

        # Create a new array without duplicates
        original_array = unique_elements

    return original_array

def create_graphs(mat, radius, gene_indicator, eval_indicator, patient_id, origin, batch_counter, save_path_graphs):
    """
    This function creates a graph from a sample of cells.
    It calculates the real coordinates, edge matrix, and attention matrix, and selects the desired features for the nodes.
    It then creates a Data object and saves it as a graph.

    Parameters:
    mat (pd.DataFrame): The sample of cells.
    radius (float): The radius to use for calculating the edge matrix.
    gene_indicator (list): A list of column names indicating the gene intensities.
    eval_indicator (list): A list of column names indicating the evaluation information.
    patient_id (str): The ID of the patient.
    origin (str): The tissue type.
    batch_counter (int): The batch counter.
    save_path_graphs (str): The path to save the graphs.

    Returns:
    str: The name of the created graph.
    """

    sub_sample_cells = mat

    real_cordinates = sub_sample_cells[['Y_value', 'X_value']]
    plate_edge, plat_att = model_utils.calc_edge_mat(real_cordinates,
                                                     dist_bool=True,
                                                     radius=radius)

    # remove all "overlapping" cells
    plate_edge = model_utils.remove_zero_distances(edge=plate_edge,
                                                   dist=plat_att,
                                                   return_dist=False)

    # select the desired features for the nodes
    cells = torch.tensor((sub_sample_cells.loc[:, gene_indicator]).to_numpy()).float()

    data = Data(x=cells,

                edge_index_plate=torch.tensor(plate_edge).long(),
                orginal_cord=torch.tensor(real_cordinates.to_numpy()),

                eval=sub_sample_cells.loc[:, eval_indicator]
                .to_numpy(dtype=np.dtype('str')),

                eval_col_names=sub_sample_cells.loc[:, eval_indicator]
                .columns.to_numpy(dtype=np.dtype('str')),

                ids=patient_id,

                y=torch.tensor(tissue_dict[origin]).flatten()).cpu()

    torch.save(data, os.path.join(f'{save_path_graphs}',
                                  f'graph_pat_{patient_id}_{origin}_{batch_counter}.pt'))

    return f'graph_pat_{patient_id}_{origin}_{batch_counter}.pt'

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
    patent_list = raw_data['Patient'].unique()

    # If the 'new_test_split_bool' argument is True, shuffle the patient list and split it into training and testing sets
    # Otherwise, use a predefined list of patient IDs for the testing set
    if bool_passer(args.new_test_split_bool):
        rnd.shuffle(patent_list)
        train_patients = patent_list[0:int(len(patent_list) * 0.8)]
        test_patients = patent_list[int(len(patent_list) * 0.8):]

        train_csv = raw_data[raw_data['Patient'].isin(train_patients)]
        test_csv = raw_data[raw_data['Patient'].isin(test_patients)]
    else:
        test_patient_ids = ['LHCC51', 'LHCC53', 'Pat53', 'LHCC45', 'Pat52']
        train_csv = raw_data[~raw_data['Patient'].isin(test_patient_ids)]
        test_csv = raw_data[raw_data['Patient'].isin(test_patient_ids)]

    # Create directories for saving the training and testing sets
    os.system(f'mkdir -p {args.path_to_save_data}/train_set')
    os.system(f'mkdir -p {args.path_to_save_data}/test_set')

    # Save the training and testing sets as CSV files
    train_csv.to_csv(os.path.join(f'{args.path_to_save_data}', 'train_set', 'train_cells.csv'))
    test_csv.to_csv(os.path.join(f'{args.path_to_save_data}', 'test_set', 'test_cells.csv'))

    return


def fill_missing_row_and_col_withNaN(data_frame, cell_types_names):
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
    data_frame = pd.concat([data_frame,
                            pd.DataFrame(np.full((len(missing_rows), data_frame.shape[1]), np.nan),
                                         columns=data_frame.columns,
                                         index=cell_types_names[~np.isin(cell_types_names, data_frame.index)])])

    # Sort the DataFrame by index and columns
    data_frame = data_frame.sort_index()[sorted(data_frame.columns)]

    return data_frame


def get_tissue_type_name(tissue_type_id):
    return tissue_type_name[tissue_type_id]

def get_tissue_type_id(tissue_type_name):
    return tissue_dict[tissue_type_name]



def turn_pixel_to_meter(pixel_radius):
    pixel_to_miliMeter_factor = 2649.291339
    mycro_meter_radius = pixel_radius * (10**3/pixel_to_miliMeter_factor)
    return round(mycro_meter_radius)


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
    raw_csv['X_value'] = (raw_csv['XMax'] + raw_csv['XMin']) / 2
    raw_csv['Y_value'] = (raw_csv['YMax'] + raw_csv['YMin']) / 2

    # Filter the columns of the DataFrame to only include those that contain certain substrings
    raw_csv = raw_csv.loc[:,raw_csv.columns.str.contains('Tissue|Patient|X_value|Y_value|.Intensity|Class|Class0|CD45.Positive.Classification|Celltype',
                                                           na = False)]
    return raw_csv


def replace_celltype(mat,celltype_list_to_replace):
    # Substring to find
    for substring_to_replace in celltype_list_to_replace:
        # Replace strings containing the substring
        for i, row in enumerate(mat):
            for j,word in enumerate(row):
                if substring_to_replace in word:
                    mat[i][j] = substring_to_replace

    return mat



def path_generator(path_to_graps, path_save_model, path_org_csv_file, path_name_list, data_type = 'train'):
    """
    This function generates and returns paths for graphs, model, original CSV file, and name list.
    If a path is not provided, it sets the path to the current working directory.

    Parameters:
    path_to_graps (str): The path to the input graphs. If None, it will be set to 'graphs' in the current working directory.
    path_save_model (str): The path to save the model. If None, it will be set to 'model' in the current working directory.
    path_org_csv_file (str): The path to the original CSV file. If None, it will be set to 'org_data/org_data.csv' in the current working directory.
    path_name_list (str): The path to the name list. If None, it will be set to 'graph_name_list.pt' in the current working directory.
    cwd (str): The current working directory.

    Returns:
    tuple: A tuple containing the paths to the graphs, model, original CSV file, and name list.
    """

    # If no path is provided for graphs, set it to the 'graphs' directory in the current working directory
    if path_to_graps is None:
        path_to_graps = os.path.join(f'{shield_dir}','data',f'{data_type}_set','graphs')
        os.system(f"mkdir -p {path_to_graps}")

    # If no path is provided for the model, set it to the 'model' directory in the current working directory
    if path_save_model is None:
        path_save_model = os.path.join(f'{shield_dir}', 'model')
        os.system(f"mkdir -p {path_save_model}")
    else:
        path_save_model = path_to_graps

    # If no path is provided for the original CSV file, set it to 'org_data/org_data.csv' in the current working directory
    if path_org_csv_file is None:
        path_org_csv_file = os.path.join(f'{shield_dir}','data',f'{data_type}_set',f'{data_type}_cells.csv')

    # If no path is provided for the name list, set it to 'graph_name_list.pt' in the current working directory
    if path_name_list is None:
        path_name_list = os.path.join(f'{shield_dir}','data',f'{data_type}_set', 'graph_name_list.pt')

    # Return the paths
    return path_to_graps, path_save_model, path_org_csv_file, path_name_list