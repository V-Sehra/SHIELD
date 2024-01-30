import pandas as pd
import numpy as np
import os

tissue_type_name = ['normalLiver', 'core', 'rim']
tissue_dict = {'normalLiver': 0,
               'core': 1,
               'rim': 2}

def get_tissue_type_name(tissue_type_id):
    return tissue_type_name[tissue_type_id]

def get_tissue_type_id(tissue_type_name):
    return tissue_dict[tissue_type_name]

def turn_pixel_to_meter(pixel_radius):
    pixel_to_miliMeter_factor = 2649.291339
    mycro_meter_radius = pixel_radius * (10**3/pixel_to_miliMeter_factor)
    return(round(mycro_meter_radius))

def replace_celltype(mat,celltype_list_to_replace):
    # Substring to find
    for substring_to_replace in celltype_list_to_replace:
        # Replace strings containing the substring
        for i, row in enumerate(mat):
            for j,word in enumerate(row):
                if substring_to_replace in word:
                    mat[i][j] = substring_to_replace

    return mat

def combine_cell_types(original_array,string_list):
    for substring_to_replace in string_list:
        # Find indices where the substring occurs
        indices = np.core.defchararray.find(original_array.astype(str), substring_to_replace)

        # Replace words containing the substring with 'substring_to_replace'
        original_array[indices != -1] = substring_to_replace

        unique_elements, unique_indices, inverse_indices = np.unique(original_array, return_index=True, return_inverse=True)

        # Create a new array without duplicates
        original_array = unique_elements

    return(original_array)


def fill_missing_row_and_col_withNaN(data_frame, cell_types_names):
    missing_cols = cell_types_names[~np.isin(cell_types_names, data_frame.columns)]
    data_frame[missing_cols] = np.full((len(data_frame), len(missing_cols)), np.nan)

    missing_rows = cell_types_names[~np.isin(cell_types_names, data_frame.index)]

    data_frame = pd.concat([data_frame,
                            pd.DataFrame(np.full((len(missing_rows), data_frame.shape[1]), np.nan),
                                         columns=data_frame.columns,
                                         index=cell_types_names[~np.isin(cell_types_names, data_frame.index)])])

    data_frame = data_frame.sort_index()[sorted(data_frame.columns)]
    return (data_frame)

def path_generator(path_to_graps,path_save_model, path_org_csv_file, path_name_list, cwd):
    # if no path is provided, the path will be set to the current working directory's
    if path_to_graps is None:
        path_to_graps = os.path.join(f'{cwd}', 'graphs')
        os.system(f"mkdir -p {path_to_graps}")
    else:
        path_to_graps = path_to_graps

    if path_save_model is None:
        path_save_model = os.path.join(f'{cwd}', 'model')
        os.system(f"mkdir -p {path_save_model}")
    else:
        path_save_model = path_to_graps

    if path_org_csv_file is None:
        path_org_csv_file = os.path.join(f'{cwd}','org_data','org_data.csv')

    else:
        path_org_csv_file = path_org_csv_file

    if path_name_list is None:
        path_name_list = os.path.join(f'{cwd}', 'graph_name_list.pt')
    else:
        path_name_list = path_name_list

    return(path_to_graps,path_save_model, path_org_csv_file, path_name_list)