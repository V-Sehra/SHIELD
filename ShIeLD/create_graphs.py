import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import multiprocessing as mp
import functools
from pathlib import Path
import pickle

import utils

parser = argparse.ArgumentParser()



# Define command-line arguments for input data paths
parser.add_argument("-req_path", "--requirements_file_path", default=Path.cwd() / 'diabetes' / 'requirements.pt')
parser.add_argument("-dat_type", "--data_set_type", default='train')

# Parse command-line arguments
args = parser.parse_args()
print(args)

# Load dataset requirements from a pickle file
requirements_file_path = args.requirements_file_path
data_set_type = args.data_set_type

with open(requirements_file_path, 'rb') as file:
    requirements = pickle.load(file)

# Extract key parameters from the requirements file
eval_columns = requirements['eval_columns']
label_dict = requirements['label_dict']
markers = requirements['markers']
radius_distance_all = requirements['radius_distance_all']
fussy_limit_all = requirements['fussy_limit_all']
prozent_of_anker_cells = requirements['prozent_of_anker_cells']
augmentation_number = requirements['augmentation_number']
number_voronoi_neighbours = requirements['voro_neighbours']
minimum_number_cells = requirements['minimum_number_cells']

# Determine the correct fold column based on dataset type
if data_set_type == 'train':
    fold_column = requirements['validation_split_column']
else:
    fold_column = requirements['test_set_fold_number']

# Load the raw dataset from CSV
input_data = pd.read_csv(requirements['path_raw_data'])

# Filter the dataset based on the validation split column
data_sample = input_data.loc[input_data[requirements['validation_split_column']].isin(fold_column)]
data_sample.reset_index(inplace=True, drop=True)

# Get unique sample names
sample = data_sample[requirements['measument_sample_name']].unique()

# Iterate over each sample in the dataset
for sub_sample in tqdm(sample):

    # Filter data for the current image sample
    single_image = data_sample[data_sample['image'] == sub_sample]

    # Iterate over different percentages of anchor cells
    for anker_prozent in prozent_of_anker_cells:

        # Iterate over different fussy limits
        for fussy_limit in fussy_limit_all:

            repeat_counter = 0  # Track the repeat count for unique graph file naming

            for augment_id in range(augmentation_number):

                # Randomly select anchor cells based on the percentage
                anchors = single_image.sample(n=int(len(single_image) * anker_prozent))

                # Compute Voronoi regions with fuzzy boundary constraints
                voroni_id_fussy = utils.data_utils.get_voronoi_id(
                    data_set=single_image,
                    anker_cell=anchors,
                    boarder_number=number_voronoi_neighbours,
                    fussy_limit=fussy_limit
                )

                # Count the number of cells in each Voronoi region
                number_cells = np.array([len(arr) for arr in voroni_id_fussy])

                # If any region has too few cells, filter out those regions and recompute Voronoi
                if any(number_cells < minimum_number_cells):
                    voroni_id_fussy = utils.data_utils.get_voronoi_id(
                        data_set=single_image,
                        anker_cell=anchors[number_cells > minimum_number_cells],
                        boarder_number=number_voronoi_neighbours,
                        fussy_limit=fussy_limit
                    )

                # Create an array of Voronoi region indices
                vornoi_id = np.arange(0, len(voroni_id_fussy))

                # Iterate over different neighborhood radius values
                for radius_distance in radius_distance_all:

                    # Define the folder structure for saving graphs
                    save_path = Path(requirements['path_graphs'] /
                                     f'anker_value_{anker_prozent}'.replace('.', '_') /
                                     f'min_cells_{minimum_number_cells}' /
                                     f'fussy_limit_{fussy_limit}'.replace('.', '_') /
                                     f'radius_{radius_distance}')

                    save_path_folder_graphs = save_path / f'{data_set_type}' / 'graphs'

                    # Ensure the directory exists
                    os.system(f'mkdir -p {save_path_folder_graphs}')

                    # Print status updates
                    print('anker_prozent', 'radius_neibourhood', 'fussy_limit', 'image')
                    print(anker_prozent, radius_distance, fussy_limit, sub_sample)
                    print(save_path_folder_graphs)

                    # Use multiprocessing to create and save graphs in parallel
                    pool = mp.Pool(mp.cpu_count() - 2)
                    graphs = pool.map(
                        functools.partial(
                            utils.data_utils.create_graph_and_save,
                            whole_data=single_image,
                            gene_col_name=markers,
                            eval_col_name=eval_columns,
                            save_path_folder=save_path_folder_graphs,
                            radius_neibourhood=radius_distance,
                            tissue_dict=label_dict,
                            voronoi_list=voroni_id_fussy,
                            sub_sample=sub_sample,
                            repeat_id=repeat_counter
                        ), vornoi_id
                    )
                    pool.close()

                # Update repeat counter for unique graph IDs
                repeat_counter += len(voroni_id_fussy)
                print(repeat_counter)


