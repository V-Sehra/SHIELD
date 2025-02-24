#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import multiprocessing as mp
import functools
from pathlib import Path
import pickle


from utils import data_utils


# Main function to process dataset and generate graphs
def main():
    # Define command-line arguments for input data paths
    parser = argparse.ArgumentParser()
    parser.add_argument("-req_path", "--requirements_file_path",
                        default=Path.cwd() / 'examples' / 'diabetes' / 'requirements.pt')
    parser.add_argument("-dat_type", "--data_set_type", default='train')

    # Parse command-line arguments
    args = parser.parse_args()
    print(args)

    # Load dataset requirements from a pickle file
    with open(args.requirements_file_path, 'rb') as file:
        requirements = pickle.load(file)



    # Determine the correct fold column based on dataset type
    if args.data_set_type == 'train':
        fold_column = requirements['number_validation_splits']
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
        single_sample = data_sample[data_sample['image'] == sub_sample]

        for anker_value in requirements['anker_value_all']:
            for fussy_limit in requirements['fussy_limit_all']:
                repeat_counter = 0  # Track repeat count for unique graph file naming

                for augment_id in range(requirements['augmentation_number']):
                    # Randomly select anchor cells based on either the percentage or absulut value
                    if requirements['anker_cell_selction_type'] == '%':
                        anchors = single_sample.sample(n=int(len(single_sample) * anker_value))
                    elif requirements['anker_cell_selction_type'] == 'absolut':
                        anchors = single_sample.sample(n=int(anker_value))

                    # Compute Voronoi regions with fuzzy boundary constraints

                    voroni_id_fussy = data_utils.get_voronoi_id(
                        data_set=single_sample,
                        anker_cell=anchors,
                        requiremets_dict=requirements,
                        fussy_limit=fussy_limit
                    )

                    # Count number of cells in each Voronoi region
                    number_cells = np.array([len(arr) for arr in voroni_id_fussy])

                    # If any region has too few cells, filter out those regions and recompute Voronoi
                    if any(number_cells < requirements['minimum_number_cells']):
                        voroni_id_fussy = data_utils.get_voronoi_id(
                            data_set=single_sample,
                            anker_cell=anchors[number_cells > requirements['minimum_number_cells']],
                            requiremets_dict=requirements,
                            fussy_limit=fussy_limit
                        )

                    # Create an array of Voronoi region indices
                    vornoi_id = np.arange(0, len(voroni_id_fussy))

                    for radius_distance in requirements['radius_distance_all']:
                        save_path = Path(requirements['path_to_data_set'] /
                                         f'anker_value_{anker_value}'.replace('.', '_') /
                                         f"min_cells_{requirements['minimum_number_cells']}" /
                                         f'fussy_limit_{fussy_limit}'.replace('.', '_') /
                                         f'radius_{radius_distance}')

                        save_path_folder_graphs = save_path / f'{args.data_set_type}' / 'graphs'

                        # Ensure the directory exists
                        os.system(f'mkdir -p {save_path_folder_graphs}')

                        # Print status updates
                        print('anker_value', 'radius_neibourhood', 'fussy_limit', 'image')
                        print(anker_value, radius_distance, fussy_limit, sub_sample)

                        # Use multiprocessing to create and save graphs in parallel
                        pool = mp.Pool(mp.cpu_count() - 2)
                        graphs = pool.map(
                            functools.partial(
                                data_utils.create_graph_and_save,
                                whole_data=single_sample,
                                save_path_folder=save_path_folder_graphs,
                                radius_neibourhood=radius_distance,
                                requiremets_dict=requirements,
                                voronoi_list=voroni_id_fussy,
                                sub_sample=sub_sample,
                                repeat_id=repeat_counter
                            ), vornoi_id
                        )
                        pool.close()

                    # Update repeat counter for unique graph IDs
                    repeat_counter += len(voroni_id_fussy)



# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()