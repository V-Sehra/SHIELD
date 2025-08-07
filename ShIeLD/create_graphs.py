#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import multiprocessing as mp
import functools
from pathlib import Path
import pickle

from utils import data_utils
from tests import input_test


# Main function to process dataset and generate graphs
def main():
    # Define command-line arguments for input data paths
    parser = argparse.ArgumentParser()
    parser.add_argument("-req_path", "--requirements_file_path",
                        default=Path.cwd() / 'examples' / 'CRC' / 'requirements.pt')
    parser.add_argument("-dat_type", "--data_set_type", default='test')

    # benchmarking parameters
    parser.add_argument("-n_label", "--noisy_labeling", default=False)
    parser.add_argument("-node_prob", "--node_prob", default=False)
    parser.add_argument("-rnd_edges", "--randomise_edges", default=False)
    parser.add_argument("-p_edges", "--percent_number_cells", type=float, default=0.1)

    parser.add_argument("-segmentation", "--segmentation", type=str, default='voronoi',
                        choices=['random', 'voronoi'])

    parser.add_argument("-downSample", "--reduce_population", default=False)
    parser.add_argument("-column_celltype_name", "--column_celltype_name", default='Class0')
    parser.add_argument("-rev", "--reverse_sampling", default=False)

    # Parse command-line arguments
    args = parser.parse_args()
    args.n_label = data_utils.bool_passer(args.n_label)
    args.node_prob = data_utils.bool_passer(args.node_prob)
    args.randomise_edges = data_utils.bool_passer(args.randomise_edges)
    args.reduce_population = data_utils.bool_passer(args.reduce_population)
    args.reverse_sampling = data_utils.bool_passer(args.reverse_sampling)
    
    print(args)

    # Load dataset requirements from a pickle file
    requirements = pickle.load(open(args.requirements_file_path, 'rb'))

    input_test.test_all_keys_in_req(req_file=requirements)
    # Determine the correct fold column based on dataset type
    fold_ids = [requirements['number_validation_splits'] if args.data_set_type == 'train' else requirements[
        'test_set_fold_number']][0]

    # Load the raw dataset from CSV
    input_data = pd.read_csv(requirements['path_raw_data'])

    if requirements['filter_cells'] != False:
        input_data = input_data[input_data[requirements['filter_column'][0]] == requirements['filter_value']]

    # Filter the dataset based on the validation split column
    data_sample = input_data.loc[input_data[requirements['validation_split_column']].isin(fold_ids)]
    data_sample.reset_index(inplace=True, drop=True)

    # Get unique sample names
    sample = data_sample[requirements['measument_sample_name']].unique()
    if args.reverse_sampling:
        sample = sample[::-1]

    # Iterate over each sample in the dataset
    for sub_sample in tqdm(sample):
        single_sample = data_sample[data_sample[requirements['measument_sample_name']] == sub_sample]

        for anker_value in requirements['anker_value_all']:

            # If downsampling is enabled, reduce the population of the sample
            if args.reduce_population is not False:
                # Check if the column for cell type exists
                if args.column_celltype_name not in single_sample.columns:
                    raise ValueError(f"Column '{args.column_celltype_name}' not found in the dataset.")
                # Ensure the downsample ratio is specified in the requirements
                if 'downSampeling' not in requirements:
                    raise ValueError("Downsampling ratio 'downSampeling' not specified in requirements.")
                downsampleRatio = requirements['downSampeling']

                single_sample = data_utils.reducePopulation(df=single_sample,
                                                            columnName=args.column_celltype_name,
                                                            cellTypeName=args.reduce_population,
                                                            downsampleRatio=downsampleRatio)

            for augment_id in range(requirements['augmentation_number']):

                # If one want to baseline the method selection from the Voronoi to random sampleing
                # change the segmentation to random
                if args.segmentation == 'voronoi':
                    for fussy_limit in requirements['fussy_limit_all']:

                        # Randomly select anchor cells based on either the percentage or absulut value
                        if requirements['anker_cell_selction_type'] == '%':
                            anchors = single_sample.sample(n=int(len(single_sample) * anker_value))

                        elif requirements['anker_cell_selction_type'] == 'absolut':
                            if requirements['multiple_labels_per_subSample']:
                                samples_per_tissue = anker_value // len(requirements['label_dict'].keys())
                                anchors = pd.DataFrame()
                                for label_dict_key in requirements['label_dict'].keys():
                                    anchors = pd.concat([anchors, single_sample[
                                        single_sample[requirements['label_column']] == label_dict_key].sample(
                                        n=samples_per_tissue)])
                            else:
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
                            save_path_folder_graphs.mkdir(parents=True, exist_ok=True)

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
                                    repeat_id=augment_id,
                                    skip_existing=True,
                                    noisy_labeling=args.noisy_labeling,
                                    node_prob=args.node_prob,
                                    randomise_edges=args.randomise_edges,
                                    percent_number_cells=args.percent_number_cells,
                                    segmentation=args.segmentation
                                ), vornoi_id
                            )
                            pool.close()


                elif args.segmentation == 'random':

                    if requirements['multiple_labels_per_subSample']:
                        samples_per_tissue = anker_value // len(requirements['label_dict'].keys())
                        sample_collection = []
                        for label_dict_key in requirements['label_dict'].keys():
                            sample_collection.extend(np.array_split(
                                single_sample[single_sample[requirements['label_column']] == label_dict_key].sample(
                                    frac=1),
                                samples_per_tissue))

                    else:

                        n_chunks = int(int(len(single_sample) * anker_value))
                        sample_collection = np.array_split(single_sample.sample(frac=1, random_state=42), n_chunks)

                    # the function create_graph_and_save will then select the dataFrames form the list
                    subsection_id = np.arange(0, len(sample_collection))

                    voroni_id_fussy = None

                    for radius_distance in requirements['radius_distance_all']:
                        save_path = Path(requirements['path_to_data_set'] /
                                         f'anker_value_{anker_value}'.replace('.', '_') /
                                         f"min_cells_{requirements['minimum_number_cells']}" /
                                         'random_sampling' /
                                         f'radius_{radius_distance}')

                        save_path_folder_graphs = save_path / f'{args.data_set_type}' / 'graphs'

                        # Ensure the directory exists
                        save_path_folder_graphs.mkdir(parents=True, exist_ok=True)

                        # Print status updates
                        print('anker_value', 'radius_neibourhood', 'image')
                        print(anker_value, radius_distance, sub_sample)

                        # Use multiprocessing to create and save graphs in parallel
                        pool = mp.Pool(mp.cpu_count() - 2)
                        graphs = pool.map(
                            functools.partial(
                                data_utils.create_graph_and_save,
                                whole_data=sample_collection,
                                save_path_folder=save_path_folder_graphs,
                                radius_neibourhood=radius_distance,
                                requiremets_dict=requirements,
                                voronoi_list=voroni_id_fussy,
                                sub_sample=sub_sample,
                                repeat_id=augment_id,
                                skip_existing=True,
                                noisy_labeling=args.noisy_labeling,
                                node_prob=args.node_prob,
                                randomise_edges=args.randomise_edges,
                                percent_number_cells=args.percent_number_cells,
                                segmentation=args.segmentation
                            ), subsection_id
                        )
                        pool.close()


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
