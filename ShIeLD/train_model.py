#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""
import argparse
import pickle
from pathlib import Path
import torch
import pandas as pd

from tqdm import tqdm

from torch_geometric.loader import DataListLoader
from utils.data_class import graph_dataset
import utils.train_utils as train_utils
import utils.model_utils as model_utils

from tests import input_test

from model import ShIeLD

torch.multiprocessing.set_start_method('fork', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-split", "--split_number", type=int, default=2)
    parser.add_argument("-rep", "--number_of_training_repeats", type=int, default=5)
    parser.add_argument("-c", "--comment", type=str, default='noVoro')
    parser.add_argument("-req_path", "--requirements_file_path",
                        default=Path.cwd() / 'examples' / 'CRC' / 'requirements.pt')
    parser.add_argument("-noisy_edge", "--noisy_edge",
                        default=False)
    parser.add_argument("-noise_yLabel", "--noise_yLabel",
                        default=False)
    parser.add_argument("-rev", "--reverse_sampling", default=False, type=bool, choices=[True, False])

    args = parser.parse_args()

    requirements = pickle.load(open(Path.cwd() / args.requirements_file_path, 'rb'))
    input_test.test_all_keys_in_req(req_file=requirements)

    print(args)

    # extract training patience if defined:
    if 'patience' in requirements.keys():
        patience = requirements['patience']
    else:
        patience = 5

    # if there where no voronois constructed the fussylimit makes no sence
    if 'sampleing' in requirements.keys():
        if requirements['sampleing'] == 'random':
            fussy_vector = ['randomSampling']
            fussy_folder_name = 'random_sampling'
        elif requirements['sampleing'] == 'voronoi':
            fussy_folder_name = True
            fussy_vector = ['fussy_limit_all']
    else:
        fussy_vector = ['fussy_limit_all']
        fussy_folder_name = True

    split_number = int(args.split_number)

    training_results_csv, csv_file_path = train_utils.get_train_results_csv(requirement_dict=requirements)

    meta_columns = ['anker_value', 'radius_distance', 'fussy_limit',
                    'droupout_rate', 'comment', 'comment_norm', 'model_no', 'split_number']

    radii = requirements['radius_distance_all']
    if args.reverse_sampling:
        radii = radii[::-1]

    for radius_distance in radii:
        for fussy_limit in fussy_vector:
            for anker_number in requirements['anker_value_all']:

                if fussy_folder_name is True:
                    path_to_graphs = Path(requirements['path_to_data_set'] /
                                          f'anker_value_{anker_number}'.replace('.', '_') /
                                          f"min_cells_{requirements['minimum_number_cells']}" /
                                          f"fussy_limit_{fussy_limit}".replace('.', '_') /
                                          f'radius_{radius_distance}')
                else:
                    path_to_graphs = Path(requirements['path_to_data_set'] /
                                          f'anker_value_{anker_number}'.replace('.', '_') /
                                          f"min_cells_{requirements['minimum_number_cells']}" /
                                          f'{fussy_folder_name}' /
                                          f'radius_{radius_distance}')

                torch.cuda.empty_cache()

                train_folds = requirements['number_validation_splits'].copy()
                train_folds.remove(split_number)

                if requirements['databased_norm'] is not None:
                    databased_norm = requirements['databased_norm']
                    file_name_data_norm = f'train_set_validation_split_{split_number}_standadizer.pkl'
                else:
                    databased_norm = None
                    file_name_data_norm = None

                data_loader_train = DataListLoader(
                    graph_dataset(
                        root=str(path_to_graphs / 'train' / 'graphs'),
                        path_to_graphs=path_to_graphs,
                        fold_ids=train_folds,
                        requirements_dict=requirements,
                        graph_file_names=f'train_set_validation_split_{split_number}_file_names.pkl',
                        normalize=databased_norm,
                        normalizer_filename=file_name_data_norm
                    ),
                    batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
                )

                data_loader_validation = DataListLoader(
                    graph_dataset(
                        root=str(path_to_graphs / 'train' / 'graphs'),
                        path_to_graphs=path_to_graphs,
                        fold_ids=[split_number],
                        requirements_dict=requirements,
                        graph_file_names=f'validation_validation_split_{split_number}_file_names.pkl',
                        normalize=databased_norm,
                        normalizer_filename=file_name_data_norm
                    ),
                    batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
                )

                for dp in requirements['droupout_rate']:
                    for num in tqdm(range(int(args.number_of_training_repeats))):

                        row_values = [anker_number, radius_distance, fussy_limit, dp, args.comment,
                                      requirements['comment_norm'], num, split_number]

                        filter_match = (training_results_csv[meta_columns] == row_values).all(axis=1)

                        if not filter_match.any():

                            loss_init_path = path_to_graphs / 'train' / 'graphs' if args.noise_yLabel is not False else path_to_graphs / f'train_set_validation_split_{split_number}_file_names.pkl'
                            loss_fkt = train_utils.initiaize_loss(
                                path=Path(loss_init_path),
                                tissue_dict=requirements['label_dict'],
                                device=device,
                                noise_yLabel=args.noise_yLabel
                            )

                            model = ShIeLD(
                                num_of_feat=int(requirements['input_layer']),
                                layer_1=requirements['layer_1'],
                                layer_final=requirements['output_layer'],
                                dp=dp,
                                self_att=False, attr_bool=requirements['attr_bool'],
                                norm_type=requirements['comment_norm'],
                                noisy_edge=args.noisy_edge
                            ).to(device)

                            model.train()

                            model, train_loss = train_utils.train_loop_shield(
                                optimizer=torch.optim.Adam(model.parameters(), lr=requirements['learning_rate']),
                                model=model,
                                data_loader=data_loader_train,
                                loss_fkt=loss_fkt,
                                attr_bool=requirements['attr_bool'],
                                device=device,
                                patience=patience,
                                noise_yLabel=args.noise_yLabel
                            )

                            model.eval()
                            print('start validation')
                            train_bal_acc, train_f1_score, train_cm = model_utils.get_acc_metrics(
                                model=model, data_loader=data_loader_train, device=device
                            )

                            val_bal_acc, val_f1_score, test_cm = model_utils.get_acc_metrics(
                                model=model, data_loader=data_loader_validation, device=device
                            )
                            model_csv = pd.DataFrame([[anker_number, radius_distance, fussy_limit,
                                                       dp, args.comment, requirements['comment_norm'], num,
                                                       train_bal_acc, train_f1_score, val_bal_acc, val_f1_score,
                                                       split_number]],
                                                     columns=training_results_csv.columns)

                            print('train_bal_acc', 'train_f1_score')
                            print(train_bal_acc, train_f1_score)
                            print('val_bal_acc', 'val_f1_score')
                            print(val_bal_acc, val_f1_score)

                            training_results_csv, csv_file_path = train_utils.get_train_results_csv(
                                requirement_dict=requirements)

                            training_results_csv = pd.concat([model_csv, training_results_csv], ignore_index=True)
                            training_results_csv.to_csv(csv_file_path, index=False)

                            torch.cuda.empty_cache()
                        else:
                            print('Model already trained:', anker_number, radius_distance, fussy_limit, dp,
                                  args.comment,
                                  requirements['comment_norm'], num)


if __name__ == "__main__":
    main()
