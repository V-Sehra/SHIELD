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

from utils.data_class import diabetis_dataset
import utils.train_utils as train_utils
import utils.model_utils as model_utils

from model import ShIeLD

torch.multiprocessing.set_start_method('fork', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-split", "--split_number", type=int, default=1)
    parser.add_argument("-rep", "--number_of_training_repeats", type=int, default=5)
    parser.add_argument("-comment", "--comment", type=str, default='random_anker')
    parser.add_argument("-comment_norm", "--comment_norm", type=str, default='noNorm')
    parser.add_argument("-req_path", "--requirements_file_path",
                        default=Path.cwd() / 'examples' / 'diabetes' / 'requirements.pt')

    args = parser.parse_args()
    print(args)

    with open(args.requirements_file_path, 'rb') as file:
        requirements = pickle.load(file)

    split_number = int(args.split_number)

    training_results_csv, csv_file_path = train_utils.get_train_results_csv(requirement_dict=requirements)
    meta_columns = ['anker_value', 'radius_neibourhood', 'fussy_limit',
                    'dp', 'comment', 'comment_norm', 'model_no','split_number']

    for radius_distance in requirements['radius_distance_all']:
        for fussy_limit in requirements['fussy_limit_all']:
            for anker_number in requirements['anker_value_all']:

                path_to_graphs = Path(requirements['path_to_data_set'] /
                                      f'anker_value_{anker_number}'.replace('.', '_') /
                                      f"min_cells_{requirements['minimum_number_cells']}" /
                                      f"fussy_limit_{fussy_limit}".replace('.', '_') /
                                      f'radius_{radius_distance}')


                torch.cuda.empty_cache()

                train_folds = requirements['number_validation_splits'].copy()
                train_folds.remove(split_number)


                data_loader_train = DataListLoader(
                    diabetis_dataset(
                        root = str(path_to_graphs / 'train' / 'graphs'),
                        path_to_graphs = path_to_graphs,
                        fold_ids = train_folds,
                        requirements_dict = requirements,
                        graph_file_names=f'train_set_validation_split_{split_number}_file_names.pkl'
                    ),
                    batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
                )

                data_loader_validation = DataListLoader(
                    diabetis_dataset(
                        root = str(path_to_graphs / 'train' / 'graphs'),
                        path_to_graphs = path_to_graphs,
                        fold_ids = [split_number],
                        requirements_dict=requirements,
                        graph_file_names= f'validation_validation_split_{split_number}_file_names.pkl'
                    ),
                    batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
                )

                for dp in requirements['droupout_rate']:
                    for num in tqdm(range(int(args.number_of_training_repeats))):

                        if not training_results_csv[meta_columns].isin(
                                [[anker_number, radius_distance, fussy_limit, dp, args.comment, args.comment_norm, num,split_number]]
                        ).all(axis=1).any():

                            loss_fkt = train_utils.initiaize_loss(
                                path=Path(Path(path_to_graphs /f'train_set_validation_split_{split_number}_file_names.pkl')),
                                tissue_dict=requirements['label_dict'],
                                device=device
                            )

                            model = ShIeLD(
                                num_of_feat=int(requirements['input_layer']),
                                layer_1 = requirements['layer_1'], dp=dp, layer_final = requirements['out_put_layer'],
                                self_att=False, attr_bool = requirements['attr_bool']
                            ).to(device)
                            model.train()

                            model, train_loss = train_utils.train_loop_shield(
                                optimizer=torch.optim.Adam(model.parameters(), lr=requirements['learning_rate']),
                                model=model,
                                data_loader=data_loader_train,
                                loss_fkt=loss_fkt,
                                attr_bool = requirements['attr_bool'],
                                device = device
                            )

                            train_bal_acc,train_f1_score = model_utils.get_acc_metrics(
                                model=model, data_loader=data_loader_train,
                                attr_bool = requirements['attr_bool'], device=device
                            )
                            print('start validation')
                            val_bal_acc, val_f1_score = model_utils.get_acc_metrics(
                                model=model, data_loader=data_loader_validation,
                                attr_bool = requirements['attr_bool'], device=device
                            )

                            model_csv = pd.DataFrame([[anker_number, radius_distance, fussy_limit,
                                                       dp, args.comment,args.comment_norm, num,
                                                       train_bal_acc,train_f1_score, val_bal_acc,val_f1_score,split_number]],
                                                     columns=['anker_value', 'radius_neibourhood', 'fussy_limit',
                                                              'dp', 'comment', 'comment_norm', 'model_no',
                                                              'bal_train_acc','train_f1_score',
                                                              'bal_val_acc','val_f1_score','split_number'])
                            training_results_csv = pd.concat([model_csv, training_results_csv], ignore_index=True)
                            training_results_csv.to_csv(csv_file_path, index=False)

                            torch.cuda.empty_cache()
                        else:
                            print('Model already trained:', anker_number, radius_distance, fussy_limit, dp, args.comment,
                                  args.comment_norm, num)

if __name__ == "__main__":
    main()
