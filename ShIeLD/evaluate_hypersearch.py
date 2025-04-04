#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.loader import DataListLoader

from model import ShIeLD
import utils.evaluation_utils as evaluation_utils
import utils.data_utils as data_utils
import utils.train_utils as train_utils
from utils.data_class import graph_dataset

torch.multiprocessing.set_start_method('fork', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-req_path", "--requirements_file_path",
                        default=Path.cwd() / 'examples' / 'HCC' / 'requirements.pt')
    parser.add_argument("-retrain", "--retain_best_model_config_bool", default=True)
    parser.add_argument("-config_dict", "--best_config_dict_path",
                        default=Path.cwd() / 'examples' / 'HCC' / 'best_config.pt')
    parser.add_argument("-rep", "--number_of_training_repeats", type=int, default=5)

    args = parser.parse_args()
    print(args)

    with open(args.requirements_file_path, 'rb') as file:
        requirements = pickle.load(file)


    print('evaluating the training results')
    hyper_search_results = evaluation_utils.get_hypersear_results(requirements_dict = requirements)

    hyper_search_results.to_csv(Path(requirements['path_training_results'] / 'hyper_search_results.csv'), index=False)

    melted_results = hyper_search_results.melt(id_vars=['total_acc_balanced_mean'], var_name='hyperparameter')


    save_path_folder = Path(requirements['path_training_results'] / 'hyper_search_plots')
    save_path_folder.mkdir(parents=True, exist_ok=True)

    print('creating hyperparameter search plots')

    for observable_of_interest in requirements['col_of_variables']:

        evaluation_utils.create_parameter_influence_plots(df = melted_results,
                                                          observed_variable = observable_of_interest,
                                                          save_path=save_path_folder / f'{observable_of_interest}.png')

    if data_utils.bool_passer(args.retain_best_model_config_bool):
        if args.best_config_dict_path.exists():
            with open(args.best_config_dict_path, 'rb') as file:
                best_config_dict = pickle.load(file)
        else:

            best_config_dict = {'layer_one' : 38,
                    'input_dim' : 38,
                    'droup_out_rate' : hyper_search_results['dp'].iloc[0],
                    'final_layer' : 3,
                    'attr_bool' : False,
                    'anker_value': hyper_search_results['anker_value'].iloc[0],
                    'radius_distance': hyper_search_results['radius_distance'].iloc[0],
                    'fussy_limit': hyper_search_results['fussy_limit'].iloc[0]}

            with open(args.best_config_dict_path, 'wb') as file:
                pickle.dump(best_config_dict, file)


        print('best configuration:')
        print(best_config_dict)
        path_to_graphs = Path(requirements['path_to_data_set'] /
                              f'anker_value_{best_config_dict["anker_value"]}'.replace('.', '_') /
                              f"min_cells_{requirements['minimum_number_cells']}" /
                              f"fussy_limit_{best_config_dict['fussy_limit']}".replace('.', '_') /
                              f"radius_{best_config_dict['radius_distance']}")

        train_graph_path = path_to_graphs / 'train' / 'graphs'

        data_loader_train = DataListLoader(
            graph_dataset(
                root=str(train_graph_path),
                path_to_graphs=path_to_graphs,
                fold_ids=requirements['number_validation_splits'],
                requirements_dict=requirements,
                graph_file_names=f'train_set_file_names.pkl'
            ),
            batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
        )
        for num in tqdm(range(args.number_of_training_repeats)):

            loss_fkt = train_utils.initiaize_loss(
                path=Path(path_to_graphs /f'train_set_file_names.pkl'),
                tissue_dict=requirements['label_dict'],
                device = device)

            model = ShIeLD(
                num_of_feat=int(requirements['input_layer']),
                layer_1=requirements['layer_1'],
                layer_final=requirements['out_put_layer'],
                dp=best_config_dict['droup_out_rate'],
                self_att=False, attr_bool=requirements['attr_bool']
            ).to(device)
            model.train()

            model, train_loss = train_utils.train_loop_shield(
                optimizer=torch.optim.Adam(model.parameters(), lr=requirements['learning_rate']),
                model=model,
                data_loader=data_loader_train,
                loss_fkt=loss_fkt,
                attr_bool=requirements['attr_bool'],
                device=device)

            requirements['path_to_model'].mkdir(parents=True, exist_ok=True)
            model_save_path = requirements['path_to_model'] / f'best_model_{num}.pt'
            torch.save(model.state_dict(), model_save_path)




if __name__ == "__main__":
    main()
