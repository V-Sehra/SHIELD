#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""
import torch
from torch_geometric.loader import DataListLoader

import argparse
import pickle
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from utils.data_class import diabetis_dataset
from model import ShIeLD
import utils.evaluation_utils as evaluation_utils




parser = argparse.ArgumentParser()
parser.add_argument("-req_path", "--requirements_file_path",
                    default=Path.cwd() / 'examples' / 'diabetes' / 'requirements.pt')
parser.add_argument("-retrain", "--retain_best_model_config_bool", default=True)
parser.add_argument("-config_dict", "--best_config_dict_path",
                    default=Path.cwd() / 'examples' / 'diabetes' / 'best_config.pt')
parser.add_argument("-dat_type", "--data_set_type", default='test')

args = parser.parse_args()
print(args)
with open(args.requirements_file_path, 'rb') as file:
    requirements = pickle.load(file)
with open(args.best_config_dict_path, 'rb') as file:
    best_config_dict = pickle.load(file)


path_to_graphs = Path(requirements['path_to_data_set'] /
                              f'anker_value_{best_config_dict["anker_value"]}'.replace('.', '_') /
                              f"min_cells_{requirements['minimum_number_cells']}" /
                              f"fussy_limit_{best_config_dict['fussy_limit']}".replace('.', '_') /
                              f"radius_{best_config_dict['radius_distance']}")


fold_ids = [requirements['number_validation_splits'] if args.data_set_type == 'train' else requirements['test_set_fold_number']]

data_loader = DataListLoader(
    diabetis_dataset(
        root=str(path_to_graphs / args.data_set_type / 'graphs'),
        path_to_graphs=path_to_graphs,
        fold_ids=fold_ids[0],
        requirements_dict=requirements,
        graph_file_names=f'{args.data_set_type}_set_file_names.pkl'
    ),
    batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
)
model = ShIeLD(
                num_of_feat=int(requirements['input_layer']),
                layer_1=requirements['layer_1'],
                layer_final=requirements['out_put_layer'],
                dp=best_config_dict['droup_out_rate'],
                self_att=False, attr_bool=requirements['attr_bool']
            ).to(device)

model.load_state_dict(torch.load(requirements['path_to_model'] / f'best_model_{best_config_dict["version"]}.pt'))
model.eval()

cell_to_cell_interaction_dict = evaluation_utils.get_cell_to_cell_interaction_dict(
        requirements_dict = requirements,
        data_loader = data_loader,
        model= model,
        device = device,
        save_dict_path = Path(requirements['path_to_model']/f'cT_t_cT_interactions_dict_{args.data_set_type}.pt'))


observed_tissues = list(requirements['label_dict'].keys())
for number_interactions in [4, len(requirements['cell_type_names'])]:
    print(f'creating the interactions for top {number_interactions}:')
    for observed_tissue in observed_tissues:

        interaction_dataFrame, mean_interaction_dataFrame = evaluation_utils.get_interaction_DataFrame(tissue_id= requirements['label_dict'][observed_tissue],
                                                                                                       interaction_dict = cell_to_cell_interaction_dict)


        top_connections = evaluation_utils.get_top_interaction_per_celltype(interaction_limit = number_interactions,
                                                         all_interaction_mean_df = mean_interaction_dataFrame,
                                                         all_interaction_df = interaction_dataFrame)

        Path(requirements['path_to_interaction_plots'] / 'boxplots').mkdir(parents=True, exist_ok=True)
        save_path_boxplots = Path(requirements['path_to_interaction_plots'] / 'boxplots'/ f'{observed_tissue}_topInteractions_{number_interactions}_{args.data_set_type}.png')

        evaluation_utils.plot_cell_cell_interaction_boxplots(significance_1= 0.0005, significance_2= 0.00005,
                                                interaction_limit = number_interactions,
                                                all_interaction_mean_df = interaction_dataFrame,
                                                top_connections = top_connections,
                                                save_path=save_path_boxplots)

