
import torch
from torch_geometric.loader import DataListLoader



import argparse
import pickle
from pathlib import Path
from itertools import compress
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import matplotlib.pyplot as plt
from utils.data_class import diabetis_dataset
import ShIeLD.model
import utils.evaluation_utils as evaluation_utils




parser = argparse.ArgumentParser()
parser.add_argument("-req_path", "--requirements_file_path",
                    default=Path.cwd() / 'examples' / 'diabetes' / 'requirements.pt')
parser.add_argument("-retrain", "--retain_best_model_config_bool", default=True)
parser.add_argument("-config_dict", "--best_config_dict_path",
                    default=Path.cwd() / 'examples' / 'diabetes' / 'best_config.pt')

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

data_loader = DataListLoader(
                    diabetis_dataset(
                        root=path_to_graphs / 'test'/'graphs',
                        csv_file=requirements['path_to_data_set'] / f'test_set.csv',
                        graph_file_names_path=requirements['path_to_data_set'] / f'test_set_file_names.pkl' ),
                    batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50)


model = ShIeLD.model.ShIeLD(
                num_of_feat=int(requirements['input_layer']),
                layer_1=requirements['layer_1'],
                layer_final=requirements['out_put_layer'],
                dp=best_config_dict['droup_out_rate'],
                self_att=False, attr_bool=requirements['attr_bool']
            ).to(device)

model.load_state_dict(torch.load(requirements['path_to_models'] / f'best_model_{best_config_dict["version"]}.pt'))
model.eval()

cell_to_cell_interaction_dict = evaluation_utils.get_cell_to_cell_interaction_dict(
        requirements_dict = requirements,
        data_loader = data_loader,
        model= model,
        device = device,
        save_dict_path = Path(requirements['path_to_model']/'cT_t_cT_interactions_dict.pt'))


number_interactions = 4
observed_tissues = list(requirements['label_dict'].keys())

observed_tissue = observed_tissues[0]

tissue_id = requirements['label_dict'][observed_tissue]


sample_id_list = cell_to_cell_interaction_dict['true_labels'] == tissue_id

all_dfs = list(compress(cell_to_cell_interaction_dict['normed_p2p'], sample_id_list))
print(len(all_dfs))
threshould = len(all_dfs)*0.01
mean_cell_att = pd.concat(all_dfs)

df = mean_cell_att.groupby(mean_cell_att.index).agg(lambda x: evaluation_utils.get_median_with_threshold(x, threshould)).sort_index()[sorted(mean_cell_att.columns)]

