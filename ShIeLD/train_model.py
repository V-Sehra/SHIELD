import os
import argparse
import pickle
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataListLoader

from utils.data_class import diabetis_dataset
import utils.train_utils as train_utils
import ShIeLD.model

torch.multiprocessing.set_start_method('fork', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-split", "--split_number", type=int, default=1)
    parser.add_argument("-comment", "--comment", type=str, default='random_anker')
    parser.add_argument("-comment_norm", "--comment_norm", type=str, default='noNorm')
    parser.add_argument("-req_path", "--requirements_file_path",
                        default=Path.cwd() / 'examples' / 'diabetes' / 'requirements.pt')

    args = parser.parse_args()
    print(args)

    with open(args.requirements_file_path, 'rb') as file:
        requirements = pickle.load(file)

    # Training-specific information
    input_dim = requirements['input_layer']
    batch_size = requirements['batch_size']
    learning_rate = requirements['learning_rate']
    f_final = requirements['out_put_layer']
    Layer_1 = requirements['layer_1']
    attr_bool = requirements['attr_bool']

    comment = args.comment
    comment_norm = args.comment_norm
    comment_att = 'attr' if attr_bool else 'Noattr'

    split_number = args.split_number

    training_results_csv, csv_file_path = train_utils.get_train_results_csv(
        requirement_dict=requirements, split_number=split_number)

    meta_columns = ['anker_value', 'radius_neibourhood', 'fussy_limit', 'dp', 'comment', 'comment_norm', 'model_no']

    for radius_distance in requirements['radius_distance_all']:
        for fussy_limit in requirements['fussy_limit_all']:
            for anker_number in requirements['prozent_of_anker_cells']:

                path_to_graphs = Path(requirements['path_to_data_set'] /
                                      f'anker_value_{anker_number}'.replace('.', '_') /
                                      f"min_cells_{requirements['minimum_number_cells']}" /
                                      f"fussy_limit_{fussy_limit}".replace('.', '_') /
                                      f'radius_{radius_distance}')

                train_graph_path = path_to_graphs / 'train' / 'graphs'
                torch.cuda.empty_cache()

                data_loader_train = DataListLoader(
                    diabetis_dataset(
                        root=train_graph_path,
                        csv_file=requirements['path_to_data_set'] / f'train_set_validation_split_{split_number}.csv',
                        graph_file_names_path=requirements[
                                                  'path_to_data_set'] / f'train_set_validation_split_{split_number}_file_names.pkl'
                    ),
                    batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=50
                )

                data_loader_validation = DataListLoader(
                    diabetis_dataset(
                        root=train_graph_path,
                        csv_file=requirements[
                                     'path_to_data_set'] / f'validation_set_validation_split_{split_number}.csv',
                        graph_file_names_path=requirements[
                                                  'path_to_data_set'] / f'validation_validation_split_{split_number}_file_names.pkl'
                    ),
                    batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=50
                )

                for dp in requirements['droupout_rate']:
                    for num in tqdm(range(5)):

                        if not training_results_csv[meta_columns].isin(
                                [[anker_number, radius_distance, fussy_limit, dp, comment, comment_norm, num]]
                        ).all(axis=1).any():

                            loss_fkt = train_utils.initiaize_loss(
                                path=os.listdir(train_graph_path),
                                tissue_dict=requirements['label_dict'],
                                device=device
                            )

                            model = ShIeLD.model.ShIeLD(
                                num_of_feat=int(input_dim),
                                layer_1=Layer_1, dp=dp, layer_final=f_final,
                                self_att=False, attr_bool=attr_bool
                            ).to(device)
                            model.train()

                            model, train_loss = train_utils.train_loop_shield(
                                optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
                                model=model,
                                data_loader=data_loader_train,
                                loss_fkt=loss_fkt,
                                attr_bool=attr_bool,
                                device=device
                            )

                            train_bal_acc = train_utils.get_balance_acc(
                                model=model, data_loader=data_loader_train,
                                attr_bool=attr_bool, device=device
                            )
                            print('start validation')
                            val_bal_acc = train_utils.get_balance_acc(
                                model=model, data_loader=data_loader_validation,
                                attr_bool=attr_bool, device=device
                            )

                            model_csv = pd.DataFrame([[anker_number, radius_distance, fussy_limit, dp, comment,
                                                       comment_norm, num, train_bal_acc, val_bal_acc]],
                                                     columns=meta_columns + ['train_acc', 'val_acc'])
                            training_results_csv = pd.concat([model_csv, training_results_csv], ignore_index=True)
                            training_results_csv.to_csv(csv_file_path, index=False)

                            torch.cuda.empty_cache()
                        else:
                            print('Model already trained:', anker_number, radius_distance, fussy_limit, dp, comment,
                                  comment_norm, num)
if __name__ == "__main__":
    main()
