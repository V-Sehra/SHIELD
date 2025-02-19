from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from torch_geometric.loader import DataListLoader

from pathlib import Path

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from utils.data_class import diabetis_dataset
import argparse
from sklearn.metrics import balanced_accuracy_score


torch.multiprocessing.set_start_method('fork')
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fold", type=int, default=1)
parser.add_argument("-comment", "--comment", type=str, default='random_anker')
parser.add_argument("-comment_norm", "--comment_norm", type=str, default='noNorm')

parser.add_argument("-req_path", "--requirements_file_path", default=Path.cwd()/'examples' / 'diabetes' / 'requirements.pt')

args = parser.parse_args()
print(args)

with open(args.requirements_file_path, 'rb') as file:
    requirements = pickle.load(file)


#training specific information
input_dim = requirements['input_layer']
batch_size = requirements['batch_size']
learning_rate = requirements['learning_rate']
f_final = requirements['out_put_layer']
Layer_1 = requirements['layer_1']

#comment specific information
comment = args.comment
comment_norm = args.comment_norm

attr_bool = requirements['attr_bool']

if attr_bool:
    comment_att = 'attr'
else:
    comment_att = 'Noattr'


# Run one job per validation version
# Each version will run through all the different hyperparameters
fold = args.fold




for radius_distance in requirements['radius_distance_all']:


    for fussy_limit in requirements['fussy_limit_all']:

        for anker_prozent in requirements['prozent_of_anker_cells']:


            path_to_graphs = Path(requirements['path_graphs'] / \
                                         f'anker_value_{anker_prozent}'.replace('.', '_') / \
                                         f"min_cells_{requirements['minimum_number_cells']}"/ \
                                         f"fussy_limit_{fussy_limit}".replace('.', '_') / \
                                         f'radius_{radius_distance}')

            train_graph_path = Path(f'{path_to_graphs}'/ 'train'/'graphs')

        ####### HERE!

            torch.cuda.empty_cache()
            data_loader_train = DataListLoader(
                    diabetis_dataset(root = train_graph_path,
                                   csv_file = os.path.join(f'{path}','voronoi',f'train_set_fold_{fold}.csv'),
                                   graph_file_names_path = os.path.join(f'{path_to_graphs}',f'train_fold_{fold}_file_names.pkl')),
                    batch_size=batch_size, shuffle=True, num_workers=8,
                    prefetch_factor=50)

            data_loader_validation = DataListLoader(
                    diabetis_dataset(root = train_graph_path,
                                   csv_file = os.path.join(f'{path}','voronoi',f'validation_set_fold_{fold}.csv'),
                                   graph_file_names_path = os.path.join(f'{path_to_graphs}',f'validation_fold_{fold}_file_names.pkl')),
                    batch_size=batch_size, shuffle=True, num_workers=8,
                    prefetch_factor=50)

            for dp in [0.2]:
                for num in tqdm(range(5)):

                    model_csv = pd.DataFrame([[anker_prozent, radius_distance,fussy_limit, dp,
                                               comment,comment_norm, num]],
                                             columns=['prozent_of_anker_cells', 'radius_neibourhood', 'fussy_limit',
                                                      'dp','comment', 'comment_norm', 'model_no'])

                    if pd.merge(training_csv, model_csv,
                                on=['prozent_of_anker_cells', 'radius_neibourhood', 'fussy_limit',
                                                      'dp','comment', 'comment_norm', 'model_no'],
                                how='inner').empty:

                        wandb.init(
                            # set the wandb project where this run will be logged
                            project=f"hyper_search_diabetis_voronoi_shield",

                            # track hyperparameters and run metadata
                            config={
                                "learning_rate": learning_rate,
                                "architecture": "GNN",
                                "dropout": dp,
                                "Layer_1": Layer_1,
                                "layer_final": f_final,
                                "prozent_anker_cells": anker_prozent,
                                "radius_distance": radius_distance,
                                "fussy_limit": fussy_limit,
                                "comment": comment,
                                "comment_norm": comment_norm,
                                "fold": fold,
                                "model_no": num
                            }
                        )
                        config = wandb.config

                        loss_fkt = fct_eval.initiaize_loss(path = os.listdir(train_graph_path),device = device)

                        model = fct_eval.ShIeLD(num_of_feat=int(input_dim),
                                       layer_1=Layer_1, dp=dp, layer_final=f_final,
                                       self_att=False, attr_bool=attr_bool)
                        model = model.to(device)
                        model.train()

                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                        train_loss = []
                        epoch = 0
                        early_stopping = False
                        print('start training')
                        while not early_stopping:

                            loss_batch = []

                            for train_sample in tqdm(data_loader_train):
                                optimizer.zero_grad()

                                prediction, attention, output, y = fct_voronoi.prediction_step(batch_sample = train_sample,
                                                                                              model = model,
                                                                                              attr_bool = attr_bool,
                                                                                              device = device)

                                loss = loss_fkt(output, y)

                                loss.backward()
                                optimizer.step()
                                loss_batch.append(loss.item())

                                wandb.log({"train_loss_running": loss})

                            epoch_loss= np.mean(loss_batch)
                            train_loss.append(epoch_loss)
                            wandb.log({"train_loss_epoch": epoch_loss})
                            epoch += 1

                            if epoch > 5:
                                if ((train_loss[-2] - train_loss[-1]) < 0.001):
                                    early_stopping = True

                                    wandb.log({"number_epoch": epoch})



                        model_prediction_validation = []
                        true_label_validation = []
                        torch.cuda.empty_cache()
                        print('start validation')
                        for validation_sample in data_loader_validation:

                            prediction, attention, output, y = fct_voronoi.prediction_step(batch_sample=validation_sample,
                                                                                          model=model,
                                                                                          attr_bool=attr_bool,
                                                                                          device=device)

                            _, value_pred = torch.max(output, dim=1)

                            wandb.log(
                                {"batch_acc_raw": sum(value_pred.cpu() == y.cpu()) / len(y.cpu())})
                            wandb.log({"batch_acc_balanced": balanced_accuracy_score(y.cpu(),value_pred.cpu())})

                            model_prediction_validation.extend(value_pred.cpu())
                            true_label_validation.extend(y.cpu())


                        wandb.log({"toal_acc_balanced": balanced_accuracy_score(true_label_validation,
                                                                                model_prediction_validation)})
                        wandb.finish()

                        training_csv = pd.concat([model_csv, training_csv], ignore_index=True)
                        training_csv.to_csv(csv_file, index=False)

                        torch.cuda.empty_cache()

                    else:
                        print('model already trained:')
                        print(model_csv)