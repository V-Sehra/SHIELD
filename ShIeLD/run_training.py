#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:18:15 2024

@author: Vivek
"""
import numpy as np
from torch_geometric.loader import DataListLoader
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(f'{cwd}', 'utils'))
import model_utils
import data_utils
from ShIeLD.utils.data_class import local_immune_graph_dataset
from model import ShIeLD
import argparse
import torch
idx_folder = cwd.index('ShIeLD') + len('ShIeLD')
shield_dir = os.path.join(f'{cwd[:idx_folder]}')

def train_model(radius_neibourhood, minimum_number_cells, attr_bool,
                input_dim, Layer_1, droup_out_rate, final_layer, batch_size, learning_rate, device,
                path_data,path_save_model,path_name_list):

    # load the training data
    data_train = DataListLoader(
        local_immune_graph_dataset(root=path_data,
                                   path_to_name_file=path_name_list),
                                   batch_size=batch_size, shuffle=True, num_workers=8,
                                   prefetch_factor=50)


    if attr_bool:
        comment_att = '_attr'
    else:
        comment_att = '_Noattr'

    #define the model name with which it will be safed under
    model_name = f"HL1_{Layer_1}_dp_{droup_out_rate}_r_{data_utils.turn_pixel_to_meter(radius_neibourhood)}_noSelfATT_{minimum_number_cells}{comment_att}".replace('.', '_')
    save_path = os.path.join(f'{path_save_model}', f'{model_name}.pt')

    # initialize the loss function as the weighted cross entropy loss
    # the weights are calculated based on the number of graphs per tissue type
    loss_fkt = model_utils.initiaize_loss(path=path_name_list, device=device)

    train_loss_epoch = []

    model = ShIeLD(num_of_feat=int(input_dim),
                  layer_1=Layer_1, dp=droup_out_rate, Layer_final=final_layer,
                  self_att=False, attr_bool=attr_bool)


    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0
    early_stopping = False
    while not early_stopping:


        model,loss_batch = model_utils.train_loop(data_loader = data_train,
                               model = model, optimizer = optimizer,
                               loss_fkt = loss_fkt, attr_bool=attr_bool)

        train_loss_epoch.append(np.mean(loss_batch))
        print('mean epoch loss = ', train_loss_epoch[-1], 'for epoch =', epoch + 1)
        early_stopping = model_utils.early_stopping(loss_epoch = train_loss_epoch, patience=15)

    print('training finished')
    print('mean epoch loss = ', train_loss_epoch[-1], 'for epoch =', epoch + 1)
    torch.save(model.state_dict(), save_path)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.set_start_method('fork')
    torch.multiprocessing.set_sharing_strategy('file_system')


    parser = argparse.ArgumentParser()
    # data spezific
    parser.add_argument("-r_s", "--number_steps_region_subsampleing", type=int, default=100)
    parser.add_argument("-radius", "--radius", type=int, default=50)
    parser.add_argument("-min_cells", "--minimum_number_cells", type=str, default=50)
    parser.add_argument("-attr_bool", "--attr_bool", type=str, default='False', choices=['True', 'False'])

    #all paths that need to be provided
    parser.add_argument("-path_to_graps", "--path_to_graps", type=str, default=None)
    parser.add_argument("-path_save_model", "--path_save_model", type=str, default=None)
    parser.add_argument("-path_name_list", "--path_name_list", type=str, default=None)


    # model spezific
    parser.add_argument("-s_m", "--similarity_measure", type=str, default='euclide')
    parser.add_argument("-l_1", "--Layer_1", type=int, default=27)
    parser.add_argument("-dp", "--droup_out_rate", type=int, default=0.4)

    # training spezific
    parser.add_argument("-input_dim", "--input_dim", type=int, default=27)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
    parser.add_argument("-final_layer", "--final_layer", type=int, default=3)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=450)

    args = parser.parse_args()


    comment = args.comment

    attr_bool = data_utils.bool_passer(args.attr_bool)
    if attr_bool:
        comment_att = '_attr'
    else:
        comment_att = '_Noattr'

    path_to_graps, path_save_model, path_org_csv_file, path_name_list = data_utils.path_generator(path_to_graps = args.path_to_graps,
                                                                                                  path_save_model = args.path_save_model,
                                                                                                  path_org_csv_file = args.path_org_csv_file,
                                                                                                  path_name_list = args.path_name_list)


    train_model(radius_neibourhood = args.radius,
                minimum_number_cells = args.minimum_number_cells,
                attr_bool = attr_bool,input_dim = args.input_dim, Layer_1 = args.Layer_1,
                droup_out_rate = args.droup_out_rate, final_layer = args.final_layer,
                batch_size = args.batch_size, learning_rate = args.learning_rate,
                device = device, path_data = path_to_graps, path_save_model = path_save_model,
                path_name_list = path_name_list)