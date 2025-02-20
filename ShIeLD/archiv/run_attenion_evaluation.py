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
sys.path.append(os.path.join(f'{cwd}', '../utils'))
import model_utils
import data_utils
from ShIeLD.utils.data_class import local_immune_graph_dataset
from model import ShIeLD
import argparse
import torch
import pickle

idx_folder = cwd.index('ShIeLD') + len('ShIeLD')
shield_dir = os.path.join(f'{cwd[:idx_folder]}')

# caluclate the cell type (phenotype) attention score for each cell type (phenotype)
# per tissue type (normalLiver, core, rim)
def calcluate_cT_2_cT_att_score(radius_neibourhood, minimum_number_cells, cell_names,
                                attr_bool, batch_size,
                                device, input_dim, Layer_1, droup_out_rate, final_layer,
                                path_to_graps, path_model, path_name_list, data_set):
    data_loader = DataListLoader(
        local_immune_graph_dataset(root=path_to_graps,
                                   path_to_name_file=path_name_list),
        batch_size=batch_size, shuffle=True, num_workers=8,
        prefetch_factor=50)

    model = ShIeLD(num_of_feat=int(input_dim),
                   layer_1=Layer_1, dp=droup_out_rate, layer_final=final_layer,
                   self_att=False, attr_bool=attr_bool).to(device)

    model.load_state_dict(torch.load(path_model))

    model.eval()

    # lists to collect the information for each sample
    fals_pred = []
    correct_predicted = []
    true_labels_train = []
    ids_list = []

    raw_att_p2p = []
    normalisation_factor_edge_number = []
    normed_p2p = []

    # combin the Granulocytes and B cells into one cell phenotype as there rare subpopulations
    cell_names_shortend = data_utils.combine_cell_types(cell_names, ['B cells', 'Granulocytes'])

    # run through all graphs
    for data_sample in data_loader:
        # turn the batch data samples into one list as the minibatching from
        # pythorch geometric is to memory intensive
        sample_x, sample_edge, sample_att, ids = model_utils.turn_data_list_into_batch(
            data_sample=data_sample, device=device)

        prediction, attenion = model(sample_x, sample_edge, sample_att)
        _, value_pred = torch.max(torch.vstack(prediction), dim=1)

        target_test = torch.tensor([sample.y for sample in data_sample]).to(device)

        # collect the performance information for each sample
        fals_pred.extend((value_pred != target_test).flatten().cpu().detach().numpy())
        correct_predicted.extend((value_pred == target_test).flatten().cpu().detach().numpy())
        true_labels_train.extend(target_test.flatten().cpu().detach().numpy())
        # collect the patient id for the sample
        ids_list.extend(ids)

        # orignal att scores on a node level
        nodel_level_attention_scores = [att_val[1].cpu().detach().numpy() for att_val in attenion]
        cell_type_names = data_utils.replace_celltype(np.array([np.array([cell_type[2] for cell_type in sample.eval])
                                                                 for sample in data_sample]),
                                                       ['B cells', 'Granulocytes'])

        # turn the orignal att scores into the phenotype to phenotype attention scores
        phenotype_attention_matrix = model_utils.get_p2p_att_score(sample=data_sample,
                                                                   cell_phenotypes_sample=cell_type_names,
                                                                   all_phenotypes=cell_names_shortend,
                                                                   node_attention_scores=nodel_level_attention_scores)
        raw_att_p2p.append(phenotype_attention_matrix[0])
        normalisation_factor_edge_number.append(phenotype_attention_matrix[1])
        normed_p2p.append(phenotype_attention_matrix[2])

    # collect all information into one dictionary
    dict_all_info = {
        'fals_pred': np.array(fals_pred),
        'correct_predicted': np.array(correct_predicted),
        'true_labels_train': np.array(true_labels_train),
        'ids_list': np.array(ids_list),
        'normed_p2p': normed_p2p,
        'raw_att_p2p': raw_att_p2p,
        'normalisation_factor_edge_number': normalisation_factor_edge_number,
    }

    dict_name = f"HL1_{Layer_1}_dp_{droup_out_rate}_r_{radius_neibourhood}_noSelfATT_{minimum_number_cells}{comment_att}_eval_{data_set}".replace(
        '.', '_')

    save_path = os.path.join(f'{shield_dir}', 'eval', f'{dict_name}.pkl')
    os.system(f'mkdir -p {os.path.join(f"{shield_dir}", "eval")}')
    with open(save_path, 'wb') as f:
        pickle.dump(dict_all_info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data spezific
    parser.add_argument("-r_s", "--number_steps_region_subsampleing", type=int, default=100)
    parser.add_argument("-radius", "--radius", type=int, default=50)
    parser.add_argument("-min_cells", "--minimum_number_cells", type=str, default=50)
    parser.add_argument("-attr_bool", "--attr_bool", type=str, default='False', choices=['True', 'False'])

    parser.add_argument("-cell_types", "--cell_types", type=str,
                        default=['B cells CD38+', 'B cells CD45RA', 'B cells PD-L1+',
                                 'Granulocytes CD38+', 'Granulocytes CD38-', 'Kupffer cells',
                                 'M2 Macrophages PD-L1+', 'M2 Macrophages PD-L1-', 'MAITs',
                                 'MHCII APCs', 'Mixed Immune CD45+', 'NK Cells CD16',
                                 'NK Cells CD56', 'T cells CD4', 'T cells CD4 PD-L1+',
                                 'T cells CD4 naïve', 'T cells CD57', 'T cells CD8 PD-1high',
                                 'T cells CD8 PD-1low', 'T cells CD8 PD-L1+', 'Tregs'])

    # all paths that need to be provided
    parser.add_argument("-path_to_graps", "--path_to_graps", type=str, default=None)
    parser.add_argument("-path_model", "--path_model", type=str, default=None)
    parser.add_argument("-path_name_list", "--path_name_list", type=str, default=None)
    parser.add_argument("-data_set", "--data_set", type=str, default='Test', choices=['Test', 'Train'])

    # model spezific
    parser.add_argument("-l_1", "--layer_one", type=int, default=27)
    parser.add_argument("-dp", "--droup_out_rate", type=int, default=0.4)

    # training spezific
    parser.add_argument("-input_dim", "--input_dim", type=int, default=27)
    parser.add_argument("-final_layer", "--final_layer", type=int, default=3)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=450)

    args = parser.parse_args()

    attr_bool = False

    if attr_bool:
        comment_att = '_attr'
    else:
        comment_att = '_Noattr'

    path_to_graps, path_save_model, path_org_csv_file, path_name_list = data_utils.path_generator(
        path_to_graps=args.path_to_graps,
        path_save_model=args.path_save_model,
        path_org_csv_file=args.path_org_csv_file,
        path_name_list=args.path_name_list, cwd=cwd)

    calcluate_cT_2_cT_att_score(radius_neibourhood=args.radius, minimum_number_cells=args.minimum_number_cells,
                                cell_names=args.cell_types, attr_bool=comment_att, batch_size=args.batch_size,
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                input_dim=args.input_dim, Layer_1=args.layer_one, droup_out_rate=args.droup_out_rate,
                                final_layer=args.final_layer,
                                path_to_graps=path_to_graps,
                                path_model=path_save_model,
                                path_name_list=path_name_list, data_set=args.data_set)
