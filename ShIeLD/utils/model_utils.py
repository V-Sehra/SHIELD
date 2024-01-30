import numpy as np
import pickle
import torch
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import pandas as pd

def calc_edge_mat(mat, dist_bool=True, radius=265):
    neigh = NearestNeighbors(radius=radius).fit(mat)
    neigh_dist, neigh_ind = neigh.radius_neighbors(mat)

    # check the number of conections each cell has
    total_number_conections = [len(conections) for conections in neigh_ind]

    # create the source node indices based on the number of conections a cell has
    edge_scr = np.concatenate([np.repeat((idx), total_number_conections[idx])
                               for idx in range(len(total_number_conections))])

    # concat all idx calculated by sklearn as the dest node
    edge_dest = np.concatenate(neigh_ind)

    # remove the self connection
    remove_self_node = np.zeros(len(total_number_conections), dtype=np.dtype('int'))

    idx_counter = 0
    for idx in range(len(total_number_conections)):
        remove_self_node[idx] = int(idx_counter)
        idx_counter += total_number_conections[idx]

    edge_scr = np.delete(edge_scr, remove_self_node)
    edge_dest = np.delete(edge_dest, remove_self_node)

    edge = np.array([edge_scr, edge_dest])

    if dist_bool == True:

        return (edge, np.delete(np.concatenate(neigh_dist), remove_self_node))
    else:
        return (edge)


def early_stopping(loss_epoch, patience=15):
    if len(loss_epoch) > patience:

        if ((loss_epoch[-2] - loss_epoch[-1]) < 0.001):
            return (True)
        else:
            return (False)

    else:
        return (False)

# calculate the attention score for each cell phenotype to all other phenotypes
# to reduce confusion we call the end attention score p2p (phenotype to phenotype)
# also we are using the word "node" instead of cell
def get_p2p_att_score(sample,cell_phenotypes_sample,all_phenotypes,
                      node_attention_scores):
    raw_att_p2p = []
    normalisation_factor_edge_number = []
    normalised_p2p = []

    #find the cell type (phenotype) for each cell/node
    scr_node = [cell_phenotypes_sample[sample_idx][cell_phenotypes_sample[sample_idx][0]] for sample_idx in range(len(sample))]
    dst_node = [cell_phenotypes_sample[sample_idx][cell_phenotypes_sample[sample_idx][1]] for sample_idx in range(len(sample))]

    for sample_idx in range(len(scr_node)):
        df_att = pd.DataFrame(data={'src': scr_node[sample_idx],
                                    'dst': dst_node[sample_idx],
                                    'att': node_attention_scores[sample_idx].flatten()})
        # have the same DF containig all phenotypes
        # is a connections is not present in the sample we fill it with NaN
        att_df = data_utils.fill_missing_row_and_col_withNaN(
            data_frame=df_att.groupby(['src', 'dst'])['att'].sum().reset_index().pivot(index='src', columns='dst',
                                                                                       values='att'),
            cell_types_names=all_phenotypes)
        # unnormalised att score
        raw_att_p2p.append(att_df)

        edge_df = data_utils.fill_missing_row_and_col_withNaN(
            data_frame=df_att.groupby(['src', 'dst']).count().reset_index().pivot(index='src', columns='dst',
                                                                                  values='att'),
            cell_types_names=all_phenotypes)

        # normalise the p2p based on the raw count of ed between these two pheno types
        normalisation_factor_edge_number.append(edge_df)
        normalised_p2p.append(att_df / edge_df)
    return(raw_att_p2p, normalisation_factor_edge_number, normalised_p2p)


# load the loss as weightes loss
def initiaize_loss(path, device, tissue_dict=None):
    if tissue_dict is None:
        tissue_dict = {'normalLiver': 0,
                       'core': 1,
                       'rim': 2}
    class_weights = []
    # collect all files names and prevent non graphs to count
    with open(path,'rb') as f:
        all_train_file_names = pickle.load(f)


    for origin in tissue_dict.keys():
        number_tissue_graphs = len([file_name for file_name in all_train_file_names
                                    if file_name.find(origin) != -1])
        class_weights.append(1 - (number_tissue_graphs / len(all_train_file_names)))

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    return(criterion)

def turn_data_list_into_batch(data_sample,device):
    sample_x = [sample.x.to(device) for sample in data_sample]
    sample_edge = [sample.edge_index_plate.to(device) for sample in data_sample]
    sample_att = [sample.plate_euc.to(device) for sample in data_sample]
    ids_list = [sample.ids for sample in data_sample]

    return(sample_x, sample_edge, sample_att, ids_list)


def train_loop(data_loader, model, optimizer, loss_fkt, attr_bool=False, loss_batch=[]):

    for train_sample in data_loader:
        optimizer.zero_grad()
        sample_x = [sample.x.to('cuda') for sample in train_sample]
        sample_edge = [sample.edge_index_plate.to('cuda') for sample in train_sample]
        if attr_bool:
            sample_att = [sample.plate_euc.to('cuda') for sample in train_sample]

            prediction, attenion = model(sample_x, sample_edge, sample_att)
        else:
            prediction, attenion = model(sample_x, sample_edge)

        output = torch.vstack(prediction)
        y = torch.tensor([sample.y for sample in train_sample]).to(output.device)

        loss = loss_fkt(output, y)

        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())

    return(model,loss_batch)

def remove_zero_distances(edge, dist, limit=0, return_dist=False):

    zero_index = np.where(dist <= limit)[0]

    if len(zero_index) != 0:
        edge = np.delete(edge, zero_index, 1)

    if return_dist:
        dist = np.delete(dist, zero_index, 0)
        return (edge, dist)

    else:
        return (edge)