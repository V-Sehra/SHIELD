from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F


class ShIeLD(nn.Module):
    def __init__(self, num_of_feat, layer_1, dp,
                 Layer_final=2, edge_dim=1, similarity_typ='euclide',
                 self_att=True, attr_bool=True):
        super(ShIeLD, self).__init__()

        self.conv1 = GATv2Conv(num_of_feat, layer_1, edge_dim=edge_dim,
                               add_self_loops=self_att)

        self.lin = Linear(layer_1, Layer_final)
        self.attr_bool = attr_bool
        self.dp = dp
        self.similarity_typ = similarity_typ
        self.Mean_agg = MeanAggregation()

    def forward(self, node_list, edge_list, edge_att=None):
        prediction = []
        attenion = []

        sample_number = len(node_list)

        for idx in range(sample_number):

            x = node_list[idx].float()

            edge_index = edge_list[idx].long()

            if self.attr_bool:
                edge_attr = edge_att[idx].float()

                x, att = self.conv1(x=x,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr, return_attention_weights=True)
            else:
                x, att = self.conv1(x=x,
                                    edge_index=edge_index, return_attention_weights=True)

            x = F.relu(x)
            x = F.dropout(x, p=self.dp, training=self.training)

            x = self.Mean_agg(x, dim=0)

            x = self.lin(x)

            prediction.append(F.softmax(x, dim=1))
            attenion.append(att)
        return prediction, attenion