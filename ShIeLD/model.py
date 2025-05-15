from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
from torch.nn import Linear,BatchNorm1d
import torch.nn.functional as F
from torch_geometric.data import Batch

class ShIeLD(nn.Module):
    """
    ShIeLD: A simple Graph Attention Network (GAT) variant that applies self-attention
    and edge attributes to process graph-structured data. Uses a single GAT layer followed
    by mean aggregation and a final linear layer for classification.

    Attributes:
        conv1 (GATv2Conv): Single-layer graph attention convolutional network.
        lin (nn.Linear): Linear layer mapping learned features to output classes.
        attr_bool (bool): Whether edge attributes are used in the GAT layer.
        dp (float): Dropout probability for regularization.
        similarity_typ (str): Type of similarity metric used (default: 'euclide').
        Mean_agg (MeanAggregation): Aggregation function for node embeddings.
    """

    def __init__(self, num_of_feat: int, layer_1:int, dp: float,
                 layer_final: int =2, edge_dim: int =1, similarity_typ: str ='euclide',
                 self_att: bool =True, attr_bool:bool = True, norm_type: str = 'No_norm'):
        """
        Initializes the ShIeLD model.

        Args:
            num_of_feat (int): Number of input node features.
            layer_1 (int): Number of features in the GAT layer.
            dp (float): Dropout probability for regularization.
            layer_final (int): Number of output classes (default: 2).
            edge_dim (int): Dimension of edge attributes.
            similarity_typ (str): Type of similarity function to use (default: 'euclide').
            self_att (bool): Whether to add self-loops in the GAT layer.
            attr_bool (bool): Whether edge attributes are included in the attention mechanism.
            norm_type (str): Type of normalization to apply after the GAT layer (default: 'No_norm').
        """
        super(ShIeLD, self).__init__()

        # Single GAT layer with optional self-loops
        self.conv1 = GATv2Conv(num_of_feat, layer_1, edge_dim=edge_dim,
                               add_self_loops=self_att)

        self.lin = Linear(layer_1, layer_final)  # Linear output layer
        self.attr_bool = attr_bool  # Determines if edge attributes are used
        self.dp = dp  # Dropout probability
        self.similarity_typ = similarity_typ  # Similarity metric
        self.Mean_agg = MeanAggregation()  # Aggregates node embeddings

        self.norm_type = norm_type  # Type of normalization to apply

        # Determine what normalization is used and needs to be loaded
        if ('batch_norm'.lower() in self.norm_type.lower()) or \
                ('sample_norm'.lower() in self.norm_type):
            # Batch normalization layer for the output of the GAT layer
            self.norm = BatchNorm1d(layer_1)
        elif 'layer_norm'.lower() in self.norm_type.lower():
            # Layer normalization layer for the output of the GAT layer
            self.norm = nn.LayerNorm(layer_1)
        else:
            # If batch normalization is not used, set it to None
            self.norm = None

        if 'forceTrain'.lower() in self.norm_type.lower():
            self.force_norm_train = True
        else:
            self.force_norm_train = False


    def forward(self, data_list):
        """
        Forward pass for processing multiple graph samples.

        Args:
            data_list (list[torch.Tensor]): List of the Dataobjects i.e:
                                            -node feature tensors, each corresponding to a graph.
                                            -edge index tensors defining graph connectivity.
                                            -edge attribute tensors [optional].

        Returns:
            tuple:
                - list[torch.Tensor]: List of softmax predictions for each input graph.
                - list[torch.Tensor]: List of attention scores from the GAT layer.
        """

        # Force batch normalization to be in training mode
        if self.force_norm_train:

            self.norm.train()

        prediction = []  # List to store predictions for each graph
        attention = []  # List to store attention weights

        if 'batch_norm' in self.norm_type.lower():
            whole_batch = Batch.from_data_list(data_list)
            x_normed = self.norm(whole_batch.x.float())

        sample_number = len(data_list)  # Number of graphs in the batch

        for idx in range(sample_number):

            if 'batch_norm' in self.norm_type.lower():
                x = x_normed[whole_batch.ptr[idx]:whole_batch.ptr[idx + 1]]
            elif 'sample_norm'.lower() in self.norm_type:
                x = self.norm(data_list[idx].x.float())
            else:
                x = data_list[idx].x.float()  # Convert node features to float

            edge_index = data_list[idx].edge_index_plate.long()  # Convert edge indices to long tensor

            # Apply GAT convolution, with or without edge attributes
            if self.attr_bool:
                edge_attr = data_list[idx].edge_att.float()
                x, att = self.conv1(x=x, edge_index=edge_index,
                                    edge_attr=edge_attr, return_attention_weights=True)
            else:
                x, att = self.conv1(x=x, edge_index=edge_index,
                                    return_attention_weights=True)

            if 'layer_norm' in self.norm_type.lower():
                x = self.norm(x)

            x = F.relu(x)  # Apply ReLU activation
            x = F.dropout(x, p=self.dp, training=self.training)  # Apply dropout

            x = self.Mean_agg(x, dim=0)  # Aggregate node embeddings into a graph representation
            x = self.lin(x)  # Final linear transformation

            prediction.append(F.softmax(x, dim=1))  # Softmax activation for classification
            attention.append(att)  # Store attention scores

        return prediction, attention



