from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F


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
                 self_att: bool =True, attr_bool:bool = True):
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

    def forward(self, node_list, edge_list, edge_att=None):
        """
        Forward pass for processing multiple graph samples.

        Args:
            node_list (list[torch.Tensor]): List of node feature tensors, each corresponding to a graph.
            edge_list (list[torch.Tensor]): List of edge index tensors defining graph connectivity.
            edge_att (list[torch.Tensor], optional): List of edge attribute tensors.

        Returns:
            tuple:
                - list[torch.Tensor]: List of softmax predictions for each input graph.
                - list[torch.Tensor]: List of attention scores from the GAT layer.
        """
        prediction = []  # List to store predictions for each graph
        attention = []  # List to store attention weights

        sample_number = len(node_list)  # Number of graphs in the batch

        for idx in range(sample_number):
            x = node_list[idx].float()  # Convert node features to float
            edge_index = edge_list[idx].long()  # Convert edge indices to long tensor

            # Apply GAT convolution, with or without edge attributes
            if self.attr_bool:
                edge_attr = edge_att[idx].float()
                x, att = self.conv1(x=x, edge_index=edge_index,
                                    edge_attr=edge_attr, return_attention_weights=True)
            else:
                x, att = self.conv1(x=x, edge_index=edge_index,
                                    return_attention_weights=True)

            x = F.relu(x)  # Apply ReLU activation
            x = F.dropout(x, p=self.dp, training=self.training)  # Apply dropout

            x = self.Mean_agg(x, dim=0)  # Aggregate node embeddings into a graph representation
            x = self.lin(x)  # Final linear transformation

            prediction.append(F.softmax(x, dim=1))  # Softmax activation for classification
            attention.append(att)  # Store attention scores

        return prediction, attention



