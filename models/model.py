import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sigmoid, Softmax, Tanh, BatchNorm1d, GELU, Dropout
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, GINConv
from torch_geometric.nn.models import MLP
from torch_geometric.nn.aggr import Set2Set, MaxAggregation
from torch_geometric.nn import BatchNorm
from tqdm import tqdm





class MLP(nn.Module):
    def __init__(self, feature_size, seq_size, output_size, hidden_size=128):
        super(MLP, self).__init__()
        self.feature_size = feature_size
        self.seq_size = seq_size
        self.back = nn.Sequential(
            Linear(feature_size*seq_size,hidden_size),
             BatchNorm1d(hidden_size),ReLU(),
            Linear(hidden_size,hidden_size),
            BatchNorm1d(hidden_size),ReLU()
            #BatchNorm(hidden_size), ReLU(),
            #Linear(hidden_size,output_size)
        )
        self.predictor = nn.Linear(hidden_size,1)
    def forward(self,x,edge_index, batch_index):
        x = x.view(-1,self.feature_size*self.seq_size)
        x = self.back(x)
        x = self.predictor(x)
        return x.view(-1)
    
class GNN_QFM(nn.Module):
    def __init__(self, feature_size, seq_size, output_size, hidden_size=128, num_layers=10, act=GELU(), dropout=0):
        super(GNN_QFM, self).__init__()
        #self.first_layer = GINConv(
        #    nn.Sequential(
        #    Linear(feature_size,hidden_size),
        #    BatchNorm(hidden_size), ReLU(),
        #    Linear(hidden_size,hidden_size), ReLU()
        #    )
        #)
        self.emb = nn.Sequential(Linear(feature_size, hidden_size), BatchNorm(hidden_size), act)
        self.first_layer = TransformerConv(hidden_size, 
                            hidden_size, 
                            heads=4, 
                            concat=False,
                            beta=True,
                            dropout=dropout
                            )#, 
        self.linear = [nn.Sequential(Linear(hidden_size, hidden_size*4), BatchNorm(hidden_size*4), act, Dropout(dropout), Linear(hidden_size*4, hidden_size))]
        self.layers = [self.first_layer]
        #for i in range(10):
        #    self.layers.append(GINConv(
        #        nn.Sequential(
        #            Linear(hidden_size,hidden_size),
        #            BatchNorm(hidden_size), ReLU(),
        #            Linear(hidden_size,hidden_size), ReLU()
        #        )
        #    ))
        self.bn = nn.ModuleList([BatchNorm(hidden_size)])
        for i in range(num_layers):
            self.bn.append(BatchNorm(hidden_size))
            self.layers.append(#nn.Sequential(
                TransformerConv(
                    hidden_size,
                    hidden_size,
                    heads=4,
                    concat=False,
                    beta=True,
                    dropout=dropout))
            self.linear.append(nn.Sequential(Linear(hidden_size, hidden_size*4), BatchNorm(hidden_size*4), act,Dropout(dropout), Linear(hidden_size*4, hidden_size)))

        self.layers = nn.ModuleList(self.layers)
        self.linear = nn.ModuleList(self.linear)

        #self.pooling = Set2Set(hidden_size,4)\
        self.pooling = MaxAggregation()

        self.drop = nn.Dropout(0.5)
        self.predictor = Linear(hidden_size,1)


    def encode(self,x,edge_index,batch_index):
        x = self.emb(x)
        for i,n in enumerate(self.layers):
            x = n(x,edge_index)
            x = self.linear[i](x)

        x = self.pooling(x,batch_index)
        return x

    def forward(self, x, edge_index, batch_index):
        x = self.emb(x)
        for i,n in enumerate(self.layers):
            x = n(x,edge_index)
            x = self.linear[i](x)

        x = self.pooling(x,batch_index)
        #x = self.drop(x)
        x = self.predictor(x)
        return x.view(-1)