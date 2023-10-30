import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from CRL_Conv import CRL_Conv

class CRL_SGNN(nn.Module):
    def __init__(self):
        super(CRL_SGNN, self).__init__()
        self.Embedding_Generation_Module = Embedding_Generation_Module()
        self.Assistant = Assistant_Branch()
        self.Main = Main_Branch()
        
        physical_bias = 0.001*torch.ones(999,requires_grad=True)
        physical_bias[500:] = 0
        self.physical_bias = nn.Parameter(physical_bias)

    def forward(self,X,edge_index):

        X = self.Embedding_Generation_Module(X,edge_index)
        X_backbone = X.view(-1,1)
        X_backbone = X_backbone.squeeze()


        X_embedding,X_loss1 = self.Assistant(X,edge_index)


        X = torch.cat((X_backbone,X_embedding))
        X_output = self.Main(X)

        X_output = X_output + self.physical_bias
        X_loss1 = X_loss1 + self.physical_bias

        return X_loss1,X_output



class CRL_block(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(CRL_block, self).__init__()
        self.bonesage = CRL_Conv(input_dim,output_dim)
        self.bonesage1 = CRL_Conv(output_dim,output_dim)

    def forward(self,X,edge_index):

        X1 = self.bonesage(X,edge_index)
        X2 = self.bonesage1(X1,edge_index)
        X_res = X1 + X2

        return X_res

class Embedding_Generation_Module(nn.Module):
    def __init__(self):
        super(Embedding_Generation_Module, self).__init__()
        self.bonesage = CRL_block(2,4)
        self.bonesage_res = CRL_block(4,4)
        self.bonesage1 = CRL_block(4,2)


    def forward(self,X,edge_index):

        X1 = self.bonesage(X,edge_index)

        X_res= self.bonesage_res(X1,edge_index)
        X1 = X1+X_res

        X2= self.bonesage1(X1,edge_index)
        return X2

class Assistant_Branch(nn.Module):
    def __init__(self):
        super(Assistant_Branch, self).__init__()
        self.sage = SAGEConv(2,1)
        self.sage1 = SAGEConv(2,1,aggr='max')
        self.linear = nn.Linear(1000,1000)
        self.linear1 = nn.Linear(1000,1000)
        self.linear2 = nn.Linear(1000,999)


    def forward(self,X,edge_index):
        X_mean = F.relu(self.sage(X,edge_index))
        X_max = F.relu(self.sage1(X,edge_index))
        X = X_mean + X_max
        X = X.view(-1,1)
        X = X.squeeze()
        X_embedding = X
        X = F.relu(self.linear(X))
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))

        return  X_embedding,X



class Main_Branch(nn.Module):
    def __init__(self):
        super(Main_Branch, self).__init__()
        self.linear1 = nn.Linear(3000,2000)
        self.linear2 = nn.Linear(2000,1000)
        self.linear3 = nn.Linear(1000,999)
    def forward(self,X):

        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        return X


