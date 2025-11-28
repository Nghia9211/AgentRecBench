import torch 
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=False)
    
    def forward(self, A, X):
        H = self.linear(X) 
        out = torch.sparse.mm(A, H)
        return out

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gc1 = GCNLayer(input_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, A, X):
        x = self.gc1(A, X)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gc2(A, x)
        return x
