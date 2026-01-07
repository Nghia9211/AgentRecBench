import torch 
import torch.nn as nn

class LightGCN_3Hop(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, A):
        E0 = self.embedding.weight 
        
        E1 = torch.sparse.mm(A, E0)
        
        E2 = torch.sparse.mm(A, E1)

        E3 = torch.sparse.mm(A, E2)
        

        E_final = (E0 + E1 + E2 + E3) / 4
        
        return E_final, E0