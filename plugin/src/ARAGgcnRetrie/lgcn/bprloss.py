import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self, reg_weight = 1e-4):
        super().__init__()
        self.reg_weight = reg_weight
    
    def forward(self, users, pos_items, neg_items, 
                user_e_final, pos_e_final, neg_e_final, 
                user_e0, pos_e0, neg_e0):
        pos_scores = torch.mul(user_e_final, pos_e_final).sum(dim=1)
        neg_scores = torch.mul(user_e_final, neg_e_final).sum(dim=1)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = (1/2)*(user_e0.norm(2).pow(2) +
                          pos_e0.norm(2).pow(2) + 
                          neg_e0.norm(2).pow(2)) / users.shape[0]
        
        total_loss = loss + self.reg_weight * reg_loss

        return total_loss, loss, reg_loss