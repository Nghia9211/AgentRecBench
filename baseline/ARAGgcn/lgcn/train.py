import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from layer import LightGCN_3Hop
from utils import sample_bpr_batch, plot_loss
from bprloss import BPRLoss

# Load data
DATA_FILE = 'processed_graph_data.pt'
data = torch.load(DATA_FILE)

train_dict = data['train_dict']
node_item_mapping = data['node_item_mapping']
A_hat = data['adj_norm']
num_users = data['num_users']
num_nodes = data['num_nodes']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

embedding_dim = 64
model = LightGCN_3Hop(num_nodes, embedding_dim).to(device)
A_hat = A_hat.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = BPRLoss(reg_weight=1e-4)

print("\n--- Start Training (LightGCN 3-Hop) ---")
model.train()
start_time = time.time()

BATCH_SIZE = 1024
EPOCHS = 1000

loss_history = []
reg_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    final_embs, initial_embs = model(A_hat)

    users_idx, pos_idx, neg_idx = sample_bpr_batch(train_dict, num_users, num_nodes, BATCH_SIZE)

    users_idx = users_idx.to(device)
    pos_idx = pos_idx.to(device)
    neg_idx = neg_idx.to(device)

    u_final = final_embs[users_idx]
    i_pos_final = final_embs[pos_idx]
    i_neg_final = final_embs[neg_idx]
    
    u_0 = initial_embs[users_idx]
    i_pos_0 = initial_embs[pos_idx]
    i_neg_0 = initial_embs[neg_idx]

    loss, bpr, reg = criterion(users_idx, pos_idx, neg_idx,
                               u_final, i_pos_final, i_neg_final,
                               u_0, i_pos_0, i_neg_0)

    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    reg_history.append(reg.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

print(f"Training finished in {time.time() - start_time:.2f}s")

plot_loss(loss_history, reg_history)
print("Save plot Loss Figure")

EXPORT_FILE = 'gcn_embeddings_3hop.pt'

model.eval()
with torch.no_grad():
    final_node_embeddings, _ = model(A_hat)
    
    final_dict = {}
    for idx, original_id in enumerate(node_item_mapping):
        final_dict[original_id] = final_node_embeddings[idx].cpu()

    torch.save(final_dict, EXPORT_FILE)
    print(f"--> Saved 3-hop embeddings to {EXPORT_FILE}")
    print(f"--> Vector Shape: {final_node_embeddings.shape}")


