import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .layer import GCN

DATA_FILE = 'processed_graph_data.pt'
print(f"Load Data from {DATA_FILE}...")

data = torch.load(DATA_FILE)
node_item_mapping = data['node_item_mapping']
X = data['features']
Y = data['labels']
A_hat = data['adj_norm']
input_dim = data['input_dim']
num_classes = data['num_classes']

Y[Y < 0] = 0 
Y[Y >= num_classes] = num_classes - 1

print(f"Loaded -> Nodes: {X.shape[0]}, Features: {input_dim}")

device = torch.device('cpu') 
print(f"Training on: {device}")

model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=num_classes).to(device)

X = X.to(device)
Y = Y.to(device)
A_hat = A_hat.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
EPOCHS = 200 

print("\Start training...")
model.train()
start_time = time.time()

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    output = model(A_hat, X)
    loss = criterion(output, Y)

    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

print(f"Training in {time.time() - start_time:.2f}s")

EXPORT_FILE = 'gcn_embeddings.pt'

model.eval()
with torch.no_grad():
    node_embeddings = model.gc1(A_hat, X) 
    
    num_items = len(node_item_mapping)
    num_users = X.shape[0] - num_items
    
    item_embeddings_tensor = node_embeddings[num_users:]
    
    final_embeddings = {}
    for i, item_id in enumerate(node_item_mapping):
        final_embeddings[item_id] = item_embeddings_tensor[i].cpu()

    torch.save(final_embeddings, EXPORT_FILE)
    print(f"--> Save successfully {len(final_embeddings)} item embeddings.")
    print(f"--> Size of embedding vector : {list(final_embeddings.values())[0].shape}")