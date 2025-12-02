import json
import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp
import argparse
import os
from utils import load_groundtruth_pairs





def build_graph_with_masking(user_file, item_file, review_file, gt_folder, output_file):
    print("\n--- 1. INITIALIZE GRAPH NETWORKX (WITH MASKING) ---")
    G = nx.Graph() 
    
    # 1. Load User
    print("Loading Users...")
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            G.add_node(str(data['user_id']), node_type='user')

    # 2. Load Item
    print("Loading Items...")
    with open(item_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            G.add_node(str(data['item_id']), node_type='item')

    # 3. Load Mask List
    masked_pairs = load_groundtruth_pairs(gt_folder)

    # 4. Load Reviews & Apply Mask
    print("Loading Reviews and Building Edges...")
    edge_count = 0
    skipped_count = 0
    
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                u = str(data.get('user_id'))
                i = str(data.get('item_id'))
                
                # --- LOGIC MASKING ---
                if (u, i) in masked_pairs:
                    skipped_count += 1
                    continue 
                # ---------------------

                if u in G and i in G:
                    G.add_edge(u, i)
                    edge_count += 1
                
    except FileNotFoundError:
        print("Error: Review file not found!")
        return

    print(f"Total Edges Added: {edge_count}")
    print(f"Total Edges Masked (Skipped): {skipped_count}")

    print("\n--- 2. CONVERTING TO PYTORCH FORMAT ---")
    
    print("Mapping String IDs -> Integer IDs...")
    G_int = nx.convert_node_labels_to_integers(G, label_attribute='original_id')
    
    num_nodes = G_int.number_of_nodes()
    node_mapping = [G_int.nodes[i]['original_id'] for i in range(num_nodes)]

    print("Generating Interaction Dictionary...")
    train_dict = {}
    num_users = 0
    
    for node_idx in G_int.nodes():
        node_data = G_int.nodes[node_idx]
        if node_data.get('node_type') == 'user':
            num_users += 1
            neighbors = list(G_int.neighbors(node_idx))
            if len(neighbors) > 0:
                train_dict[node_idx] = neighbors

    print(f"Detected {num_users} users with interactions.")

    print("Creating Adjacency Matrix...")
    A = nx.to_scipy_sparse_array(G_int, format='coo') 

    print("Normalizing Adjacency Matrix...")
    I = sp.eye(A.shape[0])
    A_hat = A + I 
    
    degrees = np.array(A_hat.sum(1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_norm = sp.diags(d_inv_sqrt)
    
    A_norm = D_norm @ A_hat @ D_norm 
    A_norm = A_norm.tocoo() 

    indices = torch.from_numpy(
        np.vstack((A_norm.row, A_norm.col)).astype(np.int64)
    )
    values = torch.from_numpy(A_norm.data.astype(np.float32))
    shape = torch.Size(A_norm.shape)
    
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape)
    
    save_data = {
        'adj_norm': adj_tensor,
        'train_dict': train_dict,    
        'node_item_mapping': node_mapping, 
        'num_users': num_users,
        'num_nodes': num_nodes
    }
    
    torch.save(save_data, output_file)
    print(f"\n✅ MASKED Graph saved to: {output_file}")
    print("You can now run 'train.py' using this file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Masked Graph for LightGCN")
    
    parser.add_argument('--user_file', type=str, required=True, help='Path to user.json')
    parser.add_argument('--item_file', type=str, required=True, help='Path to item.json')
    parser.add_argument('--review_file', type=str, required=True, help='Path to review.json (FULL data)')
    parser.add_argument('--gt_folder', type=str, required=True, help='Folder containing groundtruth_*.json files')
    parser.add_argument('--output_file', type=str, default='processed_graph_masked.pt', help='Output .pt file')

    args = parser.parse_args()

    build_graph_with_masking(
        user_file=args.user_file,
        item_file=args.item_file,
        review_file=args.review_file,
        gt_folder=args.gt_folder,
        output_file=args.output_file,
    )