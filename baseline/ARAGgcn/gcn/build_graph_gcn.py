import json
import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp

def build_graph_with_networkx(user_file, item_file, review_file):
    print("--- 1. INITAILIZE GRAPH NETWORKX ---")
    G = nx.Graph() 
    
    print("Loading Users...")
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            G.add_node(data['user_id'], node_type='user')

    print("Loading Items...")
    item_ids_list = [] 
    with open(item_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            G.add_node(data['item_id'], node_type='item')
            item_ids_list.append(data['item_id'])

    print(f"Total Initialize Nodes: {G.number_of_nodes()}")
    print("Loading Reviews (Edges)...")
    
    edge_count = 0
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                u = data.get('user_id')
                i = data.get('item_id')
                
                if u in G and i in G:
                    G.add_edge(u, i)
                    edge_count += 1
    except FileNotFoundError:
        print("Not Found Review File .json!")

    print(f"Total Edges: {edge_count}")

    print("\n--- 2. CONVERTING TO PYTORCH FORMAT ---")
    
    print("Mapping String IDs -> Integer IDs...")
    G_int = nx.convert_node_labels_to_integers(G, label_attribute='original_id')
    
    num_nodes = G_int.number_of_nodes()
    
    node_mapping = [G_int.nodes[i]['original_id'] for i in range(num_nodes)]

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
    
    features = torch.randn(num_nodes, 64) 
    
    labels = torch.randint(0, 5, (num_nodes,))
    save_data = {
        'adj_norm': adj_tensor,
        'features': features,
        'labels': labels,
        'node_item_mapping': node_mapping, 
        'input_dim': 64,
        'num_classes': 5
    }
    
    torch.save(save_data, 'processed_graph_data.pt')
    print("\n✅ Done! File saved: processed_graph_data.pt")


build_graph_with_networkx('user_amazon.json', 'item_amazon.json', 'review_amazon.json')