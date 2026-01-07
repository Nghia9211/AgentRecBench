import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import argparse  # Thư viện để xử lý tham số dòng lệnh

from layer import LightGCN_3Hop
from utils import sample_bpr_batch, plot_loss
from bprloss import BPRLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Training LightGCN 3-Hop Model")
    
    # Data paths
    parser.add_argument('--data_file', type=str, default='processed_graph_data.pt', 
                        help='Đường dẫn file dữ liệu graph đã xử lý (.pt)')
    parser.add_argument('--export_file', type=str, default='gcn_embeddings_3hop.pt', 
                        help='Đường dẫn file để lưu embeddings kết quả (.pt)')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=1000, help='Số lượng Epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Kích thước Batch')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--reg', type=float, default=1e-4, help='Hệ số Regularization (Weight decay)')
    parser.add_argument('--emb_dim', type=int, default=64, help='Kích thước vector Embedding')
    
    # System
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='Thiết bị training (cuda hoặc cpu)')

    return parser.parse_args()

def main():
    args = parse_args()

    # Cấu hình thiết bị
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Training on: {device}")
    print(f"Settings: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}, Dim={args.emb_dim}")

    # Load data
    try:
        data = torch.load(args.data_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{args.data_file}'")
        return

    train_dict = data['train_dict']
    node_item_mapping = data['node_item_mapping']
    A_hat = data['adj_norm']
    num_users = data['num_users']
    num_nodes = data['num_nodes']

    # Khởi tạo model
    model = LightGCN_3Hop(num_nodes, args.emb_dim).to(device)
    A_hat = A_hat.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BPRLoss(reg_weight=args.reg)

    print("\n--- Start Training (LightGCN 3-Hop) ---")
    model.train()
    start_time = time.time()

    loss_history = []
    reg_history = []

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        final_embs, initial_embs = model(A_hat)

        users_idx, pos_idx, neg_idx = sample_bpr_batch(train_dict, num_users, num_nodes, args.batch_size)

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
            print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {loss.item():.4f}")

    print(f"Training finished in {time.time() - start_time:.2f}s")
    plot_loss(loss_history, reg_history)
    print("Save plot Loss Figure")

    # Lưu kết quả
    model.eval()
    with torch.no_grad():
        final_node_embeddings, _ = model(A_hat)
        
        final_dict = {}
        for idx, original_id in enumerate(node_item_mapping):
            final_dict[original_id] = final_node_embeddings[idx].cpu()

        torch.save(final_dict, args.export_file)
        print(f"--> Saved 3-hop embeddings to {args.export_file}")
        print(f"--> Vector Shape: {final_node_embeddings.shape}")

if __name__ == "__main__":
    main()