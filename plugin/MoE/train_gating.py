"""
moe_fusion/train_gating.py
───────────────────────────
Script train MLP Gating Network offline từ interaction data.

Fixes so với version cũ:
  - Bơm InteractionTool vào lúc train để Semantic Scorer có thể 
    truy xuất Review Text của user cho TẤT CẢ dataset.
  - [Fix MỚI]: Thêm bộ lọc chống Data Leakage. Đảm bảo ground-truth item 
    và các candidate items bị ẩn khỏi review history lúc train (giống hệt 
    lúc inference).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
root_dir    = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, root_dir)

baseline_dir = os.path.join(root_dir, 'baseline')
if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)

from config           import MoEConfig, GatingConfig, DEFAULT_CONFIG
from gating_network   import GatingNetwork, GatingMLP
from seq_scorer       import SeqScorer
from gcn_scorer       import GCNScorer
from semantic_scorer  import SemanticScorer
from dataset.general_dataset import GeneralDataset
from websocietysimulator.tools import CacheInteractionTool


def compute_batch_scores_optimized(
    batch_seqs,      
    batch_lens,      
    batch_queries,   
    batch_pools,     
    seq_scorer,
    gcn_scorer,
    sem_scorer,
    id2name
):
    B = len(batch_seqs)
    device = seq_scorer.device
    
    # --- 1. Batch Sequential Scores (SASRec) ---
    with torch.no_grad():
        batch_logits = seq_scorer.model.forward_eval(batch_seqs, batch_lens.cpu().numpy()) 
    
    # --- 2. Batch GCN Scores ---
    batch_gcn_scores = []
    if gcn_scorer:
        with torch.no_grad():
            h_users = []
            for i in range(B):
                h_u = gcn_scorer._user_embedding(batch_seqs[i].tolist(), int(batch_lens[i]))
                h_users.append(h_u if h_u is not None else torch.zeros(gcn_scorer.gcn_norm.shape[1], device=device))
            h_users = torch.stack(h_users) 
            all_gcn_logits = h_users @ gcn_scorer.gcn_norm.T 
    
    # --- 3. Batch Semantic Scores ---
    batch_sem_logits_map = [{} for _ in range(B)]
    if sem_scorer and sem_scorer.embedding_function:
        all_unique_names = set()
        for pool in batch_pools:
            for cid in pool:
                all_unique_names.add(id2name.get(cid, f"item_{cid}"))
        
        unique_names_list = list(all_unique_names)
        
        q_vecs = np.array(sem_scorer.embedding_function.embed_documents(batch_queries), dtype=np.float32)
        d_vecs = np.array(sem_scorer.embedding_function.embed_documents(unique_names_list), dtype=np.float32) 
        
        q_vecs = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8)
        d_vecs = d_vecs / (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-8)
        sim_matrix = q_vecs @ d_vecs.T 
        
        name_to_idx = {name: idx for idx, name in enumerate(unique_names_list)}
        for i in range(B):
            for cid in batch_pools[i]:
                name = id2name.get(cid, f"item_{cid}")
                batch_sem_logits_map[i][name] = sim_matrix[i, name_to_idx[name]]

    # Gom kết quả
    final_scores = []
    for i in range(B):
        pool = batch_pools[i]
        s_seq = {id2name[cid]: batch_logits[i, cid].item() for cid in pool if cid in id2name}
        s_gcn = {id2name[cid]: all_gcn_logits[i, cid].item() for cid in pool if cid in id2name} if gcn_scorer else {}
        s_sem = batch_sem_logits_map[i]
        final_scores.append((s_seq, s_gcn, s_sem))
        
    return final_scores

def _get_signal_features(item_name: str, seq_sc: Dict, gcn_sc: Dict, sem_sc: Dict) -> List[float]:
    return [seq_sc.get(item_name, 0.0), gcn_sc.get(item_name, 0.0), sem_sc.get(item_name, 0.0)]

def _sample_hard_negatives(gt_name, seq_sc, gcn_sc, sem_sc, all_names, n_neg, hard_ratio=0.5):
    non_gt = [n for n in all_names if n != gt_name]
    if not non_gt: return []

    n_hard = max(1, int(n_neg * hard_ratio))
    n_easy = n_neg - n_hard
    top_k = max(n_neg, 10)
    hard_pool = set()

    for sc_dict in [seq_sc, gcn_sc, sem_sc]:
        if sc_dict:
            top_items = sorted([n for n in sc_dict if n != gt_name and n in set(non_gt)], 
                               key=sc_dict.get, reverse=True)[:top_k]
            hard_pool.update(top_items)

    hard_pool = list(hard_pool)
    np.random.shuffle(hard_pool)
    hard_negs = hard_pool[:n_hard]

    easy_pool = [n for n in non_gt if n not in set(hard_negs)]
    np.random.shuffle(easy_pool)
    easy_negs = easy_pool[:n_easy]

    return hard_negs + easy_negs

def print_feature_report(X: np.ndarray, name: str = "Data"):
    print(f"\n=== Statistical Report for {name} (N={len(X)}) ===")
    print(f"{'Signal':<10} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'% Non-Zero':<10}")
    print("-" * 65)
    signals = ['Sequential', 'GCN', 'Semantic']
    for i in range(3):
        col = X[:, i]
        non_zero = (np.abs(col) > 1e-9).mean() * 100
        print(f"{signals[i]:<10} | {col.mean():.4f} | {col.std():.4f} | {col.min():.4f} | {col.max():.4f} | {non_zero:>9.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation (CẬP NHẬT TRUYỀN INTERACTION TOOL & LỌC LEAKAGE)
# ─────────────────────────────────────────────────────────────────────────────
def build_training_data_fast(
    loader:     DataLoader,
    seq_scorer: SeqScorer,
    gcn_scorer: Optional[GCNScorer],
    sem_scorer: Optional[SemanticScorer],
    id2name:    Dict[int, str],
    interaction_tool = None,
    dataset_name: str = 'amazon',
    n_neg:      int = 5,
    hard_ratio: float = 0.5,
    mode:       str = 'bpr'
) -> Tuple:
    X_list, y_list, X_pos_list, X_neg_list = [], [], [], []
    
    pbar = tqdm(loader, desc=f"🚀 Building {mode.upper()} Data", total=len(loader), unit="batch")

    for batch in pbar:
        seqs = batch['seq'].to(seq_scorer.device)
        lens = batch['len_seq'].to(seq_scorer.device)
        next_ids = batch['next'].cpu().numpy()
        all_cans_batch = batch.get('cans')
        
        # Bắt linh hoạt key id từ dataset loader
        user_ids = batch.get('id', batch.get('user_id', [])) 

        batch_pools, batch_queries = [], []
        for i in range(len(next_ids)):
            gt_id = int(next_ids[i])
            cans = all_cans_batch[i].tolist() if all_cans_batch is not None else []
            pool = list(set(cans) | {gt_id})
            batch_pools.append(pool)
            
            seq_str = ' '.join(id2name[iid] for iid in seqs[i].tolist() if iid in id2name)
            
            # --- FIX: BƠM REVIEW TEXT & CHỐNG DATA LEAKAGE ---
            user_data_dict = {'dataset': dataset_name}
            
            if interaction_tool and len(user_ids) > i:
                u_id = str(user_ids[i].item() if torch.is_tensor(user_ids[i]) else user_ids[i])
                try:
                    all_reviews = interaction_tool.get_reviews(user_id=u_id)
                    
                    # QUAN TRỌNG: Lọc bỏ candidate items & gt_id khỏi lịch sử review
                    pool_str_ids = set(str(c) for c in pool)
                    filtered_reviews = [
                        r for r in all_reviews 
                        if str(r.get('item_id')) not in pool_str_ids
                    ]
                    
                    user_data_dict['reviews'] = filtered_reviews
                except:
                    pass
            
            # Semantic Scorer giờ sẽ build query một cách an toàn và giàu ngữ nghĩa
            query = sem_scorer._build_query(seq_str, data=user_data_dict) if sem_scorer else ""
            batch_queries.append(query)

        batch_results = compute_batch_scores_optimized(
            seqs, lens, batch_queries, batch_pools, 
            seq_scorer, gcn_scorer, sem_scorer, id2name
        )

        for i in range(len(next_ids)):
            gt_id = int(next_ids[i])
            gt_name = id2name.get(gt_id)
            if not gt_name: continue
            
            s_seq, s_gcn, s_sem = batch_results[i]
            pos_feat = _get_signal_features(gt_name, s_seq, s_gcn, s_sem)
            
            all_pool_names = [id2name[cid] for cid in batch_pools[i] if cid in id2name]
            neg_names = _sample_hard_negatives(
                gt_name, s_seq, s_gcn, s_sem, all_pool_names, n_neg, hard_ratio
            )
            
            if mode == 'bce':
                X_list.append(pos_feat)
                y_list.append(1.0)
                for n_name in neg_names:
                    X_list.append(_get_signal_features(n_name, s_seq, s_gcn, s_sem))
                    y_list.append(0.0)
            else: 
                for n_name in neg_names:
                    X_pos_list.append(pos_feat)
                    X_neg_list.append(_get_signal_features(n_name, s_seq, s_gcn, s_sem))

        total_so_far = len(X_list) if mode == 'bce' else len(X_pos_list)
        pbar.set_postfix({"samples": f"{total_so_far:,}"})

    pbar.close()

    if mode == 'bce':
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
    return np.array(X_pos_list, dtype=np.float32), np.array(X_neg_list, dtype=np.float32)

def normalize_features(X: np.ndarray):
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mu) / std, mu, std


# ─────────────────────────────────────────────────────────────────────────────
# Training loops (BPR/BCE giữ nguyên)
# ─────────────────────────────────────────────────────────────────────────────

def train_gating_bpr(X_pos, X_neg, cfg, device, val_ratio=0.1):
    n = len(X_pos)
    n_val = max(1, int(n * val_ratio))
    idx   = np.random.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    Xp_tr, Xn_tr = X_pos[train_idx], X_neg[train_idx]
    Xp_vl, Xn_vl = X_pos[val_idx],   X_neg[val_idx]

    ds     = TensorDataset(torch.tensor(Xp_tr, dtype=torch.float32), torch.tensor(Xn_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model     = GatingMLP(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xp_vl_t = torch.tensor(Xp_vl, dtype=torch.float32).to(device)
    Xn_vl_t = torch.tensor(Xn_vl, dtype=torch.float32).to(device)

    def bpr_loss(gates_pos, x_pos, gates_neg, x_neg):
        s_pos = (gates_pos * x_pos).sum(dim=-1)
        s_neg = (gates_neg * x_neg).sum(dim=-1)
        return -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()

    best_val, best_state = float('inf'), None

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xp, xn in loader:
            xp, xn = xp.to(device), xn.to(device)
            optimizer.zero_grad()
            loss = bpr_loss(model(xp), xp, model(xn), xn)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = bpr_loss(model(Xp_vl_t), Xp_vl_t, model(Xn_vl_t), Xn_vl_t).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"  [BPR] Epoch {epoch+1:02d}/{cfg.epochs} | train={total/max(len(loader),1):.4f} | val={val_loss:.4f}")

    if best_state: model.load_state_dict(best_state)
    print(f"Best BPR val loss: {best_val:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train MoE Gating Network")
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--raw_data_dir', default=None, help="Thư mục chứa review.json, item.json")
    parser.add_argument('--model_path', required=True, help="SASRec checkpoint")
    parser.add_argument('--gcn_path',   default=None)
    parser.add_argument('--faiss_path', default=None)
    parser.add_argument('--output_dir', default='./saved_models/moe')
    parser.add_argument('--dataset',    default='amazon', choices=['amazon', 'yelp', 'goodreads'])
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--n_neg',      type=int,   default=5)
    parser.add_argument('--hard_ratio', type=float, default=0.2)
    parser.add_argument('--loss',       default='bpr', choices=['bce', 'bpr'])
    parser.add_argument('--split',      default='val', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_gating] Device={device} | loss={args.loss}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Init Interaction Tool ─────────────────────────────────────────────
    interaction_tool = None
    if not args.raw_data_dir:
        args.raw_data_dir = os.path.join(root_dir, 'dataset', 'output_data_all')
    
    if os.path.exists(args.raw_data_dir):
        print(f"[train_gating] Initializing CacheInteractionTool from {args.raw_data_dir}")
        try:
            interaction_tool = CacheInteractionTool(data_dir=args.raw_data_dir)
        except Exception as e:
            print(f"[train_gating] Could not load tool: {e}")
    else:
        print("[train_gating] raw_data_dir not found, interaction tool disabled.")

    # ── Load SASRec ────────────────────────────────────────────────────────
    from utils.model import SASRec
    data_statis = pd.read_pickle(os.path.join(args.data_dir, 'data_statis.df'))
    seq_size, item_num = data_statis['seq_size'][0], data_statis['item_num'][0]

    sasrec = SASRec(64, item_num, seq_size, 0.1, device).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    sasrec.load_state_dict(ckpt.get('model_state_dict', ckpt))
    sasrec.eval()

    id2name, name2id = {}, {}
    with open(os.path.join(args.data_dir, 'id2name.txt'), encoding='utf-8') as f:
        for line in f:
            ll = line.strip().split('::', 1)
            if len(ll) == 2:
                id2name[int(ll[0])] = ll[1].strip()

    shared = {'model': sasrec, 'id2name': id2name, 'name2id': name2id, 'item_num': item_num, 'device': device}
    seq_scorer = SeqScorer.from_shared(shared)

    # ── Load GCN ────────────────────────────────────────────────────────
    gcn_scorer = None
    if args.gcn_path and os.path.exists(args.gcn_path):
        shared['gcn_embeddings'] = torch.load(args.gcn_path, map_location=device, weights_only=True)
        gcn_scorer = GCNScorer.from_shared(shared)

    # ── Load FAISS ──────────────────────────────────────────────────────
    sem_scorer = None
    if args.faiss_path and os.path.exists(args.faiss_path):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embed_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = FAISS.load_local(folder_path=args.faiss_path, embeddings=embed_fn, allow_dangerous_deserialization=True)
        shared.update({'vector_store': vs, 'embedding_function': embed_fn})
        sem_scorer = SemanticScorer.from_shared(shared)

    # ── Load dataset ───────────────────────────────────────────────────────
    class DatasetArgs:
        def __init__(self, data_dir): self.data_dir = data_dir
    dataset = GeneralDataset(DatasetArgs(args.data_dir), stage={'train':'train','val':'valid','test':'test'}[args.split])
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ── Build & Train ──────────────────────────────────────────────────────
    cfg = GatingConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    if args.loss == 'bpr':
        X_pos, X_neg = build_training_data_fast(
            loader, seq_scorer, gcn_scorer, sem_scorer, id2name,
            interaction_tool=interaction_tool, dataset_name=args.dataset,
            n_neg=args.n_neg, hard_ratio=args.hard_ratio, mode='bpr'
        )
        X_all_n, min_v, max_v = normalize_features(np.vstack([X_pos, X_neg]))
        X_pos_n, X_neg_n = X_all_n[:len(X_pos)], X_all_n[len(X_pos):]
        
        print_feature_report(X_pos, "Raw Positive Features")
        trained_mlp = train_gating_bpr(X_pos_n, X_neg_n, cfg, device)

    out_path = os.path.join(args.output_dir, f"{args.dataset}_gating_model.pt")
    torch.save({
        'model_state_dict': trained_mlp.state_dict(),
        'cfg': cfg,
        'norm_min': min_v.tolist(),
        'norm_max': max_v.tolist(),
    }, out_path)
    print(f"\n✅ Gating model saved to {out_path}")

    # Evaluate weights
    X_t = torch.tensor(X_pos_n, dtype=torch.float32).to(device)
    with torch.no_grad(): avg_g = trained_mlp(X_t).cpu().numpy().mean(axis=0)
    print(f"\nAverage gate weights on training set:")
    print(f"  g1(seq) = {avg_g[0]:.3f} | g2(gcn) = {avg_g[1]:.3f} | g3(sem) = {avg_g[2]:.3f}")


if __name__ == '__main__':
    main()