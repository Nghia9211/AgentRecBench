"""
moe_fusion/gcn_scorer.py
────────────────────────
GCN collaborative scorer: tính s_gcn(u, i) dựa trên graph embeddings.

GCN embeddings được load sẵn trong ARAGRecAgent._shared_gcn_embeddings
(tensor shape: [item_num, embed_dim]).

s_gcn(u, i) = cosine_similarity(h_user, h_item)
  với h_user = weighted mean của item embeddings trong user history.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class GCNScorer:
    """
    Tính GCN-based collaborative score cho từng candidate item.

    Flow:
      1. Lấy GCN embedding của từng item trong user history
      2. h_user = weighted mean (gần đây → weight cao hơn)
      3. s_gcn(i) = cosine_sim(h_user, gcn_emb[i])

    Interface:
        scorer = GCNScorer(gcn_embeddings, id2name, name2id)
        scores = scorer.score(seq, len_seq, candidate_ids)
        # → Dict[item_name, float]
    """

    def __init__(
        self,
        gcn_embeddings: torch.Tensor,   # shape: [num_items+1, embed_dim]
        id2name:        Dict[int, str],
        name2id:        Dict[str, int],
        device:         torch.device    = None,
    ):
        self.device = device or torch.device("cpu")

        gcn_emb = gcn_embeddings.to(self.device).float()
        # gcn_emb = gcn_embeddings.to(self.device).float()
        norms   = gcn_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.gcn_norm = gcn_emb / norms            # (num_items, dim)

        self.id2name  = id2name
        self.name2id  = name2id
        self.num_items = gcn_emb.shape[0]

    # ─────────────────────────────────────────────────────────────────────
    # User representation
    # ─────────────────────────────────────────────────────────────────────

    def _user_embedding(self, seq: List[int], len_seq: int) -> Optional[torch.Tensor]:
        """
        Tổng hợp GCN embedding của user từ interaction history.
        Dùng exponential decay weighting: item gần đây có weight cao hơn.

        Returns:
            Normalized user embedding (dim,) hoặc None nếu history rỗng.
        """
        if len_seq == 0:
            return None

        # Lấy phần thực sự (bỏ padding)
        actual_seq = seq[-len_seq:] if len_seq <= len(seq) else seq
        # Lọc valid item ids
        valid_ids = [iid for iid in actual_seq
                     if 0 <= iid < self.num_items]
        if not valid_ids:
            return None

        # Exponential decay: item index lớn hơn (gần đây hơn) → weight cao
        n = len(valid_ids)
        weights = torch.tensor(
            [np.exp(0.1 * i) for i in range(n)],
            dtype=torch.float32,
            device=self.device,
        )
        weights = weights / weights.sum()              # normalize

        # Lấy embeddings
        ids_tensor = torch.tensor(valid_ids, dtype=torch.long, device=self.device)
        item_embs  = self.gcn_norm[ids_tensor]         # (n, dim)

        # Weighted mean
        h_user = (weights.unsqueeze(-1) * item_embs).sum(dim=0)  # (dim,)
        h_user = F.normalize(h_user, dim=0)
        return h_user

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def score(
        self,
        seq:           List[int],
        len_seq:       int,
        candidate_ids: List[int],
    ) -> Dict[str, float]:
        """
        Tính GCN score cho từng candidate.

        Returns:
            Dict[item_name → cosine_score ∈ [-1, 1]]
            Trả về dict rỗng nếu user không có history.
        """
        if not candidate_ids:
            return {}

        h_user = self._user_embedding(seq, len_seq)
        if h_user is None:
            # Cold-start: fallback về uniform score
            return {self.id2name.get(cid, f"item_{cid}"): 0.0
                    for cid in candidate_ids
                    if 0 <= cid < self.num_items}

        # Lấy embeddings cho candidates
        valid_cans = [(cid, self.id2name.get(cid, f"item_{cid}"))
                      for cid in candidate_ids
                      if 0 <= cid < self.num_items]
        if not valid_cans:
            return {}

        can_ids    = torch.tensor([c[0] for c in valid_cans],
                                  dtype=torch.long, device=self.device)
        can_embs   = self.gcn_norm[can_ids]            # (n_cans, dim)

        # Cosine similarity = dot product (vì đã normalize)
        sims = (can_embs @ h_user).cpu().tolist()      # (n_cans,)

        return {name: float(sim)
                for (_, name), sim in zip(valid_cans, sims)}

    def top_k_names(
        self,
        seq:           List[int],
        len_seq:       int,
        candidate_ids: List[int],
        k:             int = 20,
    ) -> List[str]:
        """Top-k item names theo GCN score. Dùng để build C_gcn."""
        scores = self.score(seq, len_seq, candidate_ids)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    # ─────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_shared(cls, shared: dict) -> "GCNScorer":
        """
        Tạo GCNScorer từ dict shared resources.

        shared phải có keys:
          'gcn_embeddings', 'id2name', 'name2id', 'device'
        """
        gcn_emb = shared.get('gcn_embeddings')
        if gcn_emb is None:
            raise ValueError(
                "[GCNScorer] gcn_embeddings not found in shared resources. "
                "Ensure args.gcn_path is set và file tồn tại."
            )
        return cls(
            gcn_embeddings=gcn_emb,
            id2name=shared['id2name'],
            name2id=shared['name2id'],
            device=shared.get('device', torch.device('cpu')),
        )
