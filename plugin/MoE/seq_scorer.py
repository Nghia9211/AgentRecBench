"""
moe_fusion/seq_scorer.py
────────────────────────
Sequential scorer: tính s_seq(u, i) cho từng candidate item
dựa trên SASRec model đã được train.

Fix:
  - len_seq và seq luôn được ép kiểu về Python int / List[int]
    trước khi truyền vào forward_eval, tránh lỗi:
    "only integer tensors of a single element can be converted to an index"
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union


def _to_int(x) -> int:
    """Ép bất kỳ scalar (tensor, np.integer, float, int) về Python int."""
    if isinstance(x, torch.Tensor):
        return int(x.item())
    if isinstance(x, np.integer):
        return int(x)
    return int(x)


def _to_int_list(seq) -> List[int]:
    """Ép sequence (tensor, ndarray, list) về List[int]."""
    if isinstance(seq, torch.Tensor):
        return seq.cpu().tolist()
    if isinstance(seq, np.ndarray):
        return seq.tolist()
    return [_to_int(x) for x in seq]


class SeqScorer:
    """
    Wrap SASRec để tính sequential score cho từng (user, item) pair.

    Interface:
        scorer = SeqScorer(sasrec_model, id2name, item_num, device)
        scores = scorer.score(seq, len_seq, candidate_ids)
        # → Dict[item_name, float]
    """

    def __init__(
        self,
        sasrec_model,
        id2name:    Dict[int, str],
        item_num:   int,
        device:     torch.device,
    ):
        self.model    = sasrec_model
        self.id2name  = id2name
        self.item_num = item_num
        self.device   = device

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def score(
        self,
        seq:           Union[List[int], torch.Tensor, np.ndarray],
        len_seq:       Union[int, torch.Tensor, np.integer],
        candidate_ids: List[int],
    ) -> Dict[str, float]:
        """
        Tính raw SASRec score cho từng candidate.

        Args:
            seq:           padded sequence — list, tensor, hoặc ndarray đều OK
            len_seq:       actual sequence length — int, tensor scalar, np.integer đều OK
            candidate_ids: list of inner item IDs cần score

        Returns:
            Dict[item_name → raw_score]  (chưa normalize)
        """
        if not candidate_ids:
            return {}

        # ── Type-safe conversion ──────────────────────────────────────────
        seq_list    = _to_int_list(seq)
        len_seq_int = _to_int(len_seq)

        states = torch.LongTensor([seq_list]).to(self.device)
        with torch.no_grad():
            logits = self.model.forward_eval(
                states,
                np.array([len_seq_int], dtype=np.int64),
            )

        # Mask toàn bộ items không phải candidate
        mask = torch.ones(self.item_num, dtype=torch.bool)
        for cid in candidate_ids:
            if 0 <= cid < self.item_num:
                mask[cid] = False
        mask_2d = mask.unsqueeze(0)  # (1, item_num)

        logits_cpu    = logits.cpu().detach()
        logits_masked = logits_cpu.masked_fill(
            mask_2d, logits_cpu.min().item() - 1
        )

        score_dict: Dict[str, float] = {}
        for cid in candidate_ids:
            if 0 <= cid < self.item_num:
                name = self.id2name.get(cid, f"item_{cid}")
                score_dict[name] = logits_masked[0, cid].item()

        return score_dict

    def top_k_names(
        self,
        seq:           Union[List[int], torch.Tensor, np.ndarray],
        len_seq:       Union[int, torch.Tensor, np.integer],
        candidate_ids: List[int],
        k:             int = 20,
    ) -> List[str]:
        """Top-k item names (sorted desc) từ candidate_ids."""
        scores = self.score(seq, len_seq, candidate_ids)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def top_k_ids(
        self,
        seq:           Union[List[int], torch.Tensor, np.ndarray],
        len_seq:       Union[int, torch.Tensor, np.integer],
        candidate_ids: List[int],
        k:             int = 20,
    ) -> List[int]:
        """Top-k item IDs (sorted desc)."""
        scores   = self.score(seq, len_seq, candidate_ids)
        name2id  = {v: k for k, v in self.id2name.items()}
        return [
            name2id[n] for n in
            sorted(scores, key=scores.get, reverse=True)[:k]
            if n in name2id
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_shared(cls, shared: dict) -> "SeqScorer":
        return cls(
            sasrec_model=shared['model'],
            id2name=shared['id2name'],
            item_num=shared['item_num'],
            device=shared['device'],
        )