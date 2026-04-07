"""
moe_fusion/candidate_retriever.py
──────────────────────────────────
Candidate Retrieval:
    C_R = C_seq ∪ C_gcn ∪ C_sem

Fix:
  - seq và len_seq từ DataLoader là tensor → ép về List[int] / int
    trước khi truyền vào bất kỳ scorer nào, tránh lỗi:
    "only integer tensors of a single element can be converted to an index"
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union

from config import RetrievalConfig, DEFAULT_CONFIG
from seq_scorer import SeqScorer, _to_int, _to_int_list
from gcn_scorer import GCNScorer
from semantic_scorer import SemanticScorer


class CandidateRetriever:
    """
    Union candidates từ 3 nguồn:
      C_seq: top-K từ SASRec sequential model
      C_gcn: top-K từ GCN graph model
      C_sem: top-K từ FAISS semantic search
      C_R   = C_seq ∪ C_gcn ∪ C_sem  (deduplicated)
    """

    def __init__(
        self,
        seq_scorer:   SeqScorer,
        gcn_scorer:   GCNScorer,
        sem_scorer:   SemanticScorer,
        config:       RetrievalConfig = None,
        use_seq:      bool = True,
        use_gcn:      bool = True,
        use_semantic: bool = True,
    ):
        self.seq_scorer   = seq_scorer
        self.gcn_scorer   = gcn_scorer
        self.sem_scorer   = sem_scorer
        self.cfg          = config or DEFAULT_CONFIG.retrieval
        self.use_seq      = use_seq
        self.use_gcn      = use_gcn
        self.use_semantic = use_semantic

    # ─────────────────────────────────────────────────────────────────────
    # Internal: safe type conversion
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_seq(seq) -> List[int]:
        """
        Chấp nhận tensor (từ DataLoader), ndarray, hoặc list.
        Luôn trả về List[int] thuần Python.
        """
        return _to_int_list(seq)

    @staticmethod
    def _safe_len(len_seq) -> int:
        """
        Chấp nhận tensor scalar, np.integer, hoặc int.
        Luôn trả về Python int.
        """
        return _to_int(len_seq)

    @staticmethod
    def _safe_cans(candidate_ids) -> List[int]:
        """Ép candidate_ids về List[int]."""
        if isinstance(candidate_ids, torch.Tensor):
            return candidate_ids.cpu().tolist()
        if isinstance(candidate_ids, np.ndarray):
            return candidate_ids.tolist()
        return [int(x) for x in candidate_ids]

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        seq:           Union[List[int], torch.Tensor, np.ndarray],
        len_seq:       Union[int, torch.Tensor, np.integer],
        candidate_ids: Union[List[int], torch.Tensor, np.ndarray],
        seq_str:       str = "",
        data:          dict = None,
    ) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        """
        Lấy candidates từ 3 nguồn và union lại.

        Args:
            seq:           padded user history — tensor / ndarray / list
            len_seq:       actual sequence length — tensor scalar / int
            candidate_ids: pool of candidate item IDs — tensor / list
            seq_str:       user history string (cho semantic search)
            data:          full data dict (optional)

        Returns:
            union_names:   List[str] tên items trong C_R (deduplicated)
            signal_scores: Dict[item_name → {seq, gcn, sem}]
        """
        # ── Type-safe conversion (fix lỗi tensor index) ───────────────────
        seq_list    = self._safe_seq(seq)
        len_seq_int = self._safe_len(len_seq)
        cans_list   = self._safe_cans(candidate_ids)

        signal_scores: Dict[str, Dict[str, float]] = {}

        # ── 1. Sequential (SASRec) ────────────────────────────────────────
        seq_scores: Dict[str, float] = {}
        c_seq: List[str] = []
        if self.use_seq and self.seq_scorer is not None:
            try:
                seq_scores = self.seq_scorer.score(seq_list, len_seq_int, cans_list)
                c_seq = sorted(seq_scores, key=seq_scores.get, reverse=True)
                c_seq = c_seq[:self.cfg.top_seq]
                print(f"[CandidateRetriever] C_seq: {len(c_seq)} items")
            except Exception as e:
                print(f"[CandidateRetriever] SeqScorer error: {e}")
                import traceback; traceback.print_exc()

        # ── 2. GCN (graph collaborative) ─────────────────────────────────
        gcn_scores: Dict[str, float] = {}
        c_gcn: List[str] = []
        if self.use_gcn and self.gcn_scorer is not None:
            try:
                gcn_scores = self.gcn_scorer.score(seq_list, len_seq_int, cans_list)
                c_gcn = sorted(gcn_scores, key=gcn_scores.get, reverse=True)
                c_gcn = c_gcn[:self.cfg.top_gcn]
                print(f"[CandidateRetriever] C_gcn: {len(c_gcn)} items")
            except Exception as e:
                print(f"[CandidateRetriever] GCNScorer error: {e}")
                import traceback; traceback.print_exc()

        # ── 3. Semantic (FAISS) ───────────────────────────────────────────
        sem_scores: Dict[str, float] = {}
        c_sem: List[str] = []
        if self.use_semantic and self.sem_scorer is not None:
            try:
                sem_scores = self.sem_scorer.score(seq_str, cans_list, data)
                c_sem = sorted(sem_scores, key=sem_scores.get, reverse=True)
                c_sem = c_sem[:self.cfg.top_sem]
                print(f"[CandidateRetriever] C_sem: {len(c_sem)} items")
            except Exception as e:
                print(f"[CandidateRetriever] SemanticScorer error: {e}")
                import traceback; traceback.print_exc()

        # ── 4. Union (seq first → gcn → sem) ─────────────────────────────
        seen        = set()
        union_names: List[str] = []
        for name in c_seq + c_gcn + c_sem:
            if name not in seen:
                seen.add(name)
                union_names.append(name)

        # Fallback: union rỗng → dùng toàn bộ candidate pool
        if not union_names:
            id2name = (
                self.seq_scorer.id2name if self.seq_scorer
                else self.gcn_scorer.id2name if self.gcn_scorer
                else {}
            )
            union_names = [id2name.get(cid, f"item_{cid}") for cid in cans_list]

        # ── 5. Merge signal scores ────────────────────────────────────────
        for name in union_names:
            signal_scores[name] = {
                'seq': seq_scores.get(name, 0.0),
                'gcn': gcn_scores.get(name, 0.0),
                'sem': sem_scores.get(name, 0.0),
            }

        print(f"[CandidateRetriever] C_R = {len(union_names)} items "
              f"(seq={len(c_seq)}, gcn={len(c_gcn)}, sem={len(c_sem)})")
        return union_names, signal_scores

    def retrieve_from_data(
        self,
        data: dict,
    ) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        """
        Convenience wrapper: lấy inputs trực tiếp từ data dict.
        data phải có: 'seq', 'len_seq', 'cans', 'seq_str'
        """
        return self.retrieve(
            seq           = data['seq'],
            len_seq       = data['len_seq'],
            candidate_ids = data['cans'],
            seq_str       = data.get('seq_str', ''),
            data          = data,
        )