import torch
import numpy as np
from typing import Dict, List, Tuple, Union

from config import RetrievalConfig, DEFAULT_CONFIG
from seq_scorer import SeqScorer, _to_int, _to_int_list
from gcn_scorer import GCNScorer
from semantic_scorer import SemanticScorer

class CandidateRetriever:
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

    @staticmethod
    def _safe_seq(seq) -> List[int]:
        return _to_int_list(seq)

    @staticmethod
    def _safe_len(len_seq) -> int:
        return _to_int(len_seq)

    @staticmethod
    def _safe_cans(candidate_ids) -> List[int]:
        if isinstance(candidate_ids, torch.Tensor):
            return candidate_ids.cpu().tolist()
        if isinstance(candidate_ids, np.ndarray):
            return candidate_ids.tolist()
        return [int(x) for x in candidate_ids]

    def retrieve(
        self,
        seq:           Union[List[int], torch.Tensor, np.ndarray],
        len_seq:       Union[int, torch.Tensor, np.integer],
        candidate_ids: Union[List[int], torch.Tensor, np.ndarray],
        seq_str:       str = "",
        data:          dict = None,
    ) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        seq_list    = self._safe_seq(seq)
        len_seq_int = self._safe_len(len_seq)
        cans_list   = self._safe_cans(candidate_ids)

        signal_scores: Dict[str, Dict[str, float]] = {}

        seq_scores: Dict[str, float] = {}
        c_seq: List[str] = []
        if self.use_seq and self.seq_scorer is not None:
            try:
                seq_scores = self.seq_scorer.score(seq_list, len_seq_int, cans_list)
                c_seq = sorted(seq_scores, key=seq_scores.get, reverse=True)[:self.cfg.top_seq]
            except Exception as e: print(f"[CandidateRetriever] SeqScorer error: {e}")

        gcn_scores: Dict[str, float] = {}
        c_gcn: List[str] = []
        if self.use_gcn and self.gcn_scorer is not None:
            try:
                gcn_scores = self.gcn_scorer.score(seq_list, len_seq_int, cans_list)
                c_gcn = sorted(gcn_scores, key=gcn_scores.get, reverse=True)[:self.cfg.top_gcn]
            except Exception as e: print(f"[CandidateRetriever] GCNScorer error: {e}")

        sem_scores: Dict[str, float] = {}
        c_sem: List[str] = []
        if self.use_semantic and self.sem_scorer is not None:
            try:
                sem_scores = self.sem_scorer.score(seq_str, cans_list, data)
                c_sem = sorted(sem_scores, key=sem_scores.get, reverse=True)[:self.cfg.top_sem]
            except Exception as e: print(f"[CandidateRetriever] SemanticScorer error: {e}")

        seen = set()
        union_names: List[str] = []
        for name in c_seq + c_gcn + c_sem:
            if name not in seen:
                seen.add(name)
                union_names.append(name)

        if not union_names:
            id2name = self.seq_scorer.id2name if self.seq_scorer else self.gcn_scorer.id2name if self.gcn_scorer else {}
            union_names = [id2name.get(cid, f"item_{cid}") for cid in cans_list]

        for name in union_names:
            signal_scores[name] = {
                'seq': seq_scores.get(name, 0.0),
                'gcn': gcn_scores.get(name, 0.0),
                'sem': sem_scores.get(name, 0.0),
            }

        return union_names, signal_scores

    def retrieve_from_data(self, data: dict) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        return self.retrieve(
            seq           = data['seq'],
            len_seq       = data['len_seq'],
            candidate_ids = data['cans'],
            seq_str       = data.get('seq_str', ''),
            data          = data,
        )