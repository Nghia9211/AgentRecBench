import numpy as np
from typing import Dict, List, Tuple, Optional

from config import MoEConfig, DEFAULT_CONFIG
from gating_network import GatingNetwork

def _zscore_normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict: return {}
    vals = np.array(list(score_dict.values()))
    mean_v, std_v = np.mean(vals), np.std(vals)
    if std_v == 0:
        return {k: 0.0 for k in score_dict}
    return {k: float((v - mean_v) / std_v) for k, v in score_dict.items()}

def _extract_signal(signal_scores: Dict[str, Dict[str, float]], signal_key: str) -> Dict[str, float]:
    return {name: scores.get(signal_key, 0.0) for name, scores in signal_scores.items()}

class MoEFusion:
    def __init__(self, gating: GatingNetwork, cfg: MoEConfig = None):
        self.gating = gating
        self.cfg    = cfg or DEFAULT_CONFIG

    def fuse(
        self,
        signal_scores: Dict[str, Dict[str, float]],
        len_seq: int = 0,
        top_m: int = None,
        debug: bool = False,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        
        top_m = top_m or self.cfg.retrieval.top_M
        if not signal_scores: return [], {}, {}

        raw_seq = _extract_signal(signal_scores, 'seq')
        raw_gcn = _extract_signal(signal_scores, 'gcn')
        raw_sem = _extract_signal(signal_scores, 'sem')

        norm_seq = _zscore_normalize(raw_seq)
        norm_gcn = _zscore_normalize(raw_gcn)
        norm_sem = _zscore_normalize(raw_sem)
        gate_weights = self.gating.predict(signal_scores, len_seq=len_seq)

        fused_scores: Dict[str, float] = {}
        for name in signal_scores:
            g1, g2, g3 = gate_weights.get(name, (1/3, 1/3, 1/3))
            s_seq = norm_seq.get(name, 0.0)
            s_gcn = norm_gcn.get(name, 0.0)
            s_sem = norm_sem.get(name, 0.0)

            if not self.cfg.use_seq: s_seq, g1 = 0.0, 0.0
            if not self.cfg.use_gcn: s_gcn, g2 = 0.0, 0.0
            if not self.cfg.use_semantic: s_sem, g3 = 0.0, 0.0

            total_g = g1 + g2 + g3
            if total_g > 0:
                g1, g2, g3 = g1/total_g, g2/total_g, g3/total_g

            fused_scores[name] = g1 * s_seq + g2 * s_gcn + g3 * s_sem

        c_m = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_m]

        debug_info = {}
        if debug:
            all_gates = list(gate_weights.values())
            avg_g1 = np.mean([g[0] for g in all_gates])
            avg_g2 = np.mean([g[1] for g in all_gates])
            avg_g3 = np.mean([g[2] for g in all_gates])

            debug_info = {
                'avg_gates':    {'seq': round(avg_g1, 3), 'gcn': round(avg_g2, 3), 'sem': round(avg_g3, 3)},
                'n_candidates': len(signal_scores),
                'top_m':        top_m,
                'top_item':     c_m[0] if c_m else None,
                'top_score':    round(fused_scores.get(c_m[0], 0), 4) if c_m else 0,
                'trained_gating': self.gating.trained,
            }

        return c_m, fused_scores, debug_info

    def fuse_from_data(
        self,
        data: dict,
        signal_scores: Dict[str, Dict[str, float]],
        top_m: int = None,
        debug: bool = True,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        
        len_seq = data.get('len_seq', 0)
        if len_seq == 0:
            raw_seq = data.get('seq', data.get('history', []))
            if isinstance(raw_seq, str):
                len_seq = len([x for x in raw_seq.split(',') if x.strip()])
            elif isinstance(raw_seq, (list, tuple)):
                len_seq = len(raw_seq)

        top_m = top_m or self.cfg.retrieval.top_M
        return self.fuse(signal_scores, len_seq=len_seq, top_m=top_m, debug=debug)