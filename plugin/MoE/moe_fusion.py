"""
moe_fusion/moe_fusion.py
─────────────────────────
MoE Early Fusion:
    g(u, i) = softmax(MLP(x_{u,i}))          ← gating weights
    s0(u, i) = g1*s_seq + g2*s_gcn + g3*s_sem ← fused score
    C_M      = TopM(s0)                        ← rough candidates

Đây là trái tim của MoE pipeline, kết hợp 3 signal sources
thông qua learned gating weights.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from config import MoEConfig, DEFAULT_CONFIG
from gating_network import GatingNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Normalize helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict: return {}
    vals = np.array(list(score_dict.values()))
    mean_v, std_v = np.mean(vals), np.std(vals)
    if std_v == 0:
        return {k: 0.0 for k in score_dict}
    return {k: float((v - mean_v) / std_v) for k, v in score_dict.items()}


def _extract_signal(
    signal_scores: Dict[str, Dict[str, float]],
    signal_key:    str,
) -> Dict[str, float]:
    """Trích một signal cụ thể ra khỏi signal_scores dict."""
    return {name: scores.get(signal_key, 0.0)
            for name, scores in signal_scores.items()}


# ─────────────────────────────────────────────────────────────────────────────
# MoEFusion
# ─────────────────────────────────────────────────────────────────────────────

class MoEFusion:
    """
    Thực hiện MoE Early Fusion:
        1. Normalize 3 signals về [0,1]
        2. Gating network dự đoán [g1, g2, g3] per item
        3. s0(u,i) = g1*s_seq + g2*s_gcn + g3*s_sem
        4. Trả về C_M = TopM(s0)

    Interface:
        fuser = MoEFusion(gating_network, cfg)
        c_m, fused_scores, debug = fuser.fuse(signal_scores, top_m=30)
    """

    def __init__(
        self,
        gating:  GatingNetwork,
        cfg:     MoEConfig = None,
    ):
        self.gating = gating
        self.cfg    = cfg or DEFAULT_CONFIG

    # ─────────────────────────────────────────────────────────────────────
    # Core fusion
    # ─────────────────────────────────────────────────────────────────────

    def fuse(
        self,
        signal_scores: Dict[str, Dict[str, float]],
        top_m:         int  = None,
        debug:         bool = False,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        """
        Fuse 3 signals thành 1 score và chọn C_M.

        Args:
            signal_scores: Dict[item_name → {'seq': float, 'gcn': float, 'sem': float}]
                           (output của CandidateRetriever.retrieve())
            top_m:         số items trong C_M (default: cfg.retrieval.top_M)
            debug:         có trả về thông tin debug không

        Returns:
            c_m:          List[str] — top-M item names
            fused_scores: Dict[str, float] — s0 cho mỗi item
            debug_info:   dict — thông tin debug (gates, signals, ...)
        """
        top_m = top_m or self.cfg.retrieval.top_M

        if not signal_scores:
            return [], {}, {}

        # ── 1. Normalize từng signal riêng biệt ──────────────────────────
        raw_seq = _extract_signal(signal_scores, 'seq')
        raw_gcn = _extract_signal(signal_scores, 'gcn')
        raw_sem = _extract_signal(signal_scores, 'sem')

        norm_seq = _zscore_normalize(raw_seq)
        norm_gcn = _zscore_normalize(raw_gcn)
        norm_sem = _zscore_normalize(raw_sem)

        # Gộp normalized scores trở lại (để gating network dùng)
        norm_signal_scores = {
            name: {
                'seq': norm_seq.get(name, 0.0),
                'gcn': norm_gcn.get(name, 0.0),
                'sem': norm_sem.get(name, 0.0),
            }
            for name in signal_scores
        }

        # ── 2. Gating weights ─────────────────────────────────────────────
        gate_weights = self.gating.predict(norm_signal_scores)
        # gate_weights: Dict[item_name → (g1, g2, g3)]

        # ── 3. Compute s0 = g1*s_seq + g2*s_gcn + g3*s_sem ──────────────
        fused_scores: Dict[str, float] = {}
        for name in signal_scores:
            g1, g2, g3 = gate_weights.get(name, (1/3, 1/3, 1/3))
            s_seq = norm_seq.get(name, 0.0)
            s_gcn = norm_gcn.get(name, 0.0)
            s_sem = norm_sem.get(name, 0.0)

            # Disable signals theo config
            if not self.cfg.use_seq:
                s_seq, g1 = 0.0, 0.0
            if not self.cfg.use_gcn:
                s_gcn, g2 = 0.0, 0.0
            if not self.cfg.use_semantic:
                s_sem, g3 = 0.0, 0.0

            # Re-normalize gates nếu có signal bị disable
            total_g = g1 + g2 + g3
            if total_g > 0:
                g1, g2, g3 = g1/total_g, g2/total_g, g3/total_g

            fused_scores[name] = g1 * s_seq + g2 * s_gcn + g3 * s_sem

        # ── 4. C_M = TopM(s0) ─────────────────────────────────────────────
        c_m = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_m]

        # ── 5. Debug info ─────────────────────────────────────────────────
        debug_info = {}
        if debug:
            # Tính average gate weights
            all_gates = list(gate_weights.values())
            avg_g1 = np.mean([g[0] for g in all_gates])
            avg_g2 = np.mean([g[1] for g in all_gates])
            avg_g3 = np.mean([g[2] for g in all_gates])

            debug_info = {
                'avg_gates':    {'seq': round(avg_g1, 3),
                                 'gcn': round(avg_g2, 3),
                                 'sem': round(avg_g3, 3)},
                'n_candidates': len(signal_scores),
                'top_m':        top_m,
                'top_item':     c_m[0] if c_m else None,
                'top_score':    round(fused_scores.get(c_m[0], 0), 4) if c_m else 0,
                'trained_gating': self.gating.trained,
            }
            print(f"[MoEFusion] avg gates → seq={avg_g1:.3f}, "
                  f"gcn={avg_g2:.3f}, sem={avg_g3:.3f} | "
                  f"C_M={len(c_m)} items")

        return c_m, fused_scores, debug_info

    def fuse_from_data(
        self,
        data:          dict,
        signal_scores: Dict[str, Dict[str, float]],
        top_m:         int  = None,
        debug:         bool = True,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        """
        Convenience wrapper: nhận data dict để tự detect top_m.
        """
        top_m = top_m or self.cfg.retrieval.top_M
        return self.fuse(signal_scores, top_m=top_m, debug=debug)
