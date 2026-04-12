"""
moe_fusion/score_combiner.py
─────────────────────────────
Score Combiner (Final Pre-feedback Score):
    s1(u, i) = alpha * s0(u, i) + (1 - alpha) * s_rerank(u, i)
    C_K      = TopK(s1)

Fix (Feedback Loop):
  - alpha giờ nhận thêm `epoch` để tăng dần theo round:
      Round 1: khám phá → alpha thấp hơn (trust reranker)
      Round 2+: khai thác → alpha cao hơn (trust MoE base)
  - Cold-start penalty vẫn giữ nguyên.
  - Confidence bonus vẫn giữ nguyên.
"""

import numpy as np
from typing import Dict, List, Tuple

from config import ScoreConfig, DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict: return {}
    vals = np.array(list(score_dict.values()))
    mean_v, std_v = np.mean(vals), np.std(vals)
    if std_v == 0:
        return {k: 0.0 for k in score_dict}
    return {k: float((v - mean_v) / std_v) for k, v in score_dict.items()}


def _sasrec_confidence(fused_scores: Dict[str, float]) -> float:
    """
    Tính confidence của MoE base score (s0):
    margin = (top1 - top2) / (top1 - min)
    Confidence cao → alpha tăng (tin MoE base hơn).
    """
    if len(fused_scores) < 2:
        return 1.0
    sorted_vals = sorted(fused_scores.values(), reverse=True)
    top1, top2  = sorted_vals[0], sorted_vals[1]
    score_range = sorted_vals[0] - sorted_vals[-1]
    if score_range == 0:
        return 0.5
    margin = (top1 - top2) / score_range
    return float(np.clip(margin, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# ScoreCombiner
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCombiner:
    """
    Kết hợp s0 (MoE fusion) và s_rerank (Reranker) thành s1,
    rồi chọn C_K = TopK(s1) làm final recommendation list.

    s1(u, i) = alpha * s0(u, i) + (1 - alpha) * s_rerank(u, i)
    C_K      = TopK(s1)

    Feedback loop fix:
      - get_alpha() nhận thêm `epoch` (round hiện tại trong dialogue).
      - Mỗi round tăng alpha thêm `epoch_alpha_boost` (mặc định 0.05).
      - Ý nghĩa: round đầu khám phá (reranker dẫn dắt), round sau
        khai thác (MoE base đã được validate → tin hơn).
    """

    # Boost alpha mỗi round thêm dialogue (feedback loop)
    EPOCH_ALPHA_BOOST: float = 0.00

    def __init__(self, cfg: ScoreConfig = None):
        self.cfg = cfg or DEFAULT_CONFIG.scoring

    # ─────────────────────────────────────────────────────────────────────
    # Alpha selection
    # ─────────────────────────────────────────────────────────────────────

    def get_alpha(
        self,
        dataset:       str,
        len_seq:       int,
        fused_scores:  Dict[str, float] = None,
        epoch:         int = 1,
    ) -> float:
        """
        Chọn alpha (weight của s0) theo:
          - Dataset (dense/sparse)
          - Cold-start (len_seq ngắn)
          - Confidence của s0 (nếu có fused_scores)
          - Epoch (round trong dialogue — FIX feedback loop)

        alpha cao → tin MoE base hơn
        alpha thấp → tin reranker hơn

        Epoch logic:
          Round 1: chưa có user feedback → dùng base alpha (khám phá)
          Round 2+: đã biết user thích gì → tăng trust MoE base (khai thác)
        """
        if not self.cfg.use_adaptive_alpha:
            # Non-adaptive: vẫn áp dụng epoch boost nhẹ
            base = self.cfg.alpha
            epoch_boost = (epoch - 1) * self.EPOCH_ALPHA_BOOST
            return float(np.clip(base + epoch_boost, 0.1, 0.9))

        # ── Base alpha theo dataset ───────────────────────────────────────
        alpha = self.cfg.dataset_alpha.get(dataset, self.cfg.alpha)

        # ── Confidence bonus (nếu có fused_scores) ───────────────────────
        if fused_scores:
            confidence = _sasrec_confidence(fused_scores)
            conf_bonus = (confidence - 0.5) * 0.1   # [-0.05, +0.05]
            alpha     += conf_bonus

        # ── Epoch boost (feedback loop fix) ──────────────────────────────
        # Round 1 → +0.00, Round 2 → +0.05, Round 3 → +0.10
        epoch_boost = (epoch - 1) * self.EPOCH_ALPHA_BOOST
        alpha      += epoch_boost

        final_alpha = float(np.clip(alpha, 0.1, 0.9))
        if epoch > 1:
            print(f"[ScoreCombiner] epoch={epoch} alpha_boost=+{epoch_boost:.2f} "
                  f"final_alpha={final_alpha:.3f}")
        return final_alpha

    # ─────────────────────────────────────────────────────────────────────
    # Core combine
    # ─────────────────────────────────────────────────────────────────────

    def combine(
        self,
        fused_scores:   Dict[str, float],
        rerank_scores:  Dict[str, float],
        dataset:        str  = 'amazon',
        len_seq:        int  = 0,
        top_k:          int  = None,
        epoch:          int  = 1,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        """
        Kết hợp s0 và s_rerank → s1, chọn C_K = TopK(s1).

        Args:
            fused_scores:   Dict[item_name → s0]  (từ MoEFusion)
            rerank_scores:  Dict[item_name → s_rerank]  (từ Reranker)
            dataset:        'yelp' | 'amazon' | 'goodreads'
            len_seq:        độ dài user history
            top_k:          số items trả về (default: cfg.retrieval.top_K)
            epoch:          round hiện tại trong dialogue (1-indexed)

        Returns:
            c_k:         List[str] — C_K final recommendation list
            s1_scores:   Dict[str, float] — s1 cho mỗi item
            debug_info:  dict
        """
        top_k = top_k or DEFAULT_CONFIG.retrieval.top_K

        if not fused_scores:
            # Fallback: chỉ dùng rerank
            c_k = sorted(rerank_scores, key=rerank_scores.get, reverse=True)[:top_k]
            return c_k, rerank_scores, {'alpha': 0.0, 'fallback': True}

        # ── Normalize về cùng scale ───────────────────────────────────────
        norm_s0     = _zscore_normalize(fused_scores)
        norm_rerank = _zscore_normalize(rerank_scores) if rerank_scores else {}

        # ── Alpha (epoch-aware) ───────────────────────────────────────────
        alpha = self.get_alpha(dataset, len_seq, fused_scores, epoch=epoch)
        beta  = 1.0 - alpha

        # ── s1 = alpha*s0 + beta*s_rerank ────────────────────────────────
        all_items = set(norm_s0.keys()) | set(norm_rerank.keys())
        s1_scores: Dict[str, float] = {}

        for item in all_items:
            s0_val    = norm_s0.get(item, 0.0)
            s_rer_val = norm_rerank.get(item, 0.0)
            s1_scores[item] = alpha * s0_val + beta * s_rer_val

        # ── C_K = TopK(s1) ────────────────────────────────────────────────
        c_k = sorted(s1_scores, key=s1_scores.get, reverse=True)[:top_k]

        debug_info = {
            'alpha':   round(alpha, 3),
            'beta':    round(beta, 3),
            'dataset': dataset,
            'len_seq': len_seq,
            'epoch':   epoch,
            'top_k':   top_k,
            'n_items': len(all_items),
            'top_item': c_k[0] if c_k else None,
        }
        print(f"[ScoreCombiner] epoch={epoch} alpha={alpha:.3f} β={beta:.3f} | "
              f"C_K={len(c_k)} items | top={c_k[:3]}")

        return c_k, s1_scores, debug_info

    def combine_from_pipeline(
        self,
        fused_scores:  Dict[str, float],
        rerank_scores: Dict[str, float],
        data:          dict,
        args,
        top_k:         int = None,
        epoch:         int = 1,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        """
        Convenience wrapper: auto-detect dataset và len_seq từ data dict.

        Args:
            epoch: round hiện tại trong dialogue loop (mới thêm cho FL fix)
        """
        dataset = next(
            (d for d in ['yelp', 'amazon', 'goodreads']
             if d in getattr(args, 'data_dir', '')),
            'amazon'
        )
        len_seq = data.get('len_seq', 0)
        top_k   = top_k or DEFAULT_CONFIG.retrieval.top_K

        return self.combine(
            fused_scores  = fused_scores,
            rerank_scores = rerank_scores,
            dataset       = dataset,
            len_seq       = len_seq,
            top_k         = top_k,
            epoch         = epoch,
        )