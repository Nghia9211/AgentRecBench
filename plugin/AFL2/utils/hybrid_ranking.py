"""
Hybrid Ranking: kết hợp SASRec score và ARAG rank
để tận dụng điểm mạnh của cả hai:
  - SASRec: chính xác khi item space nhỏ, data dense (Yelp)
  - ARAG:   mạnh khi item space lớn, data sparse (Amazon)

Tích hợp vào dialogue_manager.py thay cho rec_agent.act() trực tiếp.
"""

import numpy as np
from typing import List, Dict, Tuple

def compute_sasrec_confidence(score_dict: Dict[str, float]) -> float:
    """
    Tính confidence của SASRec dựa trên margin giữa top-1 và top-2.
    Confidence cao → SASRec chắc chắn → tăng alpha (weight SASRec).
    Confidence thấp → SASRec không chắc → tăng beta (weight ARAG).
    """
    if len(score_dict) < 2:
        return 1.0
    sorted_scores = sorted(score_dict.values(), reverse=True)
    top1, top2    = sorted_scores[0], sorted_scores[1]
    # Normalize margin về 0-1
    score_range = sorted_scores[0] - sorted_scores[-1]
    if score_range == 0:
        return 0.5
    margin = (top1 - top2) / score_range
    return float(np.clip(margin, 0, 1))


def get_adaptive_alpha(confidence: float,
                       dataset: str,
                       len_seq: int) -> float:
    """
    Chọn alpha (weight SASRec) dựa trên:
      - Confidence của SASRec
      - Dataset (dense/sparse)
      - Độ dài sequence (cold-start hay không)

    alpha cao → ưu tiên SASRec
    alpha thấp → ưu tiên ARAG
    """
    # Base alpha theo dataset density
    base_alpha = {
        'yelp':      0.7,   # dense, item space nhỏ → SASRec mạnh
        'amazon':    0.4,   # trung bình → cân bằng
        'goodreads': 0.3,   # sparse, item space lớn → ARAG mạnh
    }.get(dataset, 0.5)

    # Điều chỉnh theo confidence
    # confidence cao → tăng alpha thêm
    confidence_bonus = (confidence - 0.5) * 0.2  # [-0.1, +0.1]

    # Điều chỉnh theo cold-start
    # len_seq ngắn → SASRec kém hơn → giảm alpha
    if len_seq < 3:
        cold_penalty = -0.2
    elif len_seq < 5:
        cold_penalty = -0.1
    else:
        cold_penalty = 0.0

    alpha = base_alpha + confidence_bonus + cold_penalty
    return float(np.clip(alpha, 0.1, 0.9))


# ─────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────
def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize SASRec scores về [0, 1]."""
    if not score_dict:
        return {}
    vals   = list(score_dict.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {k: 1.0 for k in score_dict}
    return {k: (v - min_v) / (max_v - min_v)
            for k, v in score_dict.items()}


def rank_to_score(ranked_list: List[str]) -> Dict[str, float]:
    """
    Chuyển ARAG ranked list thành score theo reciprocal rank:
    rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, ...
    Reciprocal rank tốt hơn linear vì nhấn mạnh top positions.
    """
    return {item: 1.0 / (rank + 1)
            for rank, item in enumerate(ranked_list)}


# ─────────────────────────────────────────────────────────────
# Core hybrid ranking
# ─────────────────────────────────────────────────────────────
def hybrid_rank(sasrec_scores: Dict[str, float],
                arag_ranked:   List[str],
                data:          dict,
                dataset:       str) -> Tuple[List[str], dict]:
    """
    Kết hợp SASRec score và ARAG rank thành final ranked list.

    Args:
        sasrec_scores : {item_name: raw_score} từ UserModelAgent.score()
        arag_ranked   : [item1, item2, ...] từ ARAGRecAgent.act()
        data          : data dict chứa len_seq
        dataset       : 'yelp' | 'amazon' | 'goodreads'

    Returns:
        final_ranked  : top-5 items sau hybrid
        debug_info    : dict chứa alpha, confidence để log
    """
    if not sasrec_scores or not arag_ranked:
        return arag_ranked[:5], {}

    len_seq    = data.get('len_seq', 0)
    confidence = compute_sasrec_confidence(sasrec_scores)
    alpha      = get_adaptive_alpha(confidence, dataset, len_seq)
    beta       = 1.0 - alpha

    # Normalize về cùng scale
    sasrec_norm = normalize_scores(sasrec_scores)
    arag_norm   = rank_to_score(arag_ranked)

    # Weighted combination
    all_items = set(sasrec_norm.keys()) | set(arag_norm.keys())
    final_scores = {}
    for item in all_items:
        s_score = sasrec_norm.get(item, 0.0)
        a_score = arag_norm.get(item, 0.0)
        final_scores[item] = alpha * s_score + beta * a_score

    # Sort và lấy top-5
    final_ranked = sorted(final_scores,
                          key=final_scores.get,
                          reverse=True)[:5]

    debug_info = {
        'alpha':      round(alpha, 3),
        'beta':       round(beta, 3),
        'confidence': round(confidence, 3),
        'len_seq':    len_seq,
        'dataset':    dataset,
    }

    return final_ranked, debug_info


# ─────────────────────────────────────────────────────────────
# Tích hợp vào dialogue_manager.py
# ─────────────────────────────────────────────────────────────
"""
Thêm vào dialogue_manager.py như sau:

from hybrid_ranking import hybrid_rank

# Trong vòng lặp recommend(), sau khi có ARAG output:

if getattr(args, 'use_arag', False) and getattr(args, 'use_hybrid', False):
    # Lấy SASRec scores cho candidates
    sasrec_scores = user_agent.score(
        data['seq'], data['len_seq'], data['cans'])

    # Detect dataset
    dataset = 'yelp' if 'yelp' in args.data_dir else \
              'amazon' if 'amazon' in args.data_dir else 'goodreads'

    # Hybrid ranking
    current_rec_list, debug = hybrid_rank(
        sasrec_scores=sasrec_scores,
        arag_ranked=current_rec_list,
        data=data,
        dataset=dataset,
    )
    print(f"[Hybrid] alpha={debug['alpha']} beta={debug['beta']} "
          f"confidence={debug['confidence']}")
"""


# ─────────────────────────────────────────────────────────────
# Tune alpha — chạy offline để tìm alpha tốt nhất
# ─────────────────────────────────────────────────────────────
def tune_alpha(results_log: list, alphas=None) -> dict:
    """
    Tìm alpha tối ưu từ log kết quả đã chạy.

    results_log: list of dict với keys:
        'sasrec_scores', 'arag_ranked', 'correct_answer', 'len_seq'

    Dùng sau khi có kết quả thực nghiệm để calibrate alpha.
    """
    if alphas is None:
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    best_alpha, best_hit5 = 0.5, 0.0
    results = {}

    for alpha in alphas:
        hits = 0
        for record in results_log:
            sasrec_norm = normalize_scores(record['sasrec_scores'])
            arag_norm   = rank_to_score(record['arag_ranked'])
            all_items   = set(sasrec_norm) | set(arag_norm)

            scores = {
                item: alpha * sasrec_norm.get(item, 0)
                      + (1 - alpha) * arag_norm.get(item, 0)
                for item in all_items
            }
            top5 = sorted(scores, key=scores.get, reverse=True)[:5]
            gt   = record['correct_answer'].lower().strip()
            if gt in [x.lower().strip() for x in top5]:
                hits += 1

        hit5 = hits / len(results_log)
        results[alpha] = hit5
        if hit5 > best_hit5:
            best_hit5, best_alpha = hit5, alpha

    print("\nAlpha tuning results:")
    for a, h in sorted(results.items()):
        bar = "█" * int(h * 40)
        print(f"  alpha={a:.1f}: Hit@5={h:.4f}  {bar}")
    print(f"\nBest alpha = {best_alpha} → Hit@5 = {best_hit5:.4f}")

    return {'best_alpha': best_alpha, 'best_hit5': best_hit5,
            'all_results': results}