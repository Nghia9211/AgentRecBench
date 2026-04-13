"""
moe_fusion/reranker.py
───────────────────────
"""

import json
import re
import traceback
from typing import Dict, List, Tuple
import numpy as np
import math

def rank_to_score(ranked_list: List[str]) -> Dict[str, float]:
    # Score NDCG: 1.0, 0.63, 0.5, 0.43, 0.38...
    return {item: 1.0 / math.log2(rank + 2) for rank, item in enumerate(ranked_list)}

def _build_user_query(data: dict) -> str:
    parts = []
    reviews = data.get('reviews', [])
    if reviews:
        history_review = str(reviews[-15:])
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            encoded = enc.encode(history_review)
            if len(encoded) > 4000: history_review = enc.decode(encoded[:4000])
        except Exception:
            history_review = history_review[:15000]
        parts.append(f"User's Historical Reviews:\n{history_review}")
    else:
        seq_str = data.get('seq_str', '') or ''
        if seq_str and seq_str.strip() and seq_str != 'Empty History':
            words = seq_str.split()
            parts.append(f"User history items: {' '.join(words[-80:])}")
    return "\n\n".join(parts) if parts else ""

def _embed_similarity(query: str, candidate_names: List[str], candidate_texts: Dict[str, str], embedding_fn) -> Dict[str, float]:
    if not candidate_names or embedding_fn is None: return {n: 0.5 for n in candidate_names}
    try:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        query_vec  = embedding_fn.embed_query(query)
        texts      = [candidate_texts.get(n, n) for n in candidate_names]
        item_vecs  = embedding_fn.embed_documents(texts)
        sims       = cos_sim([query_vec], item_vecs)[0]
        min_s, max_s = float(sims.min()), float(sims.max())
        rng = max_s - min_s if max_s != min_s else 1.0
        return {n: float((sims[i] - min_s) / rng) for i, n in enumerate(candidate_names)}
    except Exception:
        return {n: 0.5 for n in candidate_names}

def _llm_rerank(
    llm, data: dict, name2id: Dict[str, int], candidate_names: List[str], 
    user_query: str, memory: List[str], max_candidates: int = 20
) -> Tuple[List[str], str]:
    cans_to_rank = candidate_names[:max_candidates]
    dataset = next((d for d in ['yelp', 'amazon', 'goodreads'] if d in str(data)), 'amazon')
    task_type = {"goodreads": "Goodreads", "yelp": "Yelp", "amazon": "Amazon"}.get(dataset, "Platform")
    task_item = {"goodreads": "book", "yelp": "business", "amazon": "product"}.get(dataset, "item")

    interaction_tool = data.get('interaction_tool')
    id2rawid         = data.get('id2rawid', {})
    item_list_info   = []

    if interaction_tool and name2id and id2rawid:
        keys = ['item_id', 'name', 'stars', 'review_count', 'attributes',
        'title', 'average_rating', 'rating_number', 'description',
        'ratings_count', 'title_without_series']
        for name in cans_to_rank:
            inner_id = name2id.get(name)
            raw_id   = id2rawid.get(inner_id)
            info_dict = {'Target_Name': name}
            if raw_id:
                try:
                    fetched = interaction_tool.get_item(item_id=raw_id)
                    if fetched:
                        for k in keys:
                            if k in fetched: info_dict[k] = fetched[k]
                except Exception: pass
            item_list_info.append(info_dict)
    else:
        item_list_info = [{'Target_Name': n} for n in cans_to_rank]

    items_desc_str = str(item_list_info)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        encoded_items = enc.encode(items_desc_str)
        if len(encoded_items) > 6000: items_desc_str = enc.decode(encoded_items[:6000])
    except Exception:
        items_desc_str = items_desc_str[:25000]

    # ── LLM PROMPT 4.0: SOFT CORRECTION ──
    if len(memory) > 0:
        dialogue_hist = "\n".join(memory[-2:])
        prompt = f"""You are a recommendation system for {task_item}s on {task_type}.
In the previous round, you recommended items but the user rejected them. 

User's Profile & History:
{user_query}

Previous Dialogue & User's Critique:
{dialogue_hist}

You must evaluate the ENTIRE candidate list again: {str(cans_to_rank)}
Candidate Details:
{items_desc_str}

CRITICAL INSTRUCTIONS:
1. Listen to the user's critique. Find items in the list that address their concerns.
2. If you firmly believe an item from the previous list is mathematically still the best fit despite the critique, YOU MAY KEEP IT in your new ranking.
3. Output ONLY a JSON object with keys "ranked_items" (list of Target_Names) and "explanation" (explain your adjustment).
4. Do not add any text outside the JSON. Example format: {{"ranked_items": ["A", "B", "C"], "explanation": "..."}}"""

    else:
        prompt = f"""You are a real human user on {task_type}, a platform for crowd-sourced {task_item} reviews.
Here is your profile and review history:
{user_query}

Now rank the following {len(cans_to_rank)} candidate {task_item}s: {str(cans_to_rank)} according to their match degree to your preference.
Candidate Details:
{items_desc_str}

Instructions:
- Output ONLY a JSON object with keys "ranked_items" (list of Target_Names) and "explanation" (brief reason).
- Include ALL candidates. Rank the most interested {task_item}s closer to the front.
- Do not add any text outside the JSON. Example format: {{"ranked_items": ["A", "B", "C"], "explanation": "..."}}"""

    try:
        response      = llm.invoke(prompt)
        content       = response.content if hasattr(response, 'content') else str(response)
        content_clean = content.replace("`" * 3 + "json", "").replace("`" * 3, "").strip()
        parsed        = json.loads(content_clean)
        ranked        = parsed.get('ranked_items', [])
        explanation   = parsed.get('explanation', '')

        valid_set    = set(cans_to_rank)
        ranked_valid = [r for r in ranked if r in valid_set]
        ranked_set   = set(ranked_valid)
        for n in cans_to_rank:
            if n not in ranked_set: ranked_valid.append(n)
        if len(candidate_names) > max_candidates: ranked_valid += candidate_names[max_candidates:]

        return ranked_valid, explanation
    except Exception as e:
        return candidate_names, "Failed to parse LLM output."

class Reranker:
    def __init__(self, embedding_fn=None, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=15):
        self.embedding_fn, self.llm, self.mode, self.enabled, self.top_llm = embedding_fn, llm, mode, enabled, top_llm

    def rerank(self, data: dict, c_m: List[str], id2name: Dict[int, str]=None, name2id: Dict[str, int]=None, memory: List[str]=None) -> Tuple[List[str], Dict[str, float], str]:
        if not self.enabled or not c_m: return c_m, rank_to_score(c_m), "Reranker disabled."
        memory = memory or []
        try:
            if self.mode == 'embed_only': return self._embed_rerank(data, c_m, memory)
            elif self.mode == 'llm': return self._llm_only_rerank(data, name2id, c_m, memory)
            elif self.mode == 'hybrid': return self._hybrid_rerank(data, name2id, c_m, memory)
            else: return self._embed_rerank(data, c_m, memory)
        except Exception as e: return c_m, rank_to_score(c_m), f"Reranker error: {e}"

    def _embed_rerank(self, data: dict, c_m: List[str], memory: List[str]) -> Tuple[List[str], Dict[str, float], str]:
        query = _build_user_query(data)
        if not query: return c_m, rank_to_score(c_m), "No user query available."
        sim_scores = _embed_similarity(query, c_m, {n: n for n in c_m}, self.embedding_fn)
        ranked = sorted(c_m, key=lambda n: sim_scores.get(n, 0.0), reverse=True)
        return ranked, rank_to_score(ranked), "Reranked by embedding similarity."

    def _llm_only_rerank(self, data: dict, name2id: Dict[str, int], c_m: List[str], memory: List[str]) -> Tuple[List[str], Dict[str, float], str]:
        if self.llm is None: return self._embed_rerank(data, c_m, memory)
        query = _build_user_query(data)
        ranked, explanation = _llm_rerank(self.llm, data, name2id, c_m, query, memory, max_candidates=len(c_m))
        return ranked, rank_to_score(ranked), explanation

    def _hybrid_rerank(self, data: dict, name2id: Dict[str, int], c_m: List[str], memory: List[str]) -> Tuple[List[str], Dict[str, float], str]:
        query = _build_user_query(data)
        sim_scores = _embed_similarity(query, c_m, {n: n for n in c_m}, self.embedding_fn)
        embed_ranked = sorted(c_m, key=lambda n: sim_scores.get(n, 0.0), reverse=True)
        if self.llm is None: return embed_ranked, rank_to_score(embed_ranked), "Fallback to embed_only"
        
        top_pool, tail_pool = embed_ranked[:self.top_llm], embed_ranked[self.top_llm:]
        llm_ranked, explanation = _llm_rerank(self.llm, data, name2id, top_pool, query, memory, max_candidates=self.top_llm)
        final_ranked = llm_ranked + tail_pool
        return final_ranked, rank_to_score(final_ranked), explanation

    @classmethod
    def from_shared(cls, shared: dict, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=15) -> "Reranker":
        return cls(embedding_fn=shared.get('embedding_function'), llm=llm, mode=mode, enabled=enabled, top_llm=top_llm)