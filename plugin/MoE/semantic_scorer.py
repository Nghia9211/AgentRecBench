"""
moe_fusion/semantic_scorer.py
──────────────────────────────
Semantic scorer với rich content support áp dụng cho TẤT CẢ dataset.

Cập nhật:
  - Bỏ phân loại dataset trong _build_query.
  - Amazon, Yelp, Goodreads đều ưu tiên sử dụng "Review text" của user 
    để làm query (nếu có) trước khi fallback về seq_str (item names).
  - Sử dụng chung logic rich query để tăng cường Semantic Feature.
"""

import numpy as np
from typing import Dict, List, Optional


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity giữa vector a (dim,) và matrix b (n, dim)."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm


class SemanticScorer:
    """
    Tính semantic score bằng cách embed trực tiếp query và candidates.

    Flow chung cho tất cả Dataset:
      - Query: review text của user (ưu tiên 1) -> seq_str (fallback).
      - Documents: rich content từ FAISS docstore nếu có,
                   fallback về embed item_name.
    """

    def __init__(
        self,
        vector_store,
        embedding_function,
        id2name:         Dict[int, str],
        name2id:         Dict[str, int],
        top_fetch:       int = 50,
        max_query_words: int = 100,
        dataset:         str = 'amazon',   # Giữ param cho tương thích interface cũ
    ):
        self.vector_store       = vector_store
        self.embedding_function = embedding_function
        self.id2name            = id2name
        self.name2id            = name2id
        self.max_query_words    = max_query_words
        self.dataset            = dataset

        # Build docstore lookup từ FAISS nếu có (để lấy rich content)
        self._docstore_cache: Dict[str, str] = {}
        if vector_store is not None:
            self._build_docstore_cache()

        if embedding_function is None:
            print("[SemanticScorer] WARNING: embedding_function is None. "
                  "Tất cả scores sẽ là 0.0.")

    def _build_docstore_cache(self):
        """
        Build Dict[item_name_lower → rich_page_content] từ FAISS docstore.
        Dùng để lấy rich text khi scoring thay vì chỉ embed item_name.
        """
        try:
            docstore = self.vector_store.docstore
            docs = getattr(docstore, '_dict', {})
            for doc_id, doc in docs.items():
                name = (doc.metadata.get('item_name') or '').strip().lower()
                if name and doc.page_content:
                    self._docstore_cache[name] = doc.page_content
            print(f"[SemanticScorer] Docstore cache built: "
                  f"{len(self._docstore_cache)} entries")
        except Exception as e:
            print(f"[SemanticScorer] Could not build docstore cache: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Query building (Áp dụng chung cho mọi Dataset)
    # ─────────────────────────────────────────────────────────────────────

    def _build_query(self, seq_str: str, data: dict = None) -> str:
        """
        Tạo query text đại diện cho user preference.

        Priority:
          1. data['reviews']: full review text (nếu đã được inject từ InteractionTool)
          2. data['review_texts']: list of review strings
          3. seq_str: item names đã tương tác (fallback truyền thống)
        """
        parts = []

        if data:
            # ── Option 1: structured reviews ──────────────────────────────
            reviews = data.get('reviews', [])
            if reviews and isinstance(reviews, list):
                snippets = []
                for r in reviews[-8:]:   # 8 reviews gần nhất
                    text = ''
                    if isinstance(r, dict):
                        # Bắt các key phổ biến của nhiều bộ dataset
                        text = (r.get('review_text')
                                or r.get('text')
                                or r.get('body')
                                or '')
                    elif isinstance(r, str):
                        text = r
                    if text:
                        words = text.strip().split()
                        snippets.append(' '.join(words[:40]))  # 40 từ/review
                if snippets:
                    parts.append("User review history: " + " | ".join(snippets))

            # ── Option 2: pre-computed review texts ───────────────────────
            if not parts:
                review_texts = data.get('review_texts', [])
                if review_texts:
                    joined = ' | '.join(
                        str(t)[:200] for t in review_texts[-5:]
                    )
                    parts.append("User review history: " + joined)

        # ── Option 3: fallback về item names ──────────────────────────────
        if not parts:
            if seq_str and seq_str.strip() and seq_str != 'Empty History':
                words = seq_str.split()
                if len(words) > self.max_query_words:
                    seq_str = ' '.join(words[-self.max_query_words:])
                parts.append(f"User interests: {seq_str}")

        # ── Option 4: cold-start ──────────────────────────────────────────
        if not parts and data:
            cans_str = data.get('cans_str', '')
            if cans_str:
                parts.append(f"Recommend items similar to: {cans_str[:200]}")

        return " ".join(parts) if parts else ""

    # ─────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────

    def _get_candidate_texts(self, candidate_names: List[str]) -> List[str]:
        """
        Lấy document text cho từng candidate.
        Thử lấy rich_page_content từ docstore cache, nếu không có thì trả về item_name.
        """
        texts = []
        for name in candidate_names:
            rich = self._docstore_cache.get(name.lower().strip())
            texts.append(rich if rich else name)
        return texts

    def _embed_and_score(
        self,
        query:           str,
        candidate_names: List[str],
    ) -> Dict[str, float]:
        """
        Embed query + candidates trực tiếp, tính cosine similarity.
        """
        if not candidate_names or self.embedding_function is None:
            return {n: 0.0 for n in candidate_names}

        candidate_texts = self._get_candidate_texts(candidate_names)

        try:
            query_vec = np.array(
                self.embedding_function.embed_query(query),
                dtype=np.float32,
            )
            cand_vecs = np.array(
                self.embedding_function.embed_documents(candidate_texts),
                dtype=np.float32,
            )
        except Exception as e:
            print(f"[SemanticScorer] Embedding failed: {e}")
            return {n: 0.0 for n in candidate_names}

        sims = _cosine_sim(query_vec, cand_vecs)

        # Normalize về [0, 1]
        min_s, max_s = float(sims.min()), float(sims.max())
        if max_s > min_s:
            norm_sims = (sims - min_s) / (max_s - min_s)
        else:
            norm_sims = np.full_like(sims, 0.5)

        return {
            name: float(np.clip(norm_sims[i], 0.0, 1.0))
            for i, name in enumerate(candidate_names)
        }

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def score(
        self,
        seq_str:       str,
        candidate_ids: List[int],
        data:          dict = None,
    ) -> Dict[str, float]:
        if not candidate_ids:
            return {}

        candidate_names = [
            self.id2name.get(cid, f"item_{cid}")
            for cid in candidate_ids
        ]

        if self.embedding_function is None:
            return {name: 0.0 for name in candidate_names}

        query = self._build_query(seq_str, data)

        if not query:
            return {name: 0.5 for name in candidate_names}

        return self._embed_and_score(query, candidate_names)

    def top_k_names(
        self,
        seq_str:       str,
        candidate_ids: List[int],
        k:             int = 20,
        data:          dict = None,
    ) -> List[str]:
        scores = self.score(seq_str, candidate_ids, data)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    # ─────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_shared(
        cls,
        shared:    dict,
        top_fetch: int = 50,
        dataset:   str = 'amazon',
    ) -> "SemanticScorer":
        ef = shared.get('embedding_function')
        if ef is None:
            print("[SemanticScorer] WARNING: embedding_function is None.")
        return cls(
            vector_store       = shared.get('vector_store'),
            embedding_function = ef,
            id2name            = shared['id2name'],
            name2id            = shared['name2id'],
            dataset            = dataset,
        )