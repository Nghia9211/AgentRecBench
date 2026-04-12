import numpy as np
from typing import Dict, List, Optional

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm

class SemanticScorer:
    def __init__(
        self,
        vector_store,
        embedding_function,
        id2name: Dict[int, str],
        name2id: Dict[str, int]
    ):
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.id2name = id2name
        self.name2id = name2id
        
        # Nạp Docstore Cache để hỗ trợ Rich Content
        self._docstore_cache: Dict[str, str] = {}
        if vector_store is not None:
            self._build_docstore_cache()

    def _build_docstore_cache(self):
        try:
            docstore = self.vector_store.docstore
            docs = getattr(docstore, '_dict', {})
            for doc_id, doc in docs.items():
                name = (doc.metadata.get('item_name') or '').strip().lower()
                if name and doc.page_content:
                    self._docstore_cache[name] = doc.page_content
        except Exception as e:
            print(f"[SemanticScorer] Docstore cache init error: {e}")

    def _get_candidate_texts(self, candidate_names: List[str]) -> List[str]:
        texts = []
        for name in candidate_names:
            rich = self._docstore_cache.get(name.lower().strip())
            texts.append(rich if rich else name)
        return texts

    def _embed_and_score(self, query: str, candidate_names: List[str]) -> Dict[str, float]:
        if not candidate_names or self.embedding_function is None:
            return {n: 0.0 for n in candidate_names}

        candidate_texts = self._get_candidate_texts(candidate_names)
        try:
            query_vec = np.array(self.embedding_function.embed_query(query), dtype=np.float32)
            cand_vecs = np.array(self.embedding_function.embed_documents(candidate_texts), dtype=np.float32)
            sims = _cosine_sim(query_vec, cand_vecs)
            
            # Min-Max Scaling Cục bộ
            min_s, max_s = float(sims.min()), float(sims.max())
            norm_sims = (sims - min_s) / (max_s - min_s) if max_s > min_s else np.full_like(sims, 0.5)
            
            return {name: float(np.clip(norm_sims[i], 0.0, 1.0)) for i, name in enumerate(candidate_names)}
        except Exception as e:
            return {n: 0.0 for n in candidate_names}

    def score(self, seq_str: str, candidate_ids: List[int], data: dict = None) -> Dict[str, float]:
        if not candidate_ids: return {}
        candidate_names = [self.id2name.get(cid, f"item_{cid}") for cid in candidate_ids]
        # V4.0: Chỉ dùng query thô, không gọi LLM
        query = f"User interested in: {seq_str}"
        return self._embed_and_score(query, candidate_names)

    @classmethod
    def from_shared(cls, shared: dict) -> "SemanticScorer":
        return cls(
            vector_store = shared.get('vector_store'),
            embedding_function = shared.get('embedding_function'),
            id2name = shared['id2name'],
            name2id = shared['name2id']
        )