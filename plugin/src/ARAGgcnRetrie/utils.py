from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any, Callable
import torch
import json


# ─────────────────────────────────────────────────────────────────────────────
#  CORE IDEA: Don't decide what fields matter — preserve everything meaningful
#  as a flat key:value string and let the LLM + embedding model do the rest.
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_FIELDS = {
    "images", "verified_purchase", "review_id", "helpful_vote",
    "timestamp", "timestamp_norm", "source", "type",
}


_ID_ALIASES   = ("item_id", "sub_item_id", "business_id")

_NAME_ALIASES = ("name", "title", "title_without_series")


def _to_str(value: Any, max_len: int = 300) -> str:
    """Safely flatten any value to a short string."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(v) for v in value if not isinstance(v, dict) and str(v).strip()]
        value = " | ".join(parts)
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items() if v not in (None, False, "False", "")]
        value = ", ".join(parts)
    return str(value).strip()[:max_len]


def normalize_item(item: Any) -> dict:
    """
    Domain-agnostic normalization.

    Rules:
    - Always produce 'item_id' and 'name' (resolved from known aliases).
    - Walk every remaining field: stringify it, skip empty/blacklisted fields.
    - Nested dicts/lists are flattened inline — no branching on field semantics.

    Output is a flat dict of short strings, ready for both embedding and prompting.
    """
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except json.JSONDecodeError:
            return {"item_id": "unknown", "name": item}

    if not isinstance(item, dict):
        return {"item_id": "unknown", "name": str(item)}

    out = {}

    # ── Resolve canonical item_id ─────────────────────────────────────────────
    for alias in _ID_ALIASES:
        if item.get(alias) not in (None, ""):
            out["item_id"] = str(item[alias])
            break
    out.setdefault("item_id", "unknown")

    # ── Resolve canonical name ────────────────────────────────────────────────
    for alias in _NAME_ALIASES:
        if item.get(alias) not in (None, ""):
            out["name"] = _to_str(item[alias])
            break
    out.setdefault("name", "Unknown")

    # ── Walk all remaining fields ─────────────────────────────────────────────
    skip = _SKIP_FIELDS | set(_ID_ALIASES) | set(_NAME_ALIASES)
    for key, val in item.items():
        if key in skip:
            continue
        s = _to_str(val)
        if s:                       # drop empty/zero-length values
            out[key] = s

    return out


def item_to_text(item: Any) -> str:
    """
    Convert a raw item (any domain) to a single natural-language string
    suitable for dense embedding.

    Format:  "<name>. <key>: <value>. <key>: <value>. ..."
    The model sees field names, so it can reason about "attributes",
    "description", "stars", "review_count" without us hard-coding anything.
    """
    norm = normalize_item(item)
    name = norm.pop("name", "Unknown")
    norm.pop("item_id", None)           # id carries no semantic content

    parts = [name] + [f"{k}: {v}" for k, v in norm.items() if v]
    return ". ".join(parts)


def find_top_k_similar_items(
    query: str,
    candidate_list: List[Any],
    embedding_function: Callable,
    k: int =5,
) -> List[Any]:
    """Pure semantic retrieval — no domain param needed."""
    query_vec  = embedding_function.embed_query(query)
    item_texts = [item_to_text(c) for c in candidate_list]
    item_vecs  = embedding_function.embed_documents(item_texts)
    sims       = cosine_similarity([query_vec], item_vecs)[0]
    ranked     = sorted(zip(candidate_list, sims), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:k]]


def get_gcn_latent_interests(
    user_history_ids: list,
    gcn_embeddings,
    candidate_list: list,
) -> str:
    """
    Domain-agnostic: uses the item's 'name' field (resolved via normalize_item)
    as the label — no category extraction logic needed.
    """
    if not gcn_embeddings or not user_history_ids:
        return "No specific graph patterns detected."

    user_vecs = [gcn_embeddings[uid] for uid in user_history_ids if uid in gcn_embeddings]
    if not user_vecs:
        return "No graph data available for this user."

    user_emb = torch.stack(user_vecs).mean(dim=0)

    # Only keep candidates that have a GCN embedding
    indexed = [
        (normalize_item(item), item)
        for item in candidate_list
        if str(normalize_item(item).get("item_id", "")) in gcn_embeddings
    ]
    if not indexed:
        return "New user profile with limited graph connections."

    cand_ids    = [norm["item_id"] for norm, _ in indexed]
    cand_tensor = torch.stack([gcn_embeddings[iid] for iid in cand_ids])
    sims        = torch.nn.functional.cosine_similarity(user_emb.unsqueeze(0), cand_tensor)
    top_k       = torch.topk(sims, k=min(5, len(cand_ids)))

    labels = []
    for idx in top_k.indices:
        norm = indexed[idx][0]
        labels.append(norm.get("name", "Unknown"))

    return f"Behavioral graph suggests affinity toward: {', '.join(labels)}."


# ─────────────────────────────────────────────────────────────────────────────
#  BLACKBOARD HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_last_message(blackboard, role):
    return next((msg for msg in reversed(blackboard) if msg.role == role), None)

def get_user_understanding(state):
    msg = get_last_message(state['blackboard'], "UserUnderStanding")
    return msg.content if msg else ""

def get_user_summary(state):
    msg = get_last_message(state['blackboard'], "ContextSummary")
    return msg.content if msg else ""

def truncate_string_tokens(text: str, max_chars: int = 30000):
    """Giới hạn độ dài chuỗi để tránh tràn token (ước tính 1 token ~ 4 ký tự)"""
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]