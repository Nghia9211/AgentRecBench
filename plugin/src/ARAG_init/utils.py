from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any, Callable
import threading
import datetime
import csv
import os
import json

def find_top_k_similar_items(
    query: str,
    candidate_list: List[Any],
    embedding_function: Callable,
    k: int = 5
) -> List[Any]:

    query_vec = embedding_function.embed_query(query)
    texts = [json.dumps(normalize_item_data(c)) for c in candidate_list]

    item_vecs = embedding_function.embed_documents(texts)
    sims = cosine_similarity([query_vec], item_vecs)[0]
    
    results = sorted(zip(candidate_list, sims), key=lambda x: x[1], reverse=True)
    return [res[0] for res in results[:k]]

def normalize_item_data(item: Any) -> dict:
    """Chuẩn hóa dữ liệu item từ nhiều nguồn khác nhau."""
    if isinstance(item, str):
        try: item = json.loads(item.replace("'", '"'))
        except: return {}

    return {
        "item_id": str(item.get('item_id', item.get('sub_item_id', 'unknown'))),
        "name": item.get('title') or item.get('name') or item.get('title_without_series'),
        "description": str(item.get('description') or item.get('attributes') or "")[:200],
        "Rating": str(item.get('stars') or item.get('average_rating') or ""),
    }

def get_last_message(blackboard, role):
    return next((msg for msg in reversed(blackboard) if msg.role == role), None)

def get_user_understanding(state):
    msg = get_last_message(state['blackboard'], "UserUnderStanding")
    return msg.content if msg else ""

def get_user_summary(state):
    msg = get_last_message(state['blackboard'], "ContextSummary")
    return msg.content if msg else ""
