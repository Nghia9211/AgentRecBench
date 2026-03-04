from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any, Callable, Dict
import torch
import torch.nn.functional as F
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

def generate_graph_context_string(
    user_history_ids: List[str],
    gcn_embeddings: Dict[str, torch.Tensor],
    candidate_list: List[dict],
    top_k_neighbors: int = 5
) -> str:
    if not gcn_embeddings or not user_history_ids:
        return ""

    user_vecs = []
    for uid in user_history_ids:
        if uid in gcn_embeddings:
            user_vecs.append(gcn_embeddings[uid])
    
    if not user_vecs:
        return ""
    
    user_emb = torch.stack(user_vecs).mean(dim=0).unsqueeze(0)

    candidates_with_gcn = []
    for item in candidate_list:
        # Đảm bảo item là dict
        if not isinstance(item, dict): continue
        
        iid = str(item.get('item_id', ''))
        if iid in gcn_embeddings:
            candidates_with_gcn.append((item, gcn_embeddings[iid]))
    
    if not candidates_with_gcn:
        return ""

    cand_tensor = torch.stack([x[1] for x in candidates_with_gcn])
    scores = F.cosine_similarity(user_emb, cand_tensor)
    top_indices = torch.topk(scores, k=min(top_k_neighbors, len(candidates_with_gcn))).indices.tolist()
    
    suggested_items = [candidates_with_gcn[i][0] for i in top_indices]
    suggested_strings = []

    for item in suggested_items:
        name = item.get('name') or item.get('title') or "Unknown Item"
        cats = item.get('categories', [])
        main_cat = "Item"

        if isinstance(cats, list) and len(cats) > 0:
            main_cat = cats[0] 
        elif isinstance(cats, str) and cats.strip():
            main_cat = cats.split(',')[0] 
        else:
            shelves = item.get('popular_shelves', []) 
            if not isinstance(shelves, list):
                shelves = []

            ignore_tags = {'to-read', 'currently-reading', 'owned', 'favorites', 'default', 'all-time-favorites'}
            main_cat = "Book" 
            
            for s in shelves:
                if isinstance(s, dict):
                    tag_name = s.get('name')
                    if tag_name and tag_name not in ignore_tags:
                        main_cat = tag_name
                        break 
        
        suggested_strings.append(f"'{name}' ({main_cat})")
    
    context_str = (
        f"Graph Analysis: The user's interaction network strongly aligns with these items: {', '.join(suggested_strings)}. "
        "Items sharing similar structural patterns in the graph should be prioritized."
    )

    return context_str


# def perform_rag_retrieval(
#     augmented_query: str,
#     candidate_list: List[dict],
#     embedding_function: Callable,
#     top_k: int = 5) -> List[dict]:
#     """
#     Hàm RAG thuần túy:
#     Dùng câu query (đã được bơm GCN Context) để tìm kiếm Semantic Similarity.
#     """
    
#     query_vec = embedding_function.embed_query(augmented_query)

#     candidate_texts = [
#         f"Name: {item['name'] or item['title']}. Category: {item['category']}. Description: {item['description'][:300]}" 
#         for item in candidate_list
#     ]
#     cand_vecs = embedding_function.embed_documents(candidate_texts)
#     sims = cosine_similarity([query_vec], cand_vecs)[0]
    
#     sorted_indices = sims.argsort()[::-1][:top_k]
    
#     results = []
#     for idx in sorted_indices:
#         results.append(candidate_list[idx])
        
#     return results


def get_last_message(blackboard, role):
    return next((msg for msg in reversed(blackboard) if msg.role == role), None)

def get_user_understanding(state):
    msg = get_last_message(state['blackboard'], "UserUnderStanding")
    return msg.content if msg else ""

def get_user_summary(state):
    msg = get_last_message(state['blackboard'], "ContextSummary")
    return msg.content if msg else ""
