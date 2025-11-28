from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any, Callable
import torch

def find_top_k_similar_items(
    query: str,
    candidate_list: List[Any],
    embedding_function: Callable,
    top_k: int = 2
) -> List[Any]:

    sims = []
    
    query_vec = embedding_function.embed_query(query)
    
    item_texts = [str(item) for item in candidate_list]
    item_vectors = embedding_function.embed_documents(item_texts) 
    
    for item, item_vec in zip(candidate_list, item_vectors):
        sim = cosine_similarity([item_vec], [query_vec])[0][0]
        sims.append((item, sim))
    
    sims.sort(key = lambda x: x[1], reverse = True)
    top_k_list = [item for item, sim in sims[:top_k]]
    
    return top_k_list

def normalize_item_data(item: dict) -> dict:
    "Normalize Input Data for ARAG - From source : Yelp, Amazon, Goodreads"
    item_id = str(item.get('item_id', item.get('sub_item_id', 'unknown_id')))
    name = item.get('title') or item.get('name') or item.get('business_name') or f"Item {item_id}"
    raw_desc = item.get('description') or item.get('text') or ""
    if isinstance(raw_desc, list):
        clean_desc = " ".join([str(x) for x in raw_desc])
    else:
        clean_desc = str(raw_desc)
    category = item.get('categories') or item.get('type') or "General"
    if isinstance(category, list):
        category = ", ".join(category)

    return {
        "item_id": item_id,
        "name": name,
        "description": clean_desc,
        "category": str(category),
        "original_data": item 
    }

def _get_gcn_similar_items(anchor_item_ids: List[str], candidate_list: list, gcn_embeddings, top_k=3) -> List[dict]:
    if not gcn_embeddings or not anchor_item_ids:
        return []

    anchor_vecs = []
    for iid in anchor_item_ids:
        if iid in gcn_embeddings:
            anchor_vecs.append(gcn_embeddings[iid])
    
    if not anchor_vecs:
        return []
        
    query_vec = torch.stack(anchor_vecs).mean(dim=0).unsqueeze(0) 

    candidates_with_score = []
    
    for item in candidate_list:
        i_id = str(item['item_id'])
        if i_id in gcn_embeddings:
            target_vec = gcn_embeddings[i_id].unsqueeze(0) 
            sim = torch.nn.functional.cosine_similarity(query_vec, target_vec).item()
            candidates_with_score.append((item, sim))

    candidates_with_score.sort(key=lambda x: x[1], reverse=True)
    
    return [x[0] for x in candidates_with_score[:top_k]]