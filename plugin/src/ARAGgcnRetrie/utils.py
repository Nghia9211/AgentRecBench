# --- START OF FILE utils.py ---

from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any, Callable, Dict
import torch
import torch.nn.functional as F
import json
import ast

def normalize_item_data(item: dict) -> dict:
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

def get_batch_embeddings(texts: List[str], embedding_function: Callable) -> torch.Tensor:
    vectors = embedding_function.embed_documents(texts)
    return torch.tensor(vectors)

def generate_graph_context_string(
    user_history_ids: List[str],
    gcn_embeddings: Dict[str, torch.Tensor],
    candidate_list: List[dict],
    top_k_neighbors: int = 5
) -> str:
    """
    Hàm này đóng vai trò 'GCN Context':
    Nó nhìn vào đồ thị (Embedding) để tìm các item có cấu trúc mạng lưới gần nhất với user,
    sau đó trả về một chuỗi văn bản mô tả các item này để làm gợi ý cho RAG.
    """
    if not gcn_embeddings or not user_history_ids:
        return ""

    # 1. Tạo User Embedding từ lịch sử (Mean Pooling các node GCN)
    user_vecs = []
    for uid in user_history_ids:
        if uid in gcn_embeddings:
            user_vecs.append(gcn_embeddings[uid])
    
    if not user_vecs:
        return ""
    
    user_emb = torch.stack(user_vecs).mean(dim=0).unsqueeze(0) # (1, dim)

    # 2. Tìm các Item trong Candidate List có vector GCN gần nhất (Graph Similarity)
    # Lưu ý: Đây chỉ là tìm 'gợi ý' từ đồ thị để tạo context, chưa phải final result
    candidates_with_gcn = []
    for item in candidate_list:
        iid = str(item['item_id'])
        if iid in gcn_embeddings:
            candidates_with_gcn.append((item, gcn_embeddings[iid]))
    
    if not candidates_with_gcn:
        return ""

    cand_tensor = torch.stack([x[1] for x in candidates_with_gcn])
    
    # Tính cosine similarity trong không gian GCN
    scores = F.cosine_similarity(user_emb, cand_tensor)
    
    # Lấy Top K items theo mạng lưới đồ thị
    top_indices = torch.topk(scores, k=min(top_k_neighbors, len(candidates_with_gcn))).indices.tolist()
    
    suggested_items = [candidates_with_gcn[i][0] for i in top_indices]
    
    # 3. Tạo câu văn Context (Prompt Expansion)
    item_names = [f"'{item['name'] or item['title']}' ({item['category']})" for item in suggested_items]
    
    context_str = (
        f"Graph Analysis: The user's interaction network strongly aligns with these items: {', '.join(item_names)}. "
        "Items sharing similar structural patterns in the graph should be prioritized."
    )
    
    return context_str

def perform_rag_retrieval(
    augmented_query: str,
    candidate_list: List[dict],
    embedding_function: Callable,
    top_k: int = 7
) -> List[dict]:
    """
    Hàm RAG thuần túy:
    Dùng câu query (đã được bơm GCN Context) để tìm kiếm Semantic Similarity.
    """
    sims = []
    
    query_vec = embedding_function.embed_query(augmented_query)
    
    item_texts = [str(item) for item in candidate_list]
    item_vectors = embedding_function.embed_documents(item_texts) 
    
    for item, item_vec in zip(candidate_list, item_vectors):
        sim = cosine_similarity([item_vec], [query_vec])[0][0]
        sims.append((item, sim))
    
    sims.sort(key = lambda x: x[1], reverse = True)
    top_k_list = [item for item, sim in sims[:top_k]]
    
    return top_k_list


def print_agent_step(agent_name: str, message: str, data: Any = None):
    header = f"=== [AGENT: {agent_name.upper()}] ==="
    print(f"\n\033[94m{header}\033[0m") 
    print(f"💬 {message}")
    if data:
        if isinstance(data, list):
            print(f"📊 Items count: {len(data)}")
            for i, item in enumerate(data[:3]):
                print(f"   - Item {i+1}: {item}")
            if len(data) > 3: print("   ...")
        else:
            print(f"📝 Data: {data}")
    print("\033[94m" + "="*len(header) + "\033[0m\n")

def get_gcn_latent_interests(user_history_ids, gcn_embeddings, candidate_list):
    """
    Tìm ra 'Gu' của người dùng dựa trên các item lân cận trong đồ thị.
    """
    if not gcn_embeddings or not user_history_ids:
        return "No specific graph patterns detected."

    # 1. Tìm User Embedding
    user_vecs = [gcn_embeddings[uid] for uid in user_history_ids if uid in gcn_embeddings]
    if not user_vecs: return "No graph data available."
    user_emb = torch.stack(user_vecs).mean(dim=0)

    # 2. Tìm các Item 'hàng xóm' trong đồ thị
    cand_ids = [str(item['item_id']) for item in candidate_list if str(item['item_id']) in gcn_embeddings]
    if not cand_ids: return "New user profile with limited graph connections."
    
    cand_tensor = torch.stack([gcn_embeddings[iid] for iid in cand_ids])
    sims = torch.nn.functional.cosine_similarity(user_emb.unsqueeze(0), cand_tensor)
    
    top_k = torch.topk(sims, k=min(5, len(cand_ids)))
    
    # 3. Lấy đặc điểm chung của các item này (Category, Style)
    suggested_features = []
    for idx in top_k.indices:
        item_id = cand_ids[idx]
        item_data = next(i for i in candidate_list if str(i['item_id']) == item_id)
        suggested_features.append(f"{item_data.get('category')} ({item_data.get('name') or item_data.get('title')} )")

    return f"Structural community patterns suggest a preference for: {', '.join(suggested_features)}."