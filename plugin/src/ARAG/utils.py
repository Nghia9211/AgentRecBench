from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any, Callable

def find_top_k_similar_items(
    query: str,
    candidate_list: List[Any],
    embedding_function: Callable,
    top_k: int = 5
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

def print_agent_step(agent_name: str, message: str, data: Any = None):
    header = f"=== [AGENT: {agent_name.upper()}] ==="
    print(f"\n\033[94m{header}\033[0m") # Màu xanh dương
    print(f"💬 {message}")
    if data:
        if isinstance(data, list):
            print(f"📊 Items count: {len(data)}")
            for i, item in enumerate(data[:3]): # In 3 cái đầu
                print(f"   - Item {i+1}: {item}")
            if len(data) > 3: print("   ...")
        else:
            print(f"📝 Data: {data}")
    print("\033[94m" + "="*len(header) + "\033[0m\n")