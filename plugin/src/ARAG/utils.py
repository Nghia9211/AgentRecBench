
from typing import List, Any, Callable
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity
import re

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

def parse_structured_output(text: str, pydantic_model):
    """Bóc tách JSON từ text và ép kiểu vào Pydantic model"""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group()
            data = json.loads(json_str)
            return pydantic_model(**data)
        else:
            data = json.loads(text)
            return pydantic_model(**data)
    except Exception as e:
        print(f"Lỗi parse JSON: {e} | Text: {text}")
        return None

def call_llm(prompt, max_tokens=10000):
    url = "http://159.223.39.25:5678/llm/simple-chat"
    
    params = {
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    
    
    try:
        response = requests.post(url, params=params, timeout=60)
        
        response.raise_for_status()

        print(f"\n\n\n Response : {response.text} \n\n\n")
        
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Lỗi kết nối: {e}"
    

def get_json_format_instructions(pydantic_model):
    """Tạo hướng dẫn định dạng JSON dựa trên Pydantic model."""
    schema = pydantic_model.schema()
    return f"\n\nIMPORTANT: You must return ONLY a JSON object that matches this schema:\n{json.dumps(schema, indent=2)}\nDO NOT include any explanation or markdown code blocks like ```json."

def parse_structured_output(text: str, pydantic_model):
    """Bóc tách JSON từ text (có xử lý wrapper 'response') và ép kiểu vào Pydantic model"""
    try:
        # 1. Thử parse lớp vỏ bên ngoài (API wrapper)
        try:
            outer_data = json.loads(text)
            if isinstance(outer_data, dict) and "response" in outer_data:
                # Lấy nội dung bên trong trường 'response'
                text = outer_data["response"]
        except Exception:
            # Nếu không phải JSON wrapper, cứ để text nguyên bản để xử lý tiếp
            pass

        # 2. Làm sạch markdown (```json ...)
        clean_text = re.sub(r'```json|```', '', text).strip()

        # 3. Tìm khối JSON { ... } thực sự
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if match:
            json_str = match.group()
            data = json.loads(json_str)
            return pydantic_model(**data)
        else:
            # Thử parse trực tiếp nếu không tìm thấy cặp ngoặc qua regex
            data = json.loads(clean_text)
            return pydantic_model(**data)
            
    except Exception as e:
        print(f"❌ Lỗi parse JSON: {e} | Text nhận được: {text[:200]}...")
        return None