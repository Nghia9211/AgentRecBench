"""
build_faiss_yelp.py
─────────────────────────
Build FAISS index với rich content cho Yelp.
"""

import json
import os
import time
import argparse
import re

def parse_camel_case(text: str) -> str:
    """Chuyển 'BikeParking' thành 'Bike Parking' để embed tốt hơn"""
    return re.sub(r"(\w)([A-Z])", r"\1 \2", text)

def build_item_text(item: dict) -> str:
    """
    Tạo document text semantic-rich cho 1 Business trên Yelp.
    """
    parts = []

    # ── Name (boost 3×) ──────────────────────────────────────────────────
    name = (item.get('name') or '').strip()
    if name:
        parts.extend([name, name, name])

    # ── Categories ───────────────────────────────────────────────────────
    categories = (item.get('categories') or '').strip()
    if categories:
        parts.append(f"Categories: {categories}")

    # ── Location ─────────────────────────────────────────────────────────
    city = (item.get('city') or '').strip()
    state = (item.get('state') or '').strip()
    if city and state:
        parts.append(f"Location: {city}, {state}")

    # ── Attributes (Trích xuất các tính năng True) ───────────────────────
    attributes = item.get('attributes')
    if attributes and isinstance(attributes, dict):
        active_attrs = []
        for k, v in attributes.items():
            if str(v).lower() == 'true' or str(v).lower() == "u'free'":
                active_attrs.append(parse_camel_case(k))
        if active_attrs:
            # Lấy tối đa 10 attributes quan trọng để tạo feature string
            parts.append("Features: " + ", ".join(active_attrs[:10]))

    # ── Rating ───────────────────────────────────────────────────────────
    stars = item.get('stars')
    if stars:
        try:
            r = float(stars)
            if 0 < r <= 5:
                parts.append(f"Rated {r:.1f} stars")
        except (ValueError, TypeError):
            pass

    # ── Fallback ──────────────────────────────────────────────────────────
    if not parts:
        return (item.get('name') or '') or str(item.get('item_id', 'unknown'))

    return " | ".join(filter(None, parts))


def dry_run(data_path: str, n: int = 5):
    """In ra n items đầu để verify rich text trước khi build thật."""
    print("\n" + "=" * 65)
    print("  DRY RUN — Kiểm tra rich text Yelp trước khi build")
    print("=" * 65)

    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            rich = build_item_text(item)

            print(f"\n--- Item {count+1} ---")
            print(f"  item_id : {item.get('item_id', '?')}")
            print(f"  name    : {item.get('name', '?')}")
            print(f"\n  Rich text ({len(rich.split())} words):")
            for part in rich.split(' | '):
                print(f"    • {part}")

            count += 1
            if count >= n: break

    print("\n" + "=" * 65)


def build_and_save(data_path: str, save_path: str, embed_model: str, batch_size: int):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    print(f"\n[Build] Loading embedding model: {embed_model}")
    embedding_function = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'batch_size': batch_size},
    )

    vector_store = None
    batch_count  = 0
    total_docs   = 0
    empty_rich   = 0
    start_time   = time.time()

    print(f"[Build] Reading {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        while True:
            batch_lines = [line for _, line in zip(range(batch_size), f) if line.strip()]
            if not batch_lines: break

            batch_count += 1
            documents_batch = []

            for line in batch_lines:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rich_text = build_item_text(item)
                if not rich_text.strip():
                    empty_rich += 1
                    rich_text = item.get('name', '') or f"item_{item.get('item_id', '')}"

                doc = Document(
                    page_content=rich_text,
                    metadata={
                        'item_id': str(item.get('item_id', '')),
                        'item_name': item.get('name', '').strip()
                    }
                )
                documents_batch.append(doc)

            if not documents_batch: continue
            total_docs += len(documents_batch)

            if vector_store is None:
                vector_store = FAISS.from_documents(documents=documents_batch, embedding=embedding_function, distance_strategy="COSINE")
            else:
                vector_store.add_documents(documents=documents_batch)

            print(f"[Build] Batch {batch_count} | docs={total_docs} | elapsed={time.time() - start_time:.0f}s")

    if vector_store:
        os.makedirs(save_path, exist_ok=True)
        vector_store.save_local(save_path)
        print(f"\n[Build] Done! Saved {total_docs} docs to {save_path}")


def verify_only(save_path: str, embed_model: str):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"\n[Verify] Loading index from {save_path}...")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vs = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True, distance_strategy="COSINE")

    queries = [
        "department store with bike parking in Tucson",
        "bakery with bubble tea and free wifi in Philadelphia",
    ]

    print("\n" + "=" * 65)
    print("  VERIFY — Yelp Similarity search results")
    print("=" * 65)

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = vs.similarity_search(query, k=3)
        for j, doc in enumerate(results):
            name = doc.metadata.get('item_name', '')
            print(f"  {j+1}. {name}")
            print(f"     {doc.page_content[:80]}...")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True)
    p.add_argument('--save_path', required=True)
    p.add_argument('--embed_model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--n_dry', type=int, default=5)
    p.add_argument('--verify_only', action='store_true')
    args = p.parse_args()

    if args.dry_run:
        dry_run(args.data_path, args.n_dry)
    elif args.verify_only:
        verify_only(args.save_path, args.embed_model)
    else:
        build_and_save(args.data_path, args.save_path, args.embed_model, args.batch_size)
        verify_only(args.save_path, args.embed_model)