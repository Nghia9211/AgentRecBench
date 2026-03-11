import json
import os
import time
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
import argparse

def build_and_save_vector_store_batched(data_path: str, save_path: str, embed_model_name: str, batch_size: int = 256):
    print(f"Loading embedding model '{embed_model_name}'...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={'device': 'cuda'}
    )
    vector_store = None
    batch_count = 0
    start_time = time.time()

    with open(data_path, 'r', encoding='utf-8') as f:
        while True:
            batch_lines = [next(f, None) for _ in range(batch_size)]
            batch_lines = [line for line in batch_lines if line is not None]
            
            if not batch_lines:
                break

            batch_count += 1
            documents_batch = []
            for line in batch_lines:
                data = json.loads(line)
                doc = Document(
                    page_content=json.dumps(data, indent=2, ensure_ascii=False), 
                )
                documents_batch.append(doc)

            if not documents_batch:
                continue

            if vector_store is None:
                vector_store = FAISS.from_documents(
                    documents=documents_batch, 
                    embedding=embedding_function,
                    distance_strategy= "COSINE" 
                )
            else:
                vector_store.add_documents(documents=documents_batch)

    end_time = time.time()
    if vector_store is None:
        return
    vector_store.save_local(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save a FAISS vector store from a JSONL file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL data file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the local FAISS vector store.")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Name of the sentence-transformer model to use.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing documents.")
    
    args = parser.parse_args()
    
    build_and_save_vector_store_batched(
        data_path=args.data_path,
        save_path=args.save_path,
        embed_model_name=args.embed_model,
        batch_size=args.batch_size
    )