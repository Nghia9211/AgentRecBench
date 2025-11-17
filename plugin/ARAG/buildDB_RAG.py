import json
import os
import time
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

def build_and_save_vector_store_batched(data_path: str, save_path: str, embed_model_name: str, batch_size: int = 256):
    """
    Xây dựng và lưu trữ vector store từ file JSON, sử dụng các đối tượng Document.
    """
    print(f"Loading embedding model '{embed_model_name}'...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={'device': 'cuda'}
    )
    
    print(f"Reading data and processing into Documents in batches of {batch_size}...")
    
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
            print(f"Processing batch {batch_count} with {len(batch_lines)} documents...")

            documents_batch = []
            for line in batch_lines:
                data = json.loads(line)
                doc = Document(
                    # Nội dung để embedding vẫn là chuỗi JSON
                    page_content=json.dumps(data, indent=2, ensure_ascii=False), 
                    # Metadata là chính dictionary đã được parse, giúp truy cập sau này
                    metadata=data 
                )
                documents_batch.append(doc)

            if not documents_batch:
                continue

            if vector_store is None:
                print("Creating initial FAISS index from Documents...")
                # === THAY ĐỔI 3: Sử dụng FAISS.from_documents ===
                vector_store = FAISS.from_documents(
                    documents=documents_batch, 
                    embedding=embedding_function,
                    distance_strategy= "COSINE" 
                )
                print(f"Distance Strategey : {vector_store.distance_strategy}")
            else:
                vector_store.add_documents(documents=documents_batch)

    end_time = time.time()
    if vector_store is None:
        print("No data found to process.")
        return

    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
    
    print(f"Saving FAISS index to '{save_path}'...")
    vector_store.save_local(save_path)
    print("Index saved successfully!")

if __name__ == "__main__":
    # build_and_save_vector_store_batched(
    #     data_path=r'C:\Users\Admin\Desktop\Document\SpeechToText\AgentSocietyChallenge\dataset\output_data_all\item.json',
    #     save_path='item_storage',
    #     embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     batch_size=256
    # )

    build_and_save_vector_store_batched(
        data_path=r'C:\Users\Admin\Desktop\Document\SpeechToText\AgentSocietyChallenge\dataset\output_data_long\review.json',
        save_path='user_storage',
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=256
    )