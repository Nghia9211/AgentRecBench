import json
from typing import List, Optional
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
import os

# ==============================================================================
# ĐỊNH NGHĨA LỚP MemoryDILU (Dán lại lớp của bạn ở đây để file có thể chạy độc lập)
# ==============================================================================
class MemoryARAG:
    """
    Một lớp quản lý bộ nhớ dài hạn cho hệ thống RAG, sử dụng một Vector Store.
    Nó đóng gói vector store (db) và retriever tương ứng.
    """
    def __init__(self,
                 llm: Optional[BaseChatModel] = None,
                 embedding_model: HuggingFaceEmbeddings = None,
                 vector_store: Optional[VectorStore] = None):
        """
        Khởi tạo MemoryARAG.
        """
        print("Initializing MemoryARAG...")
        self.llm = llm
        self.embedding_model = embedding_model
        
        self._db: Optional[VectorStore] = None
        self.retriever: Optional[BaseRetriever] = None

        if vector_store:
            self.db = vector_store
        
    @property
    def db(self) -> Optional[VectorStore]:
        """Getter cho thuộc tính db."""
        return self._db

    @db.setter
    def db(self, value: VectorStore):
        """
        Setter cho thuộc tính db.
        Khi một vector store được gán, tự động tạo ra một retriever tương ứng.
        """
        print("Vector store has been set in MemoryDILU.")
        self._db = value
        
        if self._db is not None:
            self.retriever = self._db.as_retriever(search_kwargs={'k': 5})
            print(self.retriever)
            print("Retriever has been automatically created from the vector store.")
        else:
            self.retriever = None
            print("Vector store is None. Retriever is cleared.")

    def search(self, query: str) -> List[Document]:
        """
        Thực hiện tìm kiếm các tài liệu liên quan trong vector store.
        """
        if self.retriever is None:
            raise ValueError("Retriever has not been initialized. Please set a vector store (.db) first.")
            
        print(f"\nSearching for documents related to: '{query}'")
        results = self.retriever.invoke(query)
        return results
        
    def add_documents(self, documents: List[Document]): # Cập nhật type hint thành List[Document]
        """
        Thêm các tài liệu mới (dạng Document) vào vector store.
        """
        if self._db is None:
            print(f"No existing vector store. Creating a new one with {len(documents)} documents.")
            self._db = FAISS.from_documents(documents=documents, embedding=self.embedding_model)
            self.retriever = self._db.as_retriever(search_kwargs={'k': 5})
        else:
            print(f"Adding {len(documents)} new documents to the existing vector store.")
            self._db.add_documents(documents=documents) # Sử dụng add_documents

def main():
    """
    Hàm chính để tải vector store đã lưu và thực hiện tìm kiếm.
    """
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    index_path = "user_storage"
    absolute_load_path = os.path.abspath(index_path)

    print(f"Loading embedding model '{embed_model_name}'...")
    embedding_function = HuggingFaceEmbeddings(model_name=embed_model_name)

    print(f"Loading FAISS index from '{absolute_load_path}'...")
    loaded_vector_store = FAISS.load_local(
        folder_path=absolute_load_path, 
        embeddings=embedding_function,
        allow_dangerous_deserialization=True,
        distance_strategy = "COSINE" 
    )
    print("FAISS index loaded successfully!")
    print(f" ---- Distance strategy being used: {loaded_vector_store.distance_strategy}")

    memory = MemoryARAG(
        embedding_model=embedding_function,
        vector_store=loaded_vector_store
    )

    query = "25784128"
    search_results = memory.search(query)

    print(f"\nFound {len(search_results)} relevant documents for the query.")
    print("=========================================")
    for i, doc in enumerate(search_results):
        print(f"--- Result {i+1} ---")
        
        # === THAY ĐỔI: In cả page_content và metadata ===
        # page_content vẫn là chuỗi JSON
        print(f"Page Content (dùng để embedding):")
        print(doc.page_content)
        
        # Giờ đây chúng ta có thể truy cập metadata một cách dễ dàng
        print(f"\nMetadata (dữ liệu gốc):")
        print(json.dumps(doc.metadata, indent=2, ensure_ascii=False))

        # Ví dụ truy cập một trường cụ thể trong metadata (giả sử có trường 'title')
        # if 'title' in doc.metadata:
        #    print(f"Title from metadata: {doc.metadata['title']}")
            
        print("\n")

if __name__ == "__main__":
    main()