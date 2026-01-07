import json
from typing import List, Optional
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
import os

class MemoryARAG:
    def __init__(self,
                 llm: Optional[BaseChatModel] = None,
                 embedding_model: HuggingFaceEmbeddings = None,
                 vector_store: Optional[VectorStore] = None):
        self.llm = llm
        self.embedding_model = embedding_model
        
        self._db: Optional[VectorStore] = None
        self.retriever: Optional[BaseRetriever] = None

        if vector_store:
            self.db = vector_store
        
    @property
    def db(self) -> Optional[VectorStore]:
        return self._db

    @db.setter
    def db(self, value: VectorStore):
        self._db = value
        
        if self._db is not None:
            self.retriever = self._db.as_retriever(search_kwargs={'k': 5})
            print(self.retriever)
            print("Retriever has been automatically created from the vector store.")
        else:
            self.retriever = None
            print("Vector store is None. Retriever is cleared.")

    def search(self, query: str) -> List[Document]:
        if self.retriever is None:
            raise ValueError("Retriever has not been initialized. Please set a vector store (.db) first.")
            
        print(f"\nSearching for documents related to: '{query}'")
        results = self.retriever.invoke(query)
        return results
        
    def add_documents(self, documents: List[Document]): 
        if self._db is None:
            print(f"No existing vector store. Creating a new one with {len(documents)} documents.")
            self._db = FAISS.from_documents(documents=documents, embedding=self.embedding_model)
            self.retriever = self._db.as_retriever(search_kwargs={'k': 5})
        else:
            print(f"Adding {len(documents)} new documents to the existing vector store.")
            self._db.add_documents(documents=documents) 

