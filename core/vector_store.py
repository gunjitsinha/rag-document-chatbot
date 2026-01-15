"""
Vector Store Module
===================
DAY 2: This module handles FAISS vector database operations.

SOLID Principle: Single Responsibility Principle (SRP)
- This class has ONE job: manage the vector store

Topics to teach:
- What is a vector database?
- FAISS (Facebook AI Similarity Search)
- Indexing documents
- Similarity search
- Persistence (save/load)
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from core.embeddings import EmbeddingManager


class VectorStoreManager:
    """
    Manages FAISS vector store operations.
    
    FAISS is a FREE, local vector database that runs entirely on your machine.
    No external services or API costs!
    """
    
    def __init__(self, embedding_manager: EmbeddingManager = None):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_manager: EmbeddingManager instance (creates one if not provided)
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._vector_store: Optional[FAISS] = None
        self.index_path = settings.FAISS_INDEX_PATH
        try:
            if os.path.exists(self.index_path) and os.listdir(self.index_path):
                self.load()
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
    
    @property
    def vector_store(self) -> Optional[FAISS]:
        """Get the FAISS vector store instance."""
        return self._vector_store
    
    @property
    def is_initialized(self) -> bool:
        """Check if vector store has been initialized with documents."""
        return self._vector_store is not None
    
    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            FAISS vector store instance
        """
        self._vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_manager.embeddings
        )
        return self._vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of Document objects to add
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if not self.is_initialized:
            # If not initialized, create new store
            print("---create-from_documents--")
            self.create_from_documents(documents)
        else:
            # Add to existing store
            print("---vectorstore-from_documents--")
            self._vector_store.add_documents(documents)
    
    def search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return (default from settings)
            
        Returns:
            List of similar Document objects
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Add documents first.")
        
        k = k or settings.TOP_K_RESULTS
        return self._vector_store.similarity_search(query, k=k)
    
    def search_with_scores(
        self,
        query: str,
        k: int = None
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Add documents first.")
        
        k = k or settings.TOP_K_RESULTS
        return self._vector_store.similarity_search_with_score(query, k=k)
    
    def save(self, path: str = None) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save (default from settings)
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Nothing to save.")
        
        save_path = path or self.index_path
        os.makedirs(save_path, exist_ok=True)
        self._vector_store.save_local(save_path)
    
    def load(self, path: str = None) -> FAISS:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path to load from (default from settings)
            
        Returns:
            Loaded FAISS vector store
        """
        load_path = path or self.index_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No saved index found at {load_path}")
        
        self._vector_store = FAISS.load_local(
            load_path,
            self.embedding_manager.embeddings,
            allow_dangerous_deserialization=True  # Required for loading
        )
        return self._vector_store
    
    def get_retriever(self, k: int = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object compatible with LangChain chains
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized.")
        
        k = k or settings.TOP_K_RESULTS
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def clear(self) -> None:
        """Clear the vector store from memory."""
        self._vector_store = None
    
    def create_from_documents(self, documents: List[Document]) -> FAISS:
        self._vector_store = FAISS.from_documents(documents=documents, embedding=self.embedding_manager.embeddings)
        self.save()  # persist immediately
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        if not self.is_initialized:
            self.create_from_documents(documents)
        else:
            self._vector_store.add_documents(documents)
            self.save()  # persist incremental updates
