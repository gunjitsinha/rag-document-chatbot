"""
Embeddings Module
=================
DAY 2: This module handles text embeddings using HuggingFace models.

SOLID Principle: Single Responsibility Principle (SRP)
- This class has ONE job: create embeddings from text

Topics to teach:
- What are embeddings?
- Vector representations of text
- Semantic similarity
- HuggingFace sentence-transformers (FREE!)
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class EmbeddingManager:
    """
    Manages text embeddings using HuggingFace models.
    
    Uses sentence-transformers which are FREE and run locally!
    No API costs for embeddings.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: HuggingFace model name (default from settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        # Initialize HuggingFace embeddings (downloads model on first use)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},  # Use CPU for compatibility
            encode_kwargs={"normalize_embeddings": True}  # Normalize for cosine similarity
        )
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get the embeddings model instance."""
        return self._embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self._embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer dimension size
        """
        # Create a sample embedding to get dimension
        sample_embedding = self.embed_text("sample")
        return len(sample_embedding)
