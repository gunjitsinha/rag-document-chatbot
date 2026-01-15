"""
Document Processor Module
=========================
DAY 1: This module handles loading and splitting documents.

SOLID Principle: Single Responsibility Principle (SRP)
- This class has ONE job: process documents (load + split)

Topics to teach:
- Document loaders (TextLoader, PyPDFLoader)
- Text splitting strategies
- RecursiveCharacterTextSplitter explained
- Why chunking matters for RAG
"""

from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


class DocumentProcessor:
    """
    Handles document loading and text splitting.
    
    Supports:
    - Text files (.txt)
    - PDF files (.pdf)
    
    The text is split into chunks for efficient retrieval.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk (default from settings)
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Split hierarchy
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file type is not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}. Use .txt or .pdf")
        
        return loader.load()
    
    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Create documents from raw text.
        
        Args:
            text: Raw text content
            metadata: Optional metadata dictionary
            
        Returns:
            List containing a single Document
        """
        metadata = metadata or {}
        return [Document(page_content=text, metadata=metadata)]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        return self.text_splitter.split_documents(documents)
    
    def process(self, file_path: str) -> List[Document]:
        """
        Complete pipeline: load and split a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        # Step 1: Load the document
        documents = self.load_document(file_path)
        
        # Step 2: Split into chunks
        chunks = self.split_documents(documents)
        
        return chunks
    
    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Process raw text: create document and split.
        
        Args:
            text: Raw text content
            metadata: Optional metadata
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_from_text(text, metadata)
        return self.split_documents(documents)
