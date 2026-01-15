"""
Chat Interface Module
=====================
DAY 4: Main chat interface logic.

SOLID Principle: Single Responsibility Principle (SRP)
- This module handles chat orchestration

Topics to teach:
- Streamlit chat elements
- Streaming responses
- Error handling
- User experience
"""

import streamlit as st
from typing import Generator, Optional

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.chain import RAGChain
from tools.tavily_search import TavilySearchTool, HybridSearchManager
from ui.components import add_message, save_uploaded_file


class ChatInterface:
    """
    Main chat interface orchestrator.
    
    Coordinates between:
    - Document processing
    - Vector store
    - RAG chain
    - Web search
    """
    
    def __init__(self):
        """Initialize chat interface components."""
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        if self.vector_store.is_initialized:
            st.session_state.vector_store_initialized = True
        self.rag_chain: Optional[RAGChain] = None
        self.tavily_search = TavilySearchTool()
        self.hybrid_search: Optional[HybridSearchManager] = None
    
    def process_uploaded_files(self, uploaded_files) -> int:
        """
        Process uploaded files and add to vector store.
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            Number of chunks processed
        """
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            # Save file temporarily
            file_path = save_uploaded_file(uploaded_file)
            
            # Process the document
            chunks = self.doc_processor.process(file_path)
            
            # Add source metadata
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
            
            all_chunks.extend(chunks)
            
            # Track uploaded files
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)
        
        # Add to vector store
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            st.session_state.vector_store_initialized = True
        
        return len(all_chunks)
    
    def initialize_rag_chain(self):
        """Initialize the RAG chain after documents are loaded."""
        if self.vector_store.is_initialized:
            self.rag_chain = RAGChain(self.vector_store)
            self.hybrid_search = HybridSearchManager(
                self.vector_store,
                self.tavily_search
            )
    
    def get_response(
        self,
        query: str,
        use_web_search: bool = False
    ) -> Generator[str, None, None]:
        """
        Get a streaming response for a query.
        
        Args:
            query: User's question
            use_web_search: Whether to include web search
            
        Yields:
            Response chunks
        """
        # Initialize RAG chain if needed
        if self.rag_chain is None and self.vector_store.is_initialized:
            self.initialize_rag_chain()
        
        # If no documents and no web search, provide helpful message
        if not self.vector_store.is_initialized and not use_web_search:
            yield "Please upload some documents first, or enable web search to get started!"
            return
        
        # Web search enabled
        if use_web_search:
            # Get web search results
            web_results = self.tavily_search.search(query)
            
            # Get document results if available
            doc_results = []
            if self.vector_store.is_initialized:
                doc_results = self.vector_store.search(query)
                

            # Format context
            context_parts = []
            
            if doc_results:
                context_parts.append("=== From Your Documents ===")
                for i, doc in enumerate(doc_results, 1):
                    source = doc.metadata.get("source", "Unknown")
                    context_parts.append(f"[Doc {i}] ({source}):\n{doc.page_content}")
                    print(f"vectore-store_result:{doc.page_content}")
            
            if web_results:
                context_parts.append("\n=== From Web Search ===")
                context_parts.append(web_results)
            
            context = "\n\n".join(context_parts) if context_parts else "No context available."
            print("llm--context",context)
            print("doc_results--context",doc_results)
            # Generate response with context
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate
            from config.settings import settings
            
            llm = ChatGroq(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                api_key=settings.GROQ_API_KEY
            )
            
            prompt = ChatPromptTemplate.from_template(
                "Based on the following search results, answer the question concisely and accurately.\n\n"
                "Search Results:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer: "
            )
            
            chain = prompt | llm
            for chunk in chain.stream({"context": context, "question": query}):
                yield chunk.content
        
        # Document-only search
        elif self.rag_chain:
            for chunk in self.rag_chain.query_stream(query):
                yield chunk
    
    def get_sources(self, query: str, use_web_search: bool = False) -> list:
        """
        Get source documents for a query.
        
        Args:
            query: User's question
            use_web_search: Whether web search was used
            
        Returns:
            List of source document names
        """
        sources = []
        
        # Get semantic search sources
        if self.vector_store.is_initialized:
            docs = self.vector_store.search(query)
            sources.extend(list(set(doc.metadata.get("source", "Unknown") for doc in docs)))
        
        # Get web search sources
        if use_web_search:
            web_results = self.tavily_search.search(query)
            if web_results:
                sources.append("Web Search Results")
        
        return sources