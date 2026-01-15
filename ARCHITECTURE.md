# 🏗️ RAG Chatbot - Architecture Documentation

## 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                            │
│                              (Streamlit)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  Chat Interface  │  │  File Uploader   │  │  Sidebar Info    │      │
│  │  (chat_interface)│  │  (components)    │  │  (components)    │      │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────────┘      │
│           │                     │                                        │
│           │                     │                                        │
└───────────┼─────────────────────┼────────────────────────────────────────┘
            │                     │
            │                     │
┌───────────┼─────────────────────┼────────────────────────────────────────┐
│           │    APPLICATION LAYER (app.py)   │                            │
│           │                     │                                        │
│           ▼                     ▼                                        │
│  ┌─────────────────────────────────────────┐                            │
│  │         Configuration Layer             │                            │
│  │          (config/settings.py)           │                            │
│  │  • API Keys (Groq, Tavily)              │                            │
│  │  • Environment Variables                │                            │
│  │  • Model Settings                       │                            │
│  └─────────────────────────────────────────┘                            │
└───────────┬───────────────────────────────────────────────────────────┬─┘
            │                                                             │
            │                                                             │
┌───────────┼─────────────────────────────────────────────────────────┬─┼─┐
│           │             CORE RAG PIPELINE                             │ │ │
│           │                                                             │ │
│  ┌────────▼──────────┐    ┌──────────────────┐    ┌─────────────────┼─┘ │
│  │   RAG Chain       │◄───┤  Vector Store    │◄───┤  Document       │   │
│  │   Orchestration   │    │  Manager         │    │  Processor      │   │
│  │  (chain.py)       │    │ (vector_store.py)│    │(document_       │   │
│  │                   │    │                  │    │ processor.py)   │   │
│  │ • Query handling  │    │ • FAISS Index    │    │ • Load docs     │   │
│  │ • Context inject  │    │ • Similarity     │    │ • Text split    │   │
│  │ • Response gen    │    │   search         │    │ • Chunk create  │   │
│  │ • Streaming       │    │ • Persistence    │    │                 │   │
│  └────────┬──────────┘    └──────────────────┘    └─────────────────┘   │
│           │                        ▲                        ▲            │
│           │                        │                        │            │
│           │                 ┌──────┴──────┐          ┌─────┴──────┐     │
│           │                 │  Embeddings │          │   Data     │     │
│           │                 │   Manager   │          │  Storage   │     │
│           │                 │(embeddings  │          │ documents/ │     │
│           │                 │    .py)     │          │faiss_index/│     │
│           │                 │             │          │            │     │
│           │                 │• HuggingFace│          │            │     │
│           │                 │• sentence-  │          │            │     │
│           │                 │  transformers│         │            │     │
│           │                 │• Local exec │          │            │     │
│           │                 └─────────────┘          └────────────┘     │
└───────────┼─────────────────────────────────────────────────────────────┘
            │
            │
┌───────────┼─────────────────────────────────────────────────────────────┐
│           │              EXTERNAL SERVICES LAYER                         │
│           │                                                               │
│  ┌────────▼──────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │   LLM Provider    │    │  Search Tool     │    │  Embeddings     │  │
│  │   Groq (FREE)     │    │ Tavily Search    │    │  HuggingFace    │  │
│  │                   │    │ (tavily_search   │    │    (FREE)       │  │
│  │ • Llama 3.1 70B   │    │      .py)        │    │                 │  │
│  │ • Fast inference  │    │                  │    │ • Local model   │  │
│  │ • Streaming       │    │ • Web search     │    │ • No API cost   │  │
│  │                   │    │ • Real-time data │    │                 │  │
│  └───────────────────┘    └──────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION FLOW                           │
└──────────────────────────────────────────────────────────────────────────┘

User Upload          Document              Text              Embedding
Documents     ───►   Processor     ───►    Splitter   ───►   Generation
(.txt, .pdf)        (Load & Parse)        (Chunking)        (HuggingFace)
                                                                    │
                                                                    ▼
                                                            ┌───────────────┐
                                                            │  FAISS Index  │
                                                            │   (Persist)   │
                                                            └───────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                          QUERY PROCESSING FLOW                            │
└──────────────────────────────────────────────────────────────────────────┘

User Query  ───►  Embed Query  ───►  Similarity  ───►  Retrieve
                  (HuggingFace)      Search           Top-K Docs
                                     (FAISS)              │
                                                          │
┌─────────────────────────────────────────────────────────┘
│
▼
Context     ───►   Build      ───►   LLM         ───►   Stream
+ Query            Prompt           (Groq)             Response
                   Template                            to User


┌──────────────────────────────────────────────────────────────────────────┐
│                      WEB SEARCH AUGMENTATION FLOW                         │
└──────────────────────────────────────────────────────────────────────────┘

User Query  ───►  Insufficient  ───►  Tavily    ───►  LLM        ───►  Stream
               │  Local Context       Search         (Groq)            Response
               │                      Results
               │
               └──► If relevant docs found in FAISS
                    (Skip web search)
```

## 🧩 Component Breakdown

### 1️⃣ Configuration Layer (`config/`)
**Responsibility**: Centralized configuration management

- **settings.py**
  - Environment variables (.env)
  - API key management (Groq, Tavily)
  - Model configurations
  - Directory paths
  - Validation logic

### 2️⃣ Core RAG Pipeline (`core/`)
**Responsibility**: Core business logic for RAG operations

#### Document Processor (`document_processor.py`)
- Loads documents (TXT, PDF)
- Splits text into chunks (RecursiveCharacterTextSplitter)
- Handles multiple file formats
- Configurable chunk size/overlap

#### Embeddings Manager (`embeddings.py`)
- HuggingFace sentence-transformers integration
- Local embedding generation (FREE)
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- No external API calls

#### Vector Store Manager (`vector_store.py`)
- FAISS index management
- Document indexing
- Similarity search (k=4 by default)
- Persistence (save/load from disk)
- Incremental updates

#### RAG Chain (`chain.py`)
- Orchestrates retrieval + generation
- Groq LLM integration (Llama 3.1 70B)
- Prompt template management
- Context injection
- Streaming responses
- Tavily search fallback

### 3️⃣ Tools Layer (`tools/`)
**Responsibility**: External tool integrations

#### Tavily Search (`tavily_search.py`)
- Web search capabilities
- Real-time information retrieval
- Configurable search depth
- Fallback for insufficient local context

### 4️⃣ UI Layer (`ui/`)
**Responsibility**: User interface and interaction

#### Components (`components.py`)
- Session state management
- Chat history display
- File uploader widget
- Sidebar information
- Reusable UI elements

#### Chat Interface (`chat_interface.py`)
- Chat logic orchestration
- Message handling
- Document processing coordination
- Response streaming management

### 5️⃣ Application Entry (`app.py`)
**Responsibility**: Main application orchestration

- Streamlit app initialization
- Component integration
- API key validation
- Application flow control

## 🔑 Key Design Principles

### SOLID Principles Applied

1. **Single Responsibility Principle (SRP)**
   - Each module has one clear purpose
   - DocumentProcessor → only handles documents
   - VectorStoreManager → only manages vectors
   - RAGChain → only orchestrates RAG

2. **Open/Closed Principle (OCP)**
   - Easy to add new document types
   - Easy to swap LLM providers
   - Easy to add new search tools

3. **Dependency Inversion Principle (DIP)**
   - High-level modules don't depend on low-level details
   - Configuration abstraction
   - Service abstraction layers

### Design Patterns

- **Factory Pattern**: Document loader creation
- **Strategy Pattern**: Different search strategies (local vs web)
- **Singleton Pattern**: Configuration settings
- **Observer Pattern**: Streaming response updates

## 🛠️ Technology Stack

| Component | Technology | Cost | Notes |
|-----------|-----------|------|-------|
| **LLM** | Groq (Llama 3.1 70B) | FREE | Fast inference, 8k context |
| **Embeddings** | HuggingFace (sentence-transformers) | FREE | Runs locally, 384 dims |
| **Vector DB** | FAISS | FREE | Local, fast similarity search |
| **Web Search** | Tavily API | Paid | 1000 free searches/month |
| **Framework** | LangChain | FREE | RAG orchestration |
| **UI** | Streamlit | FREE | Web app framework |
| **Language** | Python 3.11+ | FREE | - |

## 📦 Module Dependencies

```
app.py
├── config.settings
├── ui.components
│   └── config.settings
├── ui.chat_interface
│   ├── core.chain
│   ├── core.document_processor
│   └── core.vector_store
└── core.chain
    ├── core.vector_store
    │   └── core.embeddings
    ├── tools.tavily_search
    │   └── config.settings
    └── config.settings
```

## 🔐 Security Considerations

1. **API Keys**: Stored in `.env` file (not committed to git)
2. **Validation**: Early validation of required settings
3. **Error Handling**: Graceful degradation when services unavailable
4. **Data Privacy**: All embeddings generated locally
5. **File Isolation**: Documents stored in isolated directory

## 🚀 Scalability Considerations

### Current Architecture (Single User)
- Local FAISS index
- Local embeddings
- Streamlit session state

### Future Scalability Options
1. **Multi-User**
   - Replace FAISS with Pinecone/Weaviate
   - Add user authentication
   - Database for chat history

2. **Production Deployment**
   - Docker containerization
   - Cloud storage for documents
   - Load balancer for multiple instances
   - Redis for session management

3. **Performance Optimization**
   - Async processing for document ingestion
   - Batch embedding generation
   - Caching layer for common queries
   - CDN for static assets

## 📈 Performance Metrics

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Document Upload (1MB) | 2-5s | Depends on chunking |
| Embedding Generation (1 chunk) | 50-100ms | Local model |
| Vector Search | 10-50ms | FAISS is fast |
| LLM Response (streaming) | 1-3s first token | Groq is very fast |
| Full Query (end-to-end) | 2-5s | With streaming |

## 🧪 Testing Strategy

### Unit Tests
- Document processor (chunking logic)
- Embedding generation (dimension validation)
- Vector store operations (add/search/persist)

### Integration Tests
- End-to-end RAG pipeline
- File upload → index → query flow
- Tavily search integration

### UI Tests
- Session state management
- File upload handling
- Chat history persistence

## 📚 Learning Path (4-Day Workshop)

### Day 1: Foundations
- **Focus**: Document processing and text splitting
- **Files**: `config/`, `core/document_processor.py`
- **Concepts**: RAG introduction, document loaders, chunking strategies

### Day 2: Vector Store & Embeddings
- **Focus**: Embeddings and similarity search
- **Files**: `core/embeddings.py`, `core/vector_store.py`
- **Concepts**: Vector embeddings, FAISS, semantic search

### Day 3: RAG Chain & Tools
- **Focus**: LLM integration and tool usage
- **Files**: `core/chain.py`, `tools/tavily_search.py`
- **Concepts**: Prompt engineering, chain composition, web search

### Day 4: UI & Deployment
- **Focus**: User interface and application deployment
- **Files**: `ui/`, `app.py`
- **Concepts**: Streamlit, streaming, deployment options

## 🔄 State Management

```
Session State (Streamlit)
├── messages: List[Dict]           # Chat history
├── vector_store: VectorStoreManager
├── rag_chain: RAGChain
├── processed_files: Set[str]      # Avoid re-processing
└── use_web_search: bool           # Toggle web search
```

## 🌐 API Interactions

### Groq API
```
POST https://api.groq.com/v1/chat/completions
├── Headers: Authorization: Bearer {API_KEY}
├── Body: {model, messages, stream: true, ...}
└── Response: Server-Sent Events (SSE) stream
```

### Tavily API
```
POST https://api.tavily.com/search
├── Headers: Authorization: Bearer {API_KEY}
├── Body: {query, search_depth, max_results, ...}
└── Response: {results: [{title, url, content, ...}]}
```

### HuggingFace (Local)
```
Local Model Loading
├── Model: sentence-transformers/all-MiniLM-L6-v2
├── Input: Text string
├── Output: 384-dimensional vector
└── No network required (after initial download)
```

## 📖 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Groq API Documentation](https://console.groq.com/docs)
- [Tavily Search API](https://tavily.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Last Updated**: January 2, 2026
**Project**: RAG Chatbot Workshop - 4 Day Course
**Version**: 1.0.0
