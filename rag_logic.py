import os
import tempfile
from pathlib import Path
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- Configuration
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_ID = "moonshotai/Kimi-K2-Instruct:novita"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

ALLOWED_EXTS = (".txt", ".pdf", ".docx")
MAX_FILE_SIZE_MB = 2
MAX_TOTAL_CHARS = 30000
DEMO_DOC_TEXT = (
    "Paul Atreides is the protagonist of Dune. "
    "He becomes the leader of the Fremen and is known as Muad'Dib. "
    "He is the son of Duke Leto Atreides and Lady Jessica."
)

# --- Embedding infrastructure (NO vectorstore at global scope)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
class MyEmbeddings:
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()
embedder = MyEmbeddings(embed_model)

def sanitize_and_load_files(files):
    docs, total_chars = [], 0
    for file in files:
        filename = file.name
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise ValueError(f"{filename}: Invalid file type.")
        file.seek(0, os.SEEK_END)
        size_mb = file.tell() / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"{filename}: >{MAX_FILE_SIZE_MB}MB limit.")
        file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_f:
            tmp_f.write(file.read())
            tmp_path = tmp_f.name
        try:
            if ext == ".txt":
                loaded = TextLoader(tmp_path).load()
            elif ext == ".pdf":
                loaded = PyPDFLoader(tmp_path).load()
            elif ext == ".docx":
                loaded = Docx2txtLoader(tmp_path).load()
            else:
                loaded = []
        finally:
            os.remove(tmp_path)
        for doc in loaded:
            total_chars += len(doc.page_content)
            if total_chars > MAX_TOTAL_CHARS:
                raise ValueError(f"All files: character limit ({MAX_TOTAL_CHARS}) exceeded.")
        docs.extend(loaded)
    if not docs:
        raise RuntimeError("No valid, non-empty docs uploaded.")
    return docs

def load_demo_docs():
    """Loads the demo document from the data/ directory."""
    demo_path = Path("data/dune-summary.txt")   # Change this extension if using PDF or DOCX.
    ext = demo_path.suffix.lower()
    if not demo_path.exists():
        raise FileNotFoundError("Demo file not found in ./data/. Please add demo.txt, demo.pdf, or demo.docx.")
    if ext == ".txt":
        return TextLoader(str(demo_path)).load()
    elif ext == ".pdf":
        return PyPDFLoader(str(demo_path)).load()
    elif ext == ".docx":
        return Docx2txtLoader(str(demo_path)).load()
    else:
        raise ValueError("Unsupported demo file type. Use txt, pdf, or docx.")

def chunk_documents(all_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        raise RuntimeError("No content after chunking.")
    return chunks

def build_chroma(chunks):
    # Always create a new in-memory Chroma instance, never persisted!
    return Chroma.from_documents(
        chunks, embedder, persist_directory=None
    )

def condense_question_llm(followup, last_user, last_assistant):
    prompt = (
        "Given the following conversation and a follow up question, "
        "rephrase the follow up question to be a standalone question, in its original language.\n\n"
        f"Chat History:\nHuman: {last_user}\nAI: {last_assistant}\n\n"
        f"Follow Up Input: {followup}\nStandalone question:"
    )
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": MODEL_ID
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def answer_llm(question, context, conversation_history=None):
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Always answer using ONLY the provided context and chat history. Cite documents as needed."
    }
    msgs = [system_msg]
    if conversation_history:
        msgs += conversation_history[-6:]  # Use last 3 Q/A pairs (optional)
    msgs.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })
    payload = {"messages": msgs, "model": MODEL_ID}
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def get_relevant_context(vectordb, question, top_k=3):
    q_emb = embed_model.encode([question], show_progress_bar=False, convert_to_numpy=True)[0].tolist()
    return vectordb.similarity_search_by_vector(q_emb, k=top_k)
