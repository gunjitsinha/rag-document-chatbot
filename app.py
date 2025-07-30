import streamlit as st
from rag_logic import (
    sanitize_and_load_files, load_demo_docs, chunk_documents, build_chroma,
    condense_question_llm, answer_llm, get_relevant_context
)

st.set_page_config(
    page_title="Conversational RAG Chatbot", page_icon="🤖", layout="wide"
)

with st.sidebar:
    st.title("🔎 RAG Chatbot Demo")
    st.info(
        "• Upload your .txt, .pdf, or .docx files (max 2MB each, 30,000 chars total).\n"
        "• Chat with your knowledge base! Or try demo mode with no upload.\n"
        "• Data is private to your session and cleared on refresh/reset."
    )
    if st.button("🔄 Reset Session", type="primary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# --- Step 1: Handle file upload/demo and force context reset EVERY run ---
st.header("1. Upload Your Documents (or try demo ⏩)")
uploaded_files = st.file_uploader(
    "Upload txt/pdf/docx (max 2MB/file)", type=["txt", "pdf", "docx"], accept_multiple_files=True
)

# Always (re)build doc chunks, vector DB, and clear chat on file upload, demo, or reset
try:
    # Always load docs, chunk, embed, and rebuild vector DB
    raw_docs = sanitize_and_load_files(uploaded_files) if uploaded_files else load_demo_docs()
    chunks = chunk_documents(raw_docs)
    vectordb = build_chroma(chunks)

    # Set fresh state every time
    st.session_state["chunks"] = chunks
    st.session_state["vectordb"] = vectordb
    if "chat_history" not in st.session_state or st.session_state.get("clear_history_flag", False):
        st.session_state["chat_history"] = []
        st.session_state["clear_history_flag"] = False

    st.success(f"Loaded {len(raw_docs)} doc(s).")
except Exception as e:
    st.error(str(e))
    st.stop()

# --- Step 2: Chat logic (session state only contains current doc, vec, and chat) ---
st.header("2. Chat With Your Documents")
for msg in st.session_state["chat_history"]:
    st.chat_message(msg["role"]).write(msg["content"])
user_input = st.chat_input("Type your question…")
if user_input:
    history = st.session_state["chat_history"]
    if len([m for m in history if m["role"] == "user"]) >= 1:
        # Get last Q, last A
        user_msgs = [m for m in history if m["role"] == "user"]
        assistant_msgs = [m for m in history if m["role"] == "assistant"]
        last_user = user_msgs[-1]["content"] if user_msgs else ""
        last_assistant = assistant_msgs[-1]["content"] if assistant_msgs else ""
        with st.spinner("Condensing your question for follow-up context..."):
            condensed = condense_question_llm(user_input, last_user, last_assistant)
    else:
        condensed = user_input
    # Retrieval
    with st.spinner("Retrieving relevant info..."):
        vectordb = st.session_state["vectordb"]
        retrieved_docs = get_relevant_context(vectordb, condensed)
        if not retrieved_docs:
            st.error("No relevant document context found.")
            st.stop()
        context = "\n".join([d.page_content for d in retrieved_docs])
    # LLM answer
    with st.spinner("AI answering..."):
        answer = answer_llm(user_input, context, conversation_history=st.session_state["chat_history"])
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(answer)

# --- Step 3: Download chat history ---
if st.session_state.get("chat_history"):
    hist_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["chat_history"]
    )
    st.download_button("⤓ Download Chat Transcript", hist_text, file_name="chat_history.txt")
