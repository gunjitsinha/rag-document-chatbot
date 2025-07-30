# Conversational RAG Chatbot Demo

**A Modern Web App for Exploring Retrieval-Augmented Generation on Your Own Docs**

![Streamlit RAG Chatbot UI Screenshot](data/rag-chat-ui.png)

## What is this project?

This is a clean, recruiter-ready demo of a conversational **Retrieval-Augmented Generation (RAG)** chatbot app, built using [Streamlit](https://streamlit.io/), LangChain, and Hugging Face models.  
It allows any user to:

- **Securely upload their own PDF, DOCX, or TXT documents** (never seen by other users)
- **Chat with an advanced AI assistant** that can answer questions using knowledge extracted from these documents
- **Ask follow-up questions** just like in a real conversation, the app will rephrase and clarify so context is always correct!
- **Try an instant “Demo mode”** with a built-in knowledge snippet (even with zero uploads)
- Download their chat history at any time

All of this runs in your web browser with zero setup, no local Python needed!

---

## Main Features

- **File-based RAG:** Drop in your own files, ask any question, and get document-grounded answers.
- **Instant Demo Mode:** Try out the chatbot even if you don’t upload any files, using the included [demo file](data/dune-summary.txt) as a sample knowledge base.
- **Multi-turn Conversation:** Follow-up questions automatically resolved using chat history (the AI understands your context and co-reference!).
- **Resource Limits and Sanitation:** Large files are rejected gracefully (max 2MB per file, 30,000 total characters/session).
- **Downloadable Chat History:** Export your transcript with a single click for notes or sharing.
- **Streamlit Cloud Ready:** Seamless, shareable demo deploys straight from GitHub!

---

## How It Works (Tech Stack)

- **Frontend/UI:** Streamlit chat UI for smooth user interaction.
- **Document Parsing:** LangChain community loaders for TXT, PDF, DOCX.
- **Chunking & Embedding:** Fast local embedding via `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Storage:** In-memory ChromaDB per user session for similarity search (never persisted on disk, privacy by design).
- **Language Model (Q&A):** Responses powered by a state-of-the-art, cloud-hosted LLM (e.g. [MoonshotAI Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct)) via the Hugging Face Inference API.
- **Session/state:** Strict isolation per user/session, no data cross-contamination, guaranteed.

---

## Demo Limitations & Notes

- **Speed:** Model loading and first LLM answer may take 10–30s, very normal for open RAG demos using real embeddings and cloud APIs.
- **File Limits:** Files must be .txt, .pdf, or .docx (max 2MB each, up to 30,000 total characters per session).
- **Session Lifetime:** Sessions are wiped on page reload, reset, or user tab close, for privacy, not for persistence.
- **No “real” cross-user chat:** Each user is isolated, great for privacy, but not for multiplayer chatrooms.
- **Cloud API:** All LLM completions are performed using Hugging Face’s inference API as configured; usage is subject to your account’s quota.

---
