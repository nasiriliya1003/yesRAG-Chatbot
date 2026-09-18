# Project B — Chatbot WITH RAG
### Part 2 of: *Build a RAG Chatbot That Doesn't Hallucinate*

## 📖 Project Overview

This project upgrades the simple Part 1 chatbot (`User -> LLM -> Answer`)
into a full **Retrieval-Augmented Generation (RAG)** pipeline.

```
User uploads document
        │
   Load document
        │
  Split into chunks
        │
 Generate embeddings
        │
 Store in ChromaDB
        │
   User asks question
        │
Retrieve Top-K chunks
        │
  Build grounded prompt
        │
        LLM
        │
Answer ONLY from context
        │
 Display Answer + Source
```

If the answer isn't in the document, the chatbot replies:

> "I don't know based on the uploaded document."

### 🔌 Works with any OpenAI-compatible provider

Both the LLM and the embedding model are configured through the
OpenAI-compatible API format, supported by OpenAI, local Ollama,
LM Studio, Groq, Together AI, OpenRouter, and Anthropic (chat only —
it has no embeddings API). **Switching providers only requires
editing `config.py` (or your `.env`)** — no changes anywhere else.

LLM and embeddings are configured *separately*, so you can mix
providers freely — e.g. Anthropic for chat + Ollama for embeddings:

| Provider | Use for | `BASE_URL` | `MODEL_NAME` |
|---|---|---|---|
| OpenAI | LLM + embeddings | `https://api.openai.com/v1` | `gpt-4o-mini` / `text-embedding-3-small` |
| Anthropic | LLM only | `https://api.anthropic.com/v1/` | `claude-sonnet-4-5` |
| Local Ollama | LLM + embeddings | `http://localhost:11434/v1` | `llama3.1` / `nomic-embed-text` |

## 📂 Folder Structure

```
rag-chatbot/
│
├── app.py              # Streamlit UI + orchestrates the full pipeline
├── config.py            # All settings in one place
├── ingestion.py          # Load -> Chunk -> Embed -> Store (Modules 2-5)
├── rag.py                # Retrieve -> Build prompt (Modules 6-7)
├── llm.py                # Talks to Gemini (Module 8)
├── requirements.txt
├── data/                 # Optional: put sample.pdf here to test with
└── README.md
```

This is the **exact same structure** as Project A — `ingestion.py` and
`rag.py` are now fully implemented instead of placeholders, and
nothing was moved or renamed.

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```bash
# --- LLM: pick ONE ---
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o-mini

# LLM_BASE_URL=https://api.anthropic.com/v1/
# LLM_API_KEY=sk-ant-...
# LLM_MODEL_NAME=claude-sonnet-4-5

# LLM_BASE_URL=http://localhost:11434/v1
# LLM_API_KEY=ollama
# LLM_MODEL_NAME=llama3.1

# --- Embeddings: pick ONE (Anthropic has no embeddings API) ---
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL_NAME=text-embedding-3-small

# EMBEDDING_BASE_URL=http://localhost:11434/v1
# EMBEDDING_API_KEY=ollama
# EMBEDDING_MODEL_NAME=nomic-embed-text
```

Mix and match freely — e.g. Anthropic for `LLM_*` and Ollama for
`EMBEDDING_*`. No code changes needed, just update `.env` and restart.

## ▶️ Running the App

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

**Steps in the app:**
1. Upload a `.pdf` or `.txt` file in the sidebar
2. Click **"Process document"** and wait for it to be chunked, embedded, and stored
3. Ask a question in the chat box
4. Read the grounded answer, and expand **"Source chunks"** to see exactly where it came from

## 🎯 Learning Objectives

- Explain why documents need to be split into chunks
- Explain what an embedding is and how semantic similarity works
- Explain why we store embeddings in a vector database
- Explain Top-K retrieval and similarity search
- Build a prompt that grounds an LLM's answer in retrieved context
- Explain the relationship between temperature and hallucination
- Build a chatbot that transparently shows its sources

## 🧩 Workshop Modules Covered Here

| Module | Topic | File |
|---|---|---|
| 2 | Loading Documents | `ingestion.py` |
| 3 | Chunking | `ingestion.py` |
| 4 | Embeddings | `ingestion.py` |
| 5 | Vector Database | `ingestion.py` |
| 6 | Retriever | `rag.py` |
| 7 | Prompt Builder | `rag.py` |
| 8 | Generation | `llm.py` |
| 9 | Display Sources | `app.py` |

## 🐛 Common Errors & Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `401 Unauthorized` / auth error | `.env` missing, wrong key, or wrong provider block uncommented | Confirm `.env` has matching `LLM_*` and `EMBEDDING_*` values for ONE provider each |
| `Connection refused` (Ollama) | Ollama isn't running, or wrong port | Run `ollama serve`, pull the models (`ollama pull llama3.1`, `ollama pull nomic-embed-text`), and confirm URLs end in `/v1` |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` inside your virtual environment |
| Chatbot says "I don't know..." for everything | Document didn't process correctly, or question is genuinely outside the document | Re-check "Process document" succeeded; try a question you know is answered in the text |
| Embeddings error using Anthropic | Anthropic has no embeddings API | Set `EMBEDDING_BASE_URL`/`EMBEDDING_MODEL_NAME` to OpenAI or Ollama instead — Anthropic can still be used for `LLM_*` |
| Slow processing on large PDFs | Many chunks/embeddings being generated | Try a smaller PDF first, or increase `CHUNK_SIZE` in `config.py` |
| Old answers appear after uploading a new document | ChromaDB persisted the previous document | Delete the local `chroma_db/` folder before processing a new document |

## 🚀 Next Steps

- Support multiple document uploads at once
- Conversation-aware retrieval (use chat history to refine searches)
- Swap ChromaDB for a production vector database (e.g. Pinecone, Weaviate)
- Add evaluation metrics to measure groundedness and answer quality
- Wrap the pipeline behind an API (e.g. FastAPI) for use outside Streamlit
