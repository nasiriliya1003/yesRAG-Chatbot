import os
from dotenv import load_dotenv


load_dotenv()

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
LLM_TEMPERATURE = 0

# EMBEDDINGS
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/api/embeddings")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "ollama")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text:latest")

# RAG
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 3
CHROMA_PERSIST_DIRECTORY = "chroma_db"

