from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


from config import(
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIRECTORY,
    )


# Load documents
def load_document(file_path: str):
    "Load documents"

    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Only .pdf and .txt can be read")
    return loader.load()

def split_documents(documents):
    "Split the documents"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)

def get_embedding():
    "set and get the embedding"

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=EMBEDDING_BASE_URL if EMBEDDING_BASE_URL.endswith("/v1") else "http://localhost:11434/v1",
        openai_api_key=EMBEDDING_API_KEY,
        check_embedding_ctx_length=False,
    )

def build_vector_store(chunks):
    "Build vector store"

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding(),
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )
