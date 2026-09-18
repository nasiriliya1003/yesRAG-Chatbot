import os
import tempfile

import streamlit as st

from ingestion.ingestion import load_document, split_documents, build_vector_store
from rag.rag import get_relevant_chunks, format_chunks_as_context, build_prompt
from llm.llm import get_llm_response
from config import LLM_BASE_URL, LLM_MODEL_NAME, EMBEDDING_BASE_URL, EMBEDDING_MODEL_NAME

st.set_page_config(page_title="Chatbot WITH RAG", page_icon="📄")

st.title("📄 Chatbot WITH RAG")

st.info(
    "Upload a PDF or .txt file, then ask questions about it.\n\n"
    'If the answer isn\'t in the document, the chatbot will say:\n'
    '**"I don\'t know based on the uploaded document."**'
)

# -----------------------------------------------------------------
# Session state
# -----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -----------------------------------------------------------------
# Sidebar: Document upload + processing pipeline
# -----------------------------------------------------------------
with st.sidebar:
    st.header("📚 About this app")
    st.write(
        """
        This is **Project B** of the workshop:
        *Build a RAG Chatbot That Doesn't Hallucinate*.

        Flow:
        ```
        Document -> Chunks -> Embeddings -> ChromaDB
        Question -> Retrieve -> Prompt -> LLM -> Answer
        ```
        """
    )

    st.divider()
    st.caption("⚙️ Active providers (from config.py)")
    st.code(
        f"LLM        base_url: {LLM_BASE_URL}\n"
        f"LLM        model:    {LLM_MODEL_NAME}\n"
        f"Embeddings base_url: {EMBEDDING_BASE_URL}\n"
        f"Embeddings model:    {EMBEDDING_MODEL_NAME}",
        language=None,
    )

    st.divider()
    st.header("1️⃣ Upload your document")
    uploaded_file = st.file_uploader("Choose a PDF or .txt file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Process document", type="primary"):
            with st.spinner("Loading document..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                documents = load_document(tmp_path)  # Module 2
                st.success(f"Loaded {len(documents)} page(s).")

            with st.spinner("Splitting into chunks..."):
                chunks = split_documents(documents)  # Module 3
                st.success(f"Created {len(chunks)} chunk(s).")

                with st.expander("👀 Preview a few chunks"):
                    for i, chunk in enumerate(chunks[:3], start=1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(chunk.page_content)

            with st.spinner("Generating embeddings & storing in ChromaDB..."):
                vector_db = build_vector_store(chunks)  # Modules 4 + 5
                st.session_state.vector_db = vector_db
                st.success("Document is ready! You can now ask questions.")

# -----------------------------------------------------------------
# Main chat area
# -----------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about your document...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if st.session_state.vector_db is None:
            answer = "⚠️ Please upload and process a document first (see the sidebar)."
            st.markdown(answer)
        else:
            with st.spinner("Searching document..."):
                relevant_chunks = get_relevant_chunks(st.session_state.vector_db, user_question)  # Module 6
                context = format_chunks_as_context(relevant_chunks)

            with st.spinner("Generating grounded answer..."):
                prompt = build_prompt(context=context, question=user_question)  # Module 7
                answer = get_llm_response(prompt)  # Module 8

            st.markdown(answer)

            # Module 9: Display sources
            with st.expander("📌 Source chunks used for this answer"):
                for i, chunk in enumerate(relevant_chunks, start=1):
                    st.markdown(f"**Source Chunk {i}:**")
                    st.text(chunk.page_content)
                    st.caption(f"Page: {chunk.metadata.get('page', 'n/a')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
