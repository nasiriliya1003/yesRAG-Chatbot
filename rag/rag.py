from config import TOP_K

def get_relevant_chunks(vector_db, question: str):
    "Search for relevant documents and retrieve them"

    retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})
    return retriever.invoke(question)

def format_chunks_as_context(chunks) -> str:
    "for all retrived chunks"
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk.page_content.split())
        lines.append(f'{i}. "{text}"')
    print(lines)
    return "\n".join(lines)

# Prompt builder
PROMPT_TEMPLATE = """You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is missing, reply:
"I don't know based on the uploaded document."

Context:
{context}

Question:
{question}
"""
def build_prompt(context: str, question: str) -> str:
    "build system prompt"

    return PROMPT_TEMPLATE.format(context=context, question=question)