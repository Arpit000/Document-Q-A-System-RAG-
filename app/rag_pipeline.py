from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Load embeddings model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Load saved vector DB
db = FAISS.load_local(
    "../data/vector_store/",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 4})

# Load local LLM
llm = ChatOllama(
    model="llama3",
    temperature=0
)


def ask_question(question: str):
    """
    Main RAG function
    """

    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "No relevant information found."

    context = "\n\n".join(
        [doc.page_content for doc in relevant_docs]
    )

    prompt = f"""
Answer ONLY from the provided context.

Rules:
1. Do not guess
2. Do not use outside knowledge
3. If answer is not found, say:
'I could not find this in the documents.'

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content