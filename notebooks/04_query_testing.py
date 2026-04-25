from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = FAISS.load_local(
    "../data/vector_store/",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

query = "Which all organizations has Arpit Saxena worked with?"

relevant_docs = retriever.invoke(query)

# for doc in relevant_docs:
#     print(doc.page_content[:500])
#     print("----------")

llm = ChatOllama(
    model="llama3",
    temperature=0
)

context = "\n".join(
    [doc.page_content for doc in relevant_docs]
)

prompt = f"""
Answer ONLY from the provided context.

If the answer is not present, say:
'I could not find this in the documents.'

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print(response.content)