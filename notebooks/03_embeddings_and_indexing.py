from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from chunking import docs

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

vectorstore.save_local("../data/vector_store/")