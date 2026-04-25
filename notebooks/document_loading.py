from langchain.document_loaders import PyPDFLoader

file_path = "../data/ArpitSaxenaDE4Y.pdf"

loader = PyPDFLoader(file_path)
documents = loader.load()

print(len(documents))
print(documents[0].page_content[:1000])