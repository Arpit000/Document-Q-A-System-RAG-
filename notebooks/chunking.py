from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_loading import documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

print(len(docs))
print(docs[0].page_content)