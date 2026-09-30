from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

docs = PyPDFDirectoryLoader("data/").load()
print(f"Loaded {len(docs)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
print(f"Stored {len(chunks)} chunks")