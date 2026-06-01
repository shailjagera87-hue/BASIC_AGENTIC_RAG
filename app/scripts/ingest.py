# app/scripts/ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("🚀 Ingesting PDF and creating FAISS index...")

loader = PyPDFLoader("C:\\Users\\Shailja\\OneDrive\\Desktop\\BASIC_AGENTIC_RAG-main\\app\\data\\faiss\\research.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(chunks, emb)
db.save_local("app/data/faiss")

print("✅ Done")