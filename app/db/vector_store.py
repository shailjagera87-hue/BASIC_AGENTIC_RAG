# app/db/vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

db = None

def load_vector_store():
    global db

    if db:
        return db

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    )

    db = FAISS.load_local(
        "app/data/faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db