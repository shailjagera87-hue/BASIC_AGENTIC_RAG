import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()