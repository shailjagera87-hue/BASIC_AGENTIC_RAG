# app/services/llm/llm_service.py

from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings


def get_llm():
    return ChatGoogleGenerativeAI(
        model=settings.MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY
    )