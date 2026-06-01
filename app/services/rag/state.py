# app/services/rag/state.py
from typing import TypedDict, Optional

class RAGState(TypedDict):
    query: str
    context: Optional[str]
    answer: Optional[str]
    cached: Optional[bool]
    route: Optional[str]