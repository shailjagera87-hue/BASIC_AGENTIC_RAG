# app/services/rag/generator.py
from app.services.llm.llm_service import get_llm

llm = get_llm()

def generator_node(state):
    prompt = f"""
    Answer using context:

    {state.get("context", "")}

    Question:
    {state["query"]}
    """

    ans = llm.invoke(prompt).content
    return {**state, "answer": ans}