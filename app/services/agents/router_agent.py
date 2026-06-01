# app/services/agents/router_agent.py
from app.services.llm.llm_service import get_llm

llm = get_llm()

def router_agent(state):
    prompt = f"""
    Decide next step:
    Query: {state['query']}
    Context: {state.get('context', '')}

    Options:
    - cache
    - retrieve
    - generate

    Rules:
    If repeated/simple → cache
    If no context → retrieve
    Else → generate

    Return one word.
    """

    route = llm.invoke(prompt).content.strip().lower()

    return {**state, "route": route}