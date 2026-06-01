from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag.graph import build_graph

router = APIRouter()
graph = build_graph()

@router.post("/chat")
def chat(req: dict):
    try:
        print("📩 Incoming Raw:", req)

        query = req.get("query")

        if not query:
            raise HTTPException(status_code=400, detail="Query missing")

        state = {
            "query": query,
            "context": "",
            "answer": "",
            "cached": False,
            "route": ""
        }

        result = graph.invoke(state)

        return {
            "answer": result.get("answer", "No response generated"),
            "cached": result.get("cached", False)
        }

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))