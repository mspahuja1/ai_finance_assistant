"""Standalone Market Agent Microservice"""
from fastapi import FastAPI
from pydantic import BaseModel
from market_agent import market_agent
from langchain_core.messages import HumanMessage

app = FastAPI()

class QueryRequest(BaseModel):
    message: str
    thread_id: str = "default"

@app.post("/market")
async def market_query(request: QueryRequest):
    """Market agent endpoint"""
    state = {
        "messages": [HumanMessage(content=request.message)]
    }
    
    result = market_agent(state)
    
    return {
        "response": result["messages"][-1].content,
        "thread_id": request.thread_id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)