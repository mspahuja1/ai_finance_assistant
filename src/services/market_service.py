"""Market Agent Microservice - Called by LangGraph"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import sys
import os
import uvicorn

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.llm_config import llm
from utils.logging_config import create_logger
from tools.market_data import get_market_data

# Initialize
app = FastAPI(
    title="Market Agent Microservice",
    description="Market analysis with real-time data",
    version="1.0.0"
)
logger = create_logger("market_service", "market_service.log")

# Bind tools
llm_with_tools = llm.bind_tools([get_market_data])

# Request/Response Models
class Message(BaseModel):
    type: str  # "human", "ai", "system"
    content: str

class ServiceRequest(BaseModel):
    messages: List[Message]
    thread_id: str = "default"

class ServiceResponse(BaseModel):
    content: str
    tool_calls_made: List[Dict[str, Any]] = []
    thread_id: str

@app.post("/process", response_model=ServiceResponse)
async def process_market_query(request: ServiceRequest):
    """
    Process market query with tool support
    
    Called by LangGraph orchestrator
    """
    try:
        logger.info(f"📥 Market service request from thread: {request.thread_id}")
        
        # Convert to LangChain messages
        messages = []
        for msg in request.messages:
            if msg.type == "human":
                messages.append(HumanMessage(content=msg.content))
            elif msg.type == "ai":
                messages.append(AIMessage(content=msg.content))
            elif msg.type == "system":
                messages.append(SystemMessage(content=msg.content))
        
        # Add system prompt
        system = SystemMessage(content="""You are a Market Analysis Agent with access to real-time market data.

When users ask about stocks or markets:
1. Use get_market_data(symbol) to fetch current prices and trends
2. Analyze the data and provide insights
3. Explain what the numbers mean for investors

Always fetch data before providing market analysis.""")
        
        # First LLM call
        response = llm_with_tools.invoke([system] + messages)
        
        tool_calls_info = []
        
        # Handle tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"🔧 Processing {len(response.tool_calls)} tool calls")
            
            for tool_call in response.tool_calls:
                symbol = tool_call['args']['symbol']
                logger.info(f"  → Calling get_market_data({symbol})")
                
                # Execute tool
                tool_result = get_market_data.invoke({"symbol": symbol})
                
                tool_calls_info.append({
                    "tool": "get_market_data",
                    "symbol": symbol,
                    "result_preview": tool_result[:100] + "..."
                })
                
                # Add tool result to conversation
                messages.append(response)
                messages.append(HumanMessage(content=tool_result))
            
            # Get final response after tools
            response = llm_with_tools.invoke([system] + messages)
        
        logger.info(f"✅ Market service responded to thread: {request.thread_id}")
        
        return ServiceResponse(
            content=response.content,
            tool_calls_made=tool_calls_info,
            thread_id=request.thread_id
        )
        
    except Exception as e:
        logger.error(f"❌ Market service error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "market",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Service info"""
    return {
        "service": "Market Agent Microservice",
        "version": "1.0.0",
        "description": "Real-time market data and analysis",
        "endpoints": {
            "process": "POST /process - Process market queries",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Market Agent Microservice on port 8002")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )