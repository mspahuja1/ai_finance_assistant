"""News Agent Microservice - Called by LangGraph"""
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
from tools.stock_news import get_stock_news

# Initialize
app = FastAPI(
    title="News Agent Microservice",
    description="Financial news synthesis with market context",
    version="1.0.0"
)
logger = create_logger("news_service", "news_service.log")

# Bind tools
llm_with_tools = llm.bind_tools([get_market_data, get_stock_news])

# Request/Response Models
class Message(BaseModel):
    type: str
    content: str

class ServiceRequest(BaseModel):
    messages: List[Message]
    thread_id: str = "default"

class ServiceResponse(BaseModel):
    content: str
    tool_calls_made: List[Dict[str, Any]] = []
    thread_id: str

@app.post("/process", response_model=ServiceResponse)
async def process_news_query(request: ServiceRequest):
    """
    Process news query with market data + news tools
    
    Called by LangGraph orchestrator
    """
    try:
        logger.info(f"📰 News service request from thread: {request.thread_id}")
        
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
        system = SystemMessage(content="""You are a News Synthesizer Agent with access to market data and news.

Your workflow:
1. Call get_market_data(symbol) for current stock performance
2. Call get_stock_news(symbol) for recent news articles
3. Synthesize: Connect news to stock movement
4. Provide investor insights

Always combine market context with news analysis.""")
        
        # First LLM call
        response = llm_with_tools.invoke([system] + messages)
        
        tool_calls_info = []
        max_iterations = 3  # Prevent infinite loops
        iteration = 0
        
        # Handle multiple tool call rounds
        while hasattr(response, 'tool_calls') and response.tool_calls and iteration < max_iterations:
            logger.info(f"🔧 Processing {len(response.tool_calls)} tool calls (iteration {iteration + 1})")
            
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                symbol = tool_call['args']['symbol']
                logger.info(f"  → Calling {tool_name}({symbol})")
                
                # Execute appropriate tool
                if tool_name == "get_market_data":
                    tool_result = get_market_data.invoke({"symbol": symbol})
                elif tool_name == "get_stock_news":
                    tool_result = get_stock_news.invoke({"symbol": symbol})
                else:
                    tool_result = f"Unknown tool: {tool_name}"
                
                tool_calls_info.append({
                    "tool": tool_name,
                    "symbol": symbol,
                    "result_preview": tool_result[:100] + "..."
                })
                
                # Add to conversation
                messages.append(response)
                messages.append(HumanMessage(content=tool_result))
            
            # Get next response
            response = llm_with_tools.invoke([system] + messages)
            iteration += 1
        
        logger.info(f"✅ News service responded to thread: {request.thread_id}")
        
        return ServiceResponse(
            content=response.content,
            tool_calls_made=tool_calls_info,
            thread_id=request.thread_id
        )
        
    except Exception as e:
        logger.error(f"❌ News service error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "news",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {
        "service": "News Agent Microservice",
        "version": "1.0.0",
        "description": "Financial news synthesis with market context",
        "endpoints": {
            "process": "POST /process - Process news queries",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    logger.info("🚀 Starting News Agent Microservice on port 8003")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )