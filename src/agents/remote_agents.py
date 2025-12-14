"""Remote Agent Callers - For LangGraph to call microservices"""
import httpx
from langgraph.graph.message import MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from utils.logging_config import create_logger
import asyncio

logger = create_logger("remote_agents", "remote_agents.log")

# Service URLs
MARKET_SERVICE_URL = "http://localhost:8002"
NEWS_SERVICE_URL = "http://localhost:8003"

def convert_messages_to_dict(messages):
    """Convert LangChain messages to dict format for API"""
    result = []
    for msg in messages:
        msg_dict = {
            "type": msg.__class__.__name__.lower().replace("message", ""),
            "content": msg.content
        }
        result.append(msg_dict)
    return result

async def call_remote_service(url: str, state: MessagesState, service_name: str) -> dict:
    """Call a remote microservice"""
    try:
        logger.info(f"📡 Calling {service_name} service at {url}")
        
        # Convert messages to API format
        messages_dict = convert_messages_to_dict(state["messages"])
        
        # Prepare request
        request_data = {
            "messages": messages_dict,
            "thread_id": state.get("thread_id", "default")
        }
        
        # Call service with timeout
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{url}/process",
                json=request_data
            )
            
            if response.status_code != 200:
                logger.error(f"❌ {service_name} service returned {response.status_code}")
                return {
                    "messages": state["messages"] + [
                        AIMessage(content=f"Error: {service_name} service unavailable")
                    ]
                }
            
            data = response.json()
            logger.info(f"✅ {service_name} service responded successfully")
            
            # Add service response to messages
            return {
                "messages": state["messages"] + [
                    AIMessage(content=data['content'])
                ]
            }
            
    except httpx.TimeoutException:
        logger.error(f"⏱️ {service_name} service timeout")
        return {
            "messages": state["messages"] + [
                AIMessage(content=f"Error: {service_name} service timeout")
            ]
        }
    except Exception as e:
        logger.error(f"❌ Error calling {service_name} service: {str(e)}")
        return {
            "messages": state["messages"] + [
                AIMessage(content=f"Error: {str(e)}")
            ]
        }

# Remote agent functions for LangGraph
def market_agent_remote(state: MessagesState):
    """Market agent - calls remote microservice"""
    return asyncio.run(call_remote_service(MARKET_SERVICE_URL, state, "Market"))

def news_agent_remote(state: MessagesState):
    """News agent - calls remote microservice"""
    return asyncio.run(call_remote_service(NEWS_SERVICE_URL, state, "News"))

# Conditional functions
def market_agent_should_continue(state: MessagesState):
    """Market agent always ends after remote call"""
    return "end"

def news_agent_should_continue(state: MessagesState):
    """News agent always ends after remote call"""
    return "end"