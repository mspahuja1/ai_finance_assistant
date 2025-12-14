"""Router Agent - Routes queries to specialist agents"""
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage
from utils.llm_config import llm
from utils.logging_config import create_logger


router_logger = create_logger("router", "router.log")

def router_agent(state: MessagesState):
    """Route user queries to appropriate specialist agent"""
    
    router_logger.info("➡️ Entered Router")
    
    system = SystemMessage(content="""You are a routing agent. 
Your job is to read the user's message and decide which specialist agent should handle it.

Respond with ONLY one of the following labels:
- finance: General finance concepts and education
- portfolio: Investments, holdings, diversification
- market: Stock prices, market trends, sector analysis
- goal: Financial goals, retirement, budgeting
- news: Financial news and implications
- tax: Tax concepts and strategies
- reject: Non-financial queries

Choose the most appropriate specialist.""")
    
    user_message = state["messages"][-1]
    response = llm.invoke([system, user_message])
    route = response.content.strip().lower()
    
    router_logger.info(f"⬅️ Router selected: {route}")
    
    return {**state, "route": route}