"""Market Agent - Stock prices and market analysis with MCP tools"""
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from utils.llm_config import llm
from utils.logging_config import create_logger
from tools.market_data import get_market_data

market_logger = create_logger("market_agent", "market_agent.log")

# Bind tools to LLM
market_tools = [get_market_data]
llm_with_market_tools = llm.bind_tools(market_tools)
market_tool_node = ToolNode(market_tools)

def market_agent(state: MessagesState):
    """Market analysis agent with real-time data access"""
    
    market_logger.info("➡️ Entered Market Agent")
    
    system = SystemMessage(content="""You are a Market Analysis Agent with access to real-time market data.

When users ask about stocks or markets:
1. Use get_market_data(symbol) to fetch current prices and trends
2. Analyze the data and provide insights
3. Explain what the numbers mean for investors

Always fetch data before providing market analysis.""")
    
    response = llm_with_market_tools.invoke([system] + state["messages"])
    market_logger.info("⬅️ Exiting Market Agent")
    
    return {"messages": state["messages"] + [response]}

def market_agent_should_continue(state: MessagesState):
    """Check if market agent needs to call tools"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"