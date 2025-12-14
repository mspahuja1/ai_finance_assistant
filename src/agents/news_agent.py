"""News Agent - Financial news synthesis with market context"""
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
from tools.stock_news import get_stock_news

news_logger = create_logger("news_agent", "news_agent.log")

# Bind tools to LLM
news_agent_tools = [get_market_data, get_stock_news]
llm_with_news_agent_tools = llm.bind_tools(news_agent_tools)
news_agent_tool_node = ToolNode(news_agent_tools)

def news_agent(state: MessagesState):
    """News synthesis agent with market data and news access"""
    
    news_logger.info("➡️ Entered News Agent")
    
    system = SystemMessage(content="""You are a News Synthesizer Agent with access to market data and news.

Your workflow:
1. Call get_market_data(symbol) for current stock performance
2. Call get_stock_news(symbol) for recent news articles
3. Synthesize: Connect news to stock movement
4. Provide investor insights

Always combine market context with news analysis.""")
    
    response = llm_with_news_agent_tools.invoke([system] + state["messages"])
    news_logger.info("⬅️ Exiting News Agent")
    
    return {"messages": state["messages"] + [response]}

def news_agent_should_continue(state: MessagesState):
    """Check if news agent needs to call tools"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "news_tools"
    return "end"