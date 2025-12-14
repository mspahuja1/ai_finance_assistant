"""Portfolio Agent - Portfolio analysis and recommendations"""
import sys
import os

# Add src directory to path FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

# Now import packages
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage
from utils.llm_config import llm
from utils.logging_config import create_logger

portfolio_logger = create_logger("portfolio_agent", "portfolio_agent.log")

def portfolio_agent(state: MessagesState):
    """Portfolio analysis agent"""
    
    portfolio_logger.info("➡️ Entered Portfolio Agent")
    
    system = SystemMessage(content="""You are a Portfolio Analysis Agent. 
Analyze holdings, assess risk, evaluate diversification, and provide performance insights.""")
    
    response = llm.invoke([system] + state["messages"])
    portfolio_logger.info("⬅️ Exiting Portfolio Agent")
    
    return {"messages": state["messages"] + [response]}