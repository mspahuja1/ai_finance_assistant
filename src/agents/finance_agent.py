"""Finance Agent - General financial education"""
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

finance_logger = create_logger("finance_agent", "finance_agent.log")

def finance_agent(state: MessagesState):
    """Finance education and concepts agent"""
    
    finance_logger.info("➡️ Entered Finance Agent")
    
    system = SystemMessage(content="""You are a Finance Q&A Agent. 
Explain financial concepts clearly and simply. Help users understand investing, 
personal finance, and money management.""")
    
    response = llm.invoke([system] + state["messages"])
    finance_logger.info("⬅️ Exiting Finance Agent")
    
    return {"messages": state["messages"] + [response]}