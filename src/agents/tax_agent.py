"""Tax Agent - Tax education and strategies"""
import sys
import os

# Add src directory to path FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage
from utils.llm_config import llm
from utils.logging_config import create_logger

tax_logger = create_logger("tax_agent", "tax_agent.log")

def tax_agent(state: MessagesState):
    """Tax education agent"""
    
    tax_logger.info("➡️ Entered Tax Agent")
    
    system = SystemMessage(content="""You are a Tax Education Agent. 
Explain tax concepts, rules, account types, and optimization strategies. 
Remind users you're not a tax professional and they should consult one for specific advice.""")
    
    response = llm.invoke([system] + state["messages"])
    tax_logger.info("⬅️ Exiting Tax Agent")
    
    return {"messages": state["messages"] + [response]}