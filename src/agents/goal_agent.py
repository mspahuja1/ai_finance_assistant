"""Goal Agent - Financial planning and budgeting"""
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

goal_logger = create_logger("goal_agent", "goal_agent.log")

def goal_agent(state: MessagesState):
    """Financial goal planning agent"""
    
    goal_logger.info("➡️ Entered Goal Agent")
    
    system = SystemMessage(content="""You are a Goal Planning Agent. 
Help users plan financial goals, create budgets, and develop timelines for achieving objectives.""")
    
    response = llm.invoke([system] + state["messages"])
    goal_logger.info("⬅️ Exiting Goal Agent")
    
    return {"messages": state["messages"] + [response]}