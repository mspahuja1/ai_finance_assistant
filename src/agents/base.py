"""Base Agent Class"""
from langraph.graph.message import MessagesState
from langchain_core.messages import AIMessage

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, logger):
        self.name = name
        self.logger = logger
    
    def log_entry(self, state: MessagesState):
        self.logger.info(f"➡️ Entered {self.name} agent")
    
    def log_exit(self, response):
        self.logger.info(f"⬅️ Exiting {self.name} agent")
    
    def should_continue(self, state: MessagesState) -> str:
        """Check if agent should continue to tools or end"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"