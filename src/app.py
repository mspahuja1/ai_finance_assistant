"""Main Application Entry Point"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import app

# Export for Streamlit
__all__ = ['app']

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    # Test the app
    result = app.invoke(
        {"messages": [HumanMessage(content="What's the price of AAPL?")]},
        config={"configurable": {"thread_id": "test-1"}}
    )
    
    print("\n=== Response ===")
    print(result["messages"][-1].content)