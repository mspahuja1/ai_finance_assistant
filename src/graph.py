"""LangGraph Orchestration"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from agents.router import router_agent
from agents.finance_agent import finance_agent
from agents.portfolio_agent import portfolio_agent
from agents.market_agent import market_agent, market_agent_should_continue, market_tool_node
from agents.goal_agent import goal_agent
from agents.news_agent import news_agent, news_agent_should_continue, news_agent_tool_node
from agents.tax_agent import tax_agent

# User query node
def user_query_node(state: MessagesState):
    return {"messages": state["messages"]}

# Reject agent
def reject_agent(state: MessagesState):
    return {
        "messages": state["messages"] + [
            AIMessage(content="I can help with financial topics only. Try asking about investing, markets, taxes, or budgeting.")
        ]
    }

# Build graph
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("user_query", user_query_node)
graph.add_node("router", router_agent)
graph.add_node("finance", finance_agent)
graph.add_node("portfolio", portfolio_agent)
graph.add_node("market", market_agent)
graph.add_node("market_tools", market_tool_node)
graph.add_node("goal", goal_agent)
graph.add_node("news", news_agent)
graph.add_node("news_tools", news_agent_tool_node)
graph.add_node("tax", tax_agent)
graph.add_node("reject", reject_agent)

# Edges
graph.add_edge("user_query", "router")

# Router to agents
graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "finance": "finance",
        "portfolio": "portfolio",
        "market": "market",
        "goal": "goal",
        "news": "news",
        "tax": "tax",
        "reject": "reject",
    },
)

# Market agent tool flow
graph.add_conditional_edges(
    "market",
    market_agent_should_continue,
    {
        "tools": "market_tools",
        "end": END,
    },
)
graph.add_edge("market_tools", "market")

# News agent tool flow
graph.add_conditional_edges(
    "news",
    news_agent_should_continue,
    {
        "news_tools": "news_tools",
        "end": END,
    },
)
graph.add_edge("news_tools", "news")

# End edges
graph.add_edge("finance", END)
graph.add_edge("portfolio", END)
graph.add_edge("goal", END)
graph.add_edge("tax", END)
graph.add_edge("reject", END)

# Entry point
graph.set_entry_point("user_query")

# Compile with checkpointer
conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = graph.compile(checkpointer=checkpointer)