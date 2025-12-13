# agents_1.py

from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------
# Initialize Gemini LLM
# ---------------------------------------------------------

import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    api_key=api_key,
)


# ---------------------------------------------------------
# Logging Initialization
# ---------------------------------------------------------
import logging
import os
from logging.handlers import RotatingFileHandler

# Absolute path to your logs directory
LOG_DIR = "/Users/mandeep/myprojects/ai_finance_assistant/logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5),
        logging.StreamHandler()
    ]
)

"""
finance_agent_logger = logging.getLogger("finance_agent")
portfolio_agent_logger = logging.getLogger("portfolio_agent")
market_agent_logger = logging.getLogger("market_agent")
goal_agent_logger = logging.getLogger("goal_agent")
news_agent_logger = logging.getLogger("news_agent")
tax_agent_logger = logging.getLogger("tax_agent")
router_logger = logging.getLogger("router")   # ✅ 
"""




# ---------------------------------------------------------
# Define logs for each agent
# ---------------------------------------------------------
def create_agent_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, filename),
        maxBytes=5_000_000,
        backupCount=5
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Avoid duplicate handlers on Streamlit reruns
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

finance_agent_logger = create_agent_logger("finance_agent", "finance_agent.log")
portfolio_agent_logger = create_agent_logger("portfolio_agent", "portfolio_agent.log")
market_agent_logger = create_agent_logger("market_agent", "market_agent.log")
goal_agent_logger = create_agent_logger("goal_agent", "goal_agent.log")
news_agent_logger = create_agent_logger("news_agent", "news_agent.log")
tax_agent_logger = create_agent_logger("tax_agent", "tax_agent.log")
router_logger = create_agent_logger("router", "router.log")

# ---------------------------------------------------------
# User Query Node
# ---------------------------------------------------------
def user_query_node(state: MessagesState):
    # The last message is the new user message
    return {"messages": state["messages"]}

# ---------------------------------------------------------
# Router Node
# ---------------------------------------------------------

router_logger = logging.getLogger("router")

def workflow_router(state: MessagesState):
    router_logger.info("➡️ Entered LLM Router with state: %s", state)

    system = SystemMessage(content=
        """You are a routing agent. 
        Your job is to read the user's message and decide which specialist agent should handle it.

        Respond with ONLY one of the following labels:
        - finance
        - portfolio
        - market
        - goal
        - news
        - tax

        Rules:
        - If the user asks about general finance concepts → finance
        - If the user asks about investments, holdings, diversification → portfolio
        - If the user asks about markets, sectors, macro trends → market
        - If the user asks about goals, retirement, budgeting → goal
        - If the user asks about financial news → news
        - If the user asks about taxes → tax
        - Anything unrelated to finance → reject.
        """
    )

    user_message = state["messages"][-1]

    response = llm.invoke([system, user_message])
    route = response.content.strip().lower()

    router_logger.info("⬅️ LLM Router selected route: %s", route)

    return {**state, "route": route}

# ---------------------------------------------------------
# Agent Nodes (each calls Gemini with a system prompt)
# ---------------------------------------------------------

def reject_agent(state: MessagesState):
    return {
        "messages": state["messages"] + [
            AIMessage(content="I can help with financial topics only. Try asking me about investing, markets, taxes, budgeting, or financial planning.")
        ]
    }

def finance_agent(state: MessagesState):
    finance_agent_logger.info("➡️ Entered FINANCE agent with state: %s", state)
    system = SystemMessage(content="You are a Finance Q&A Agent. Explain financial concepts clearly and simply.")
    response = llm.invoke([system] + state["messages"])
    finance_agent_logger.info("⬅️ Exiting FINANCE agent with response: %s", response)
    return {"messages": state["messages"] + [response]}

#portfolio_agent_logger = logging.getLogger("portfolio_agent")

def portfolio_agent(state: MessagesState):
    portfolio_agent_logger.info("➡️ Entered PORTFOLIO agent with state: %s", state)
    system = SystemMessage(content="You are a Portfolio Analysis Agent. Analyze holdings, risk, diversification, and performance.")
    response = llm.invoke([system] + state["messages"])
    portfolio_agent_logger.info("⬅️ Exiting PORTFOLIO agent with response: %s", response)
    return {"messages": state["messages"] + [response]}


# ---------------------------------------------------------

#market_agent_logger = logging.getLogger("market_agent")

def market_agent(state):
    market_agent_logger.info("➡️ Entered MARKET agent with state: %s", state)
    system = SystemMessage(content="You are a Market Analysis Agent...")
    response = llm.invoke([system] + state["messages"])
    market_agent_logger.info("⬅️ Exiting MARKET agent with response: %s", response)
    return {"messages": state["messages"] + [response]}


# ---------------------------------------------------------

#goal_agent_logger = logging.getLogger("goal_agent")

def goal_agent(state: MessagesState):
    goal_agent_logger.info("➡️ Entered GOAL agent with state: %s", state)
    system = SystemMessage(content="You are a Goal Planning Agent. Help users plan financial goals, budgets, and timelines.")
    response = llm.invoke([system] + state["messages"])
    goal_agent_logger.info("⬅️ Exiting GOAL agent with response: %s", response)
    return {"messages": state["messages"] + [response]}

#news_agent_logger = logging.getLogger("news_agent")

def news_agent(state: MessagesState):
    news_agent_logger.info("➡️ Entered NEWS agent with state: %s", state)
    system = SystemMessage(content="You are a News Synthesizer Agent. Summarize financial news and explain its implications.")
    response = llm.invoke([system] + state["messages"])
    news_agent_logger.info("⬅️ Exiting NEWS agent with response: %s", response)
    return {"messages": state["messages"] + [response]}

#tax_agent_logger = logging.getLogger("tax_agent")

def tax_agent(state: MessagesState):
    tax_agent_logger.info("➡️ Entered TAX agent with state: %s", state)
    system = SystemMessage(content="You are a Tax Education Agent. Explain tax concepts, rules, and account types.")
    response = llm.invoke([system] + state["messages"])
    tax_agent_logger.info("⬅️ Exiting TAX agent with response: %s", response)
    return {"messages": state["messages"] + [response]}

#router_logger = logging.getLogger("router")
# ---------------------------------------------------------
# Build Graph
# ---------------------------------------------------------
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("reject", reject_agent)
graph.add_node("user_query", user_query_node)
graph.add_node("router", workflow_router)
graph.add_node("finance", finance_agent)
graph.add_node("portfolio", portfolio_agent)
graph.add_node("market", market_agent)
graph.add_node("goal", goal_agent)
graph.add_node("news", news_agent)
graph.add_node("tax", tax_agent)

# Edges
graph.add_edge("user_query", "router")

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

# End edges
graph.add_edge("finance", END)
graph.add_edge("portfolio", END)
graph.add_edge("market", END)
graph.add_edge("goal", END)
graph.add_edge("news", END)
graph.add_edge("tax", END)
graph.add_edge("reject", END)

# Entry point
graph.set_entry_point("user_query")

# ---------------------------------------------------------
# Compile with SQLite Checkpointer (LangGraph 1.0.4 + sqlite 3.0.1)
# ---------------------------------------------------------

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = graph.compile(checkpointer=checkpointer)

######################################################

# Testing, should be commented once test is done
"""
if __name__ == "__main__":
    result = app.invoke({"messages": [HumanMessage(content="Give me a market update") ]})
    print(result)
"""
######################################################