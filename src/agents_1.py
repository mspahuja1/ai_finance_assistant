from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import Annotated
from langgraph.prebuilt import ToolNode

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

# ---------------------------------------------------------
# Define logs for each agent
# ---------------------------------------------------------
def create_agent_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, filename),
        maxBytes=1_000_000,
        backupCount=1
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
# Tool Definition for Market Data
# ---------------------------------------------------------
# ---------------------------------------------------------
# Tool Definition for Market Data with MCP Logging
# ---------------------------------------------------------
import yfinance as yf
import json
import time

# Create MCP-specific logger
mcp_logger = create_agent_logger("mcp_transactions", "mcp_transactions.log")

# ---------------------------------------------------------
# Import MCP Client
# ---------------------------------------------------------
import sys
import os

# Add the mcp directory to the path
mcp_dir = os.path.join(os.path.dirname(__file__), "mcp")
sys.path.insert(0, mcp_dir)

from mcp_client import MCPClient

# Initialize MCP Client
MCP_SERVER_PATH = os.path.join(mcp_dir, "server.py")
mcp_client = MCPClient(MCP_SERVER_PATH)

# ---------------------------------------------------------
# Tool Definition Using MCP Client
# ---------------------------------------------------------

@tool
def get_market_data(symbol: Annotated[str, "Stock ticker symbol (e.g., AAPL, GOOGL)"]) -> str:
    """Fetch 1-year market history and current data for a stock symbol via MCP"""
    
    # Log the incoming request
    request_id = f"mcp-{int(time.time() * 1000)}"
    mcp_logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    mcp_logger.info(f"📥 MCP REQUEST [{request_id}]")
    mcp_logger.info(f"Tool: get_market_data")
    mcp_logger.info(f"Input Parameters: {{'symbol': '{symbol}'}}")
    mcp_logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Call MCP server via client
        mcp_logger.info(f"🔄 Calling MCP server for {symbol}...")
        response = mcp_client.call_tool("get_market_data", {"symbol": symbol})
        
        # Debug log
        mcp_logger.info(f"🔍 RAW MCP RESPONSE: {json.dumps(response, indent=2)}")
        
        if "error" in response:
            error_msg = f"MCP Error: {response['error']}"
            mcp_logger.error(f"❌ {error_msg}")
            return error_msg
        
        # Extract result from MCP response
        if "result" in response:
            result_obj = response["result"]
            
            # Check if there's an error flag
            if result_obj.get("isError"):
                error_text = result_obj.get("content", [{}])[0].get("text", "Unknown error")
                mcp_logger.error(f"❌ MCP returned error: {error_text}")
                return f"Error: {error_text}"
            
            # Extract the content
            content = result_obj.get("content", [])
            if content and len(content) > 0:
                # Get the text from the first content item
                text_content = content[0].get("text", "")
                mcp_logger.info(f"🔍 TEXT CONTENT (first 200 chars): {text_content[:200]}")
                
                # Parse the JSON data
                market_data = json.loads(text_content)
                
                if not market_data or len(market_data) == 0:
                    return f"No data found for {symbol}"
                
                # Get latest data point
                latest = market_data[-1]
                
                # Calculate metrics from all data points
                high_prices = [float(d.get('High', 0)) for d in market_data if 'High' in d]
                low_prices = [float(d.get('Low', 0)) for d in market_data if 'Low' in d]
                volumes = [float(d.get('Volume', 0)) for d in market_data if 'Volume' in d]
                
                year_high = max(high_prices) if high_prices else 0
                year_low = min(low_prices) if low_prices else 0
                avg_volume = sum(volumes) / len(volumes) if volumes else 0
                
                # Format the result
                result = f"""Market Data for {symbol}:
📊 Current Price: ${float(latest.get('Close', 0)):.2f}
📈 Day High: ${float(latest.get('High', 0)):.2f}
📉 Day Low: ${float(latest.get('Low', 0)):.2f}
📦 Volume: {float(latest.get('Volume', 0)):,.0f}
🎯 52-Week High: ${year_high:.2f}
🎯 52-Week Low: ${year_low:.2f}
📊 Avg Volume: {avg_volume:,.0f}
📅 Last Updated: {latest.get('Date', 'N/A')}
"""
                
                elapsed = time.time() - start_time
                mcp_logger.info(f"✅ MCP call successful")
                mcp_logger.info(f"Records processed: {len(market_data)}")
                mcp_logger.info(f"📤 MCP RESPONSE [{request_id}]")
                mcp_logger.info(f"Status: SUCCESS")
                mcp_logger.info(f"Execution time: {elapsed:.3f}s")
                mcp_logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                
                return result
        
        mcp_logger.error(f"🔍 Unexpected response structure")
        mcp_logger.error(f"🔍 Full response: {json.dumps(response, indent=2)}")
        return "Unexpected response format from MCP server"
        
    except json.JSONDecodeError as e:
        elapsed = time.time() - start_time
        mcp_logger.error(f"❌ JSON DECODE ERROR [{request_id}]")
        mcp_logger.error(f"Error: {str(e)}")
        mcp_logger.error(f"Execution time: {elapsed:.3f}s")
        mcp_logger.error(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return f"Error parsing data: {str(e)}"
    except Exception as e:
        elapsed = time.time() - start_time
        mcp_logger.error(f"❌ MCP ERROR [{request_id}]")
        mcp_logger.error(f"Error: {str(e)}")
        mcp_logger.error(f"Traceback: ", exc_info=True)
        mcp_logger.error(f"Execution time: {elapsed:.3f}s")
        mcp_logger.error(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return f"Error: {str(e)}"


# ---------------------------------------------------------
# Stock News Tool
# ---------------------------------------------------------
@tool
def get_stock_news(symbol: Annotated[str, "Stock ticker symbol (e.g., AAPL, RIV)"]) -> str:
    """Fetch recent news articles for a stock symbol"""
    
    request_id = f"news-{int(time.time() * 1000)}"
    news_agent_logger.info(f"📰 NEWS REQUEST [{request_id}] - {symbol}")
    start_time = time.time()
    
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return f"No recent news found for {symbol}"
        
        result = f"📰 Recent News for {symbol}:\n\n"
        
        for i, article in enumerate(news[:5], 1):
            title = article.get('title', 'No title')
            publisher = article.get('publisher', 'Unknown')
            link = article.get('link', '#')
            timestamp = article.get('providerPublishTime', 0)
            
            from datetime import datetime
            date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
            
            result += f"{i}. {title}\n"
            result += f"   Publisher: {publisher}\n"
            result += f"   Date: {date}\n"
            result += f"   Link: {link}\n\n"
        
        elapsed = time.time() - start_time
        news_agent_logger.info(f"✅ SUCCESS [{request_id}] - {elapsed:.3f}s - {len(news)} articles")
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        news_agent_logger.error(f"❌ ERROR [{request_id}] - {str(e)}")
        return f"Error: {str(e)}"



# Create tools list
tools = [get_market_data]

# ---------------------------------------------------------
# Bind Tools to LLM (AFTER llm is defined)
# ---------------------------------------------------------
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------
# News Agent Tools Setup
# ---------------------------------------------------------
news_agent_tools = [get_market_data, get_stock_news]
llm_with_news_agent_tools = llm.bind_tools(news_agent_tools)
news_agent_tool_node = ToolNode(news_agent_tools)

def news_agent_should_continue(state: MessagesState):
    """Determine if news agent needs to call tools or end"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "news_tools"
    return "end"



# ---------------------------------------------------------
# Tool Execution Node
# ---------------------------------------------------------
from langgraph.prebuilt import ToolNode

# Wrap the tool node to add logging
def logged_tool_node(state: MessagesState):
    """Tool node wrapper that logs all tool executions"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            mcp_logger.info(f"🔧 TOOL INVOCATION DETECTED")
            mcp_logger.info(f"Tool name: {tool_call.get('name', 'unknown')}")
            mcp_logger.info(f"Tool ID: {tool_call.get('id', 'unknown')}")
            mcp_logger.info(f"Arguments: {json.dumps(tool_call.get('args', {}), indent=2)}")
    
    # Execute the actual tool
    result = tool_node.invoke(state)
    
    # Log the tool execution result
    if result and "messages" in result:
        for msg in result["messages"]:
            if hasattr(msg, 'content'):
                mcp_logger.info(f"🔧 TOOL EXECUTION COMPLETE")
                mcp_logger.info(f"Result preview: {str(msg.content)[:200]}...")
    
    return result

# Create the tool node
tool_node = ToolNode(tools)


# Define should_continue function BEFORE using it
def should_continue(state: MessagesState):
    """Determine if we need to call tools or end"""
    last_message = state["messages"][-1]
    
    # If the LLM makes a tool call, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise end
    return "end"

# Wrap the tool node to add logging
def logged_tool_node(state: MessagesState):
    """Tool node wrapper that logs all tool executions"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            mcp_logger.info(f"🔧 TOOL INVOCATION DETECTED")
            mcp_logger.info(f"Tool name: {tool_call.get('name', 'unknown')}")
            mcp_logger.info(f"Tool ID: {tool_call.get('id', 'unknown')}")
            mcp_logger.info(f"Arguments: {json.dumps(tool_call.get('args', {}), indent=2)}")
    
    # Execute the actual tool
    result = tool_node.invoke(state)
    
    # Log the tool execution result
    if result and "messages" in result:
        for msg in result["messages"]:
            if hasattr(msg, 'content'):
                mcp_logger.info(f"🔧 TOOL EXECUTION COMPLETE")
                mcp_logger.info(f"Result preview: {str(msg.content)[:200]}...")
    
    return result


# ---------------------------------------------------------
# User Query Node
# ---------------------------------------------------------
def user_query_node(state: MessagesState):
    return {"messages": state["messages"]}

# ---------------------------------------------------------
# Router Node
# ---------------------------------------------------------
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
        - If the user asks about markets, sectors, macro trends, stock prices, market data → market
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
# Agent Nodes
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

def portfolio_agent(state: MessagesState):
    portfolio_agent_logger.info("➡️ Entered PORTFOLIO agent with state: %s", state)
    system = SystemMessage(content="You are a Portfolio Analysis Agent. Analyze holdings, risk, diversification, and performance.")
    response = llm.invoke([system] + state["messages"])
    portfolio_agent_logger.info("⬅️ Exiting PORTFOLIO agent with response: %s", response)
    return {"messages": state["messages"] + [response]}

def market_agent(state: MessagesState):
    market_agent_logger.info("➡️ Entered MARKET agent with state: %s", state)
    
    system = SystemMessage(content="""You are a Market Analysis Agent with access to real-time market data.

When users ask about stocks, markets, or specific companies:
1. Use the get_market_data tool to fetch current data for specific stock symbols
2. Analyze the data and provide insights
3. Explain market trends, performance, and what the numbers mean

If the user mentions a company name, identify the ticker symbol and use the tool.""")
    
    response = llm_with_tools.invoke([system] + state["messages"])
    market_agent_logger.info("⬅️ Market agent response: %s", response)
    
    return {"messages": state["messages"] + [response]}

def goal_agent(state: MessagesState):
    goal_agent_logger.info("➡️ Entered GOAL agent with state: %s", state)
    system = SystemMessage(content="You are a Goal Planning Agent. Help users plan financial goals, budgets, and timelines.")
    response = llm.invoke([system] + state["messages"])
    goal_agent_logger.info("⬅️ Exiting GOAL agent with response: %s", response)
    return {"messages": state["messages"] + [response]}

def news_agent(state: MessagesState):
    news_agent_logger.info("➡️ Entered NEWS agent with state: %s", state)
    
    system = SystemMessage(content="""You are a News Synthesizer Agent with access to market data and news.

Your workflow when asked about stock news:
1. FIRST call get_market_data(symbol) to understand current stock performance
2. THEN call get_stock_news(symbol) to get recent news articles
3. Synthesize both: Explain how the news relates to stock movement
4. Provide investor insights and implications

Always provide comprehensive analysis combining market context with news.""")
    
    response = llm_with_news_agent_tools.invoke([system] + state["messages"])
    news_agent_logger.info("⬅️ Exiting NEWS agent with response: %s", response)
    
    return {"messages": state["messages"] + [response]}

def tax_agent(state: MessagesState):
    tax_agent_logger.info("➡️ Entered TAX agent with state: %s", state)
    system = SystemMessage(content="You are a Tax Education Agent. Explain tax concepts, rules, and account types.")
    response = llm.invoke([system] + state["messages"])
    tax_agent_logger.info("⬅️ Exiting TAX agent with response: %s", response)
    return {"messages": state["messages"] + [response]}

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
graph.add_node("tools", logged_tool_node)
graph.add_node("goal", goal_agent)
graph.add_node("news", news_agent)
graph.add_node("news_tools", news_agent_tool_node)
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

# Market agent can call tools or end
graph.add_conditional_edges(
    "market",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

# After tools execute, go back to market agent
graph.add_edge("tools", "market")

# News agent can call tools or end
graph.add_conditional_edges(
    "news",
    news_agent_should_continue,
    {
        "news_tools": "news_tools",
        "end": END,
    },
)

# After news tools execute, go back to news agent
graph.add_edge("news_tools", "news")

# Entry point
graph.set_entry_point("user_query")

# ---------------------------------------------------------
# Compile with SQLite Checkpointer
# ---------------------------------------------------------
import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = graph.compile(checkpointer=checkpointer)