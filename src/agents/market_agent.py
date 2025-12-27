"""Market Agent - Stock prices and market analysis with MCP tools"""
import sys
import os

# ===========================================
# Evaluations - Add evaluation integration
# ===========================================
try:
    from judge.evaluation_runner import EvaluationRunner
    EVALUATION_ENABLED = True
except ImportError:
    EVALUATION_ENABLED = False

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from utils.llm_config import llm
from utils.logging_config import create_logger
from tools.market_data import get_market_data

# ✨ ADD: LangSmith import
from config.langsmith_config import (
    get_langsmith_config,
    trace_tool_call,
    LANGSMITH_ENABLED
)
import time

market_logger = create_logger("market_agent", "market_agent.log")

# ===========================================
# Initialize evaluation runner for market agent
# ===========================================
if EVALUATION_ENABLED:
    market_eval_runner = EvaluationRunner(eval_dir="evaluations/market")
    market_eval_runner.start()
    market_logger.info("✅ Evaluation system started for market agent")
else:
    market_eval_runner = None

# ===========================================
# Bind tools to LLM
# ===========================================
market_tools = [get_market_data]
llm_with_market_tools = llm.bind_tools(market_tools)
market_tool_node = ToolNode(market_tools)


def market_agent(state: MessagesState):
    """Market analysis agent with real-time market data access"""
    
    market_logger.info("➡️ Entered Market Agent")
    
    # Get user query for evaluation
    user_message = state["messages"][-1]
    user_query = user_message.content if hasattr(user_message, 'content') else str(user_message)
    
    # ✨ NEW: Create LangSmith metadata
    ls_metadata = {
        "agent_version": "1.0",
        "has_tools": True,
        "tools_available": ["get_market_data"]
    }
    
    system = SystemMessage(content="""You are a Market Analysis Agent with access to real-time market data.

When users ask about stocks or markets:
1. Use get_market_data(symbol) to fetch current prices and trends
2. Analyze the data and provide insights
3. Explain what the numbers mean for investors

Always fetch data before providing market analysis.""")
    
    # ✨ NEW: Get LangSmith config
    langsmith_config = get_langsmith_config(
        agent_name="market",
        tags=["tools", "real-time-data", "production"],
        metadata=ls_metadata,
        reference_context="Real-time market data from MCP tools",  # ✨ NEW
        reference_outputs=""
    )
    
    # response = llm_with_market_tools.invoke([system] + state["messages"])
    # ✨ MODIFIED: Add config to invoke
    start_time = time.time()
    response = llm_with_market_tools.invoke(
        [system] + state["messages"],
        config=langsmith_config  # ← Add this
    )
    execution_time = time.time() - start_time
    
    # ✨ NEW: Trace tool calls if present
    if LANGSMITH_ENABLED:
        has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
        
        if has_tool_calls:
            for tool_call in response.tool_calls:
                trace_tool_call(
                    tool_name=tool_call.get('name', 'unknown'),
                    tool_input=tool_call.get('args', {}),
                    tool_output="Pending execution",  # Will be filled by tool node
                    agent_name="market",
                    execution_time=execution_time
                )
    
    # ============================================================
    # QUEUE EVALUATION (only for final responses, not tool calls)
    # ============================================================
    # Only evaluate if this is a final response (no tool calls)
    has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
    
    if not has_tool_calls and EVALUATION_ENABLED and market_eval_runner:
        try:
            # Extract response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract tool usage info from message history
            tools_used = []
            ticker_symbols = []
            
            # Look through recent messages for tool calls
            for msg in state["messages"][-5:]:  # Check last 5 messages
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get('name'):
                            tools_used.append(tool_call['name'])
                        if tool_call.get('args', {}).get('symbol'):
                            ticker_symbols.append(tool_call['args']['symbol'])
            
            # Queue evaluation with market-specific context
            market_eval_runner.queue_evaluation(
                user_query=user_query,
                agent_response=response_content,
                agent_type="market",
                context={
                    "tools_used": list(set(tools_used)) if tools_used else ["none"],
                    "ticker_symbols": list(set(ticker_symbols)) if ticker_symbols else [],
                    "real_time_data": bool(tools_used),
                    "mcp_tools": True,
                    "tool_node_used": has_tool_calls
                }
            )
            market_logger.info("📊 Evaluation queued (market agent)")
        except Exception as e:
            market_logger.error(f"❌ Evaluation queue failed: {e}")
    
    market_logger.info("⬅️ Exiting Market Agent")
    
    return {"messages": state["messages"] + [response]}


def market_agent_should_continue(state: MessagesState):
    """Check if market agent needs to call tools"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        market_logger.info("🔧 Tool calls detected, routing to tools")
        return "tools"
    
    market_logger.info("✅ No tool calls, ending conversation")
    return "end"