"""News Agent - Financial news synthesis with market context"""
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from utils.llm_config import llm
from utils.logging_config import create_logger
from tools.market_data import get_market_data
from tools.stock_news import get_stock_news

# ✨ ADD: LangSmith import
from config.langsmith_config import (
    get_langsmith_config,
    trace_tool_call,
    LANGSMITH_ENABLED
)
import time

news_logger = create_logger("news_agent", "news_agent.log")

# ===================================================================
# EVALUATION INTEGRATION
# ===================================================================

# ✨ Add evaluation integration
try:
    from judge.evaluation_runner import EvaluationRunner
    EVALUATION_ENABLED = True
except ImportError:
    EVALUATION_ENABLED = False
    news_logger.warning("⚠️ Evaluation system not available")

# Initialize evaluation runner for news agent
if EVALUATION_ENABLED:
    news_eval_runner = EvaluationRunner(eval_dir="evaluations/news")
    news_eval_runner.start()
    news_logger.info("✅ Evaluation system started for news agent")
else:
    news_eval_runner = None

# Bind tools to LLM
news_agent_tools = [get_market_data, get_stock_news]
llm_with_news_agent_tools = llm.bind_tools(news_agent_tools)
news_agent_tool_node = ToolNode(news_agent_tools)


def news_agent(state: MessagesState):
    """News synthesis agent with market data and news access"""
    
    news_logger.info("➡️ Entered News Agent")
    
    # Get user query for evaluation
    user_message = state["messages"][-1]
    user_query = user_message.content if hasattr(user_message, 'content') else str(user_message)
    
    # ✨ NEW: Create LangSmith metadata
    ls_metadata = {
        "agent_version": "1.0",
        "has_tools": True,
        "tools_available": ["get_market_data", "get_stock_news"]
    }
    
    system = SystemMessage(content="""You are a News Synthesizer Agent with access to market data and news.

Your workflow:
1. Call get_market_data(symbol) for current stock performance
2. Call get_stock_news(symbol) for recent news articles
3. Synthesize: Connect news to stock movement
4. Provide investor insights

Always combine market context with news analysis.""")
    
    # ✨ NEW: Get LangSmith config
    langsmith_config = get_langsmith_config(
        agent_name="news",
        tags=["tools", "news-synthesis", "production"],
        metadata=ls_metadata,
        reference_context="Financial news and market data synthesis",  # ✨ NEW
        reference_outputs=""
    )
    
    # ✨ MODIFIED: Add config to invoke
    start_time = time.time()
    response = llm_with_news_agent_tools.invoke(
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
                    tool_output="Pending execution",
                    agent_name="news",
                    execution_time=execution_time
                )

    
    # ============================================================
    # QUEUE EVALUATION (only for final responses, not tool calls)
    # ============================================================
    # Only evaluate if this is a final response (no tool calls)
    has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
    
    if not has_tool_calls and EVALUATION_ENABLED and news_eval_runner:
        try:
            # Extract response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract tool usage info from message history
            tools_used = []
            ticker_symbols = []
            news_fetched = False
            market_data_fetched = False
            
            # Look through recent messages for tool calls
            for msg in state["messages"][-5:]:  # Check last 5 messages
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', '')
                        if tool_name:
                            tools_used.append(tool_name)
                            
                            # Track specific tool types
                            if tool_name == 'get_stock_news':
                                news_fetched = True
                            elif tool_name == 'get_market_data':
                                market_data_fetched = True
                            
                            # Extract ticker symbols
                            if tool_call.get('args', {}).get('symbol'):
                                ticker_symbols.append(tool_call['args']['symbol'])
            
            # Queue evaluation with news-specific context
            news_eval_runner.queue_evaluation(
                user_query=user_query,
                agent_response=response_content,
                agent_type="news",
                context={
                    "tools_used": list(set(tools_used)) if tools_used else ["none"],
                    "ticker_symbols": list(set(ticker_symbols)) if ticker_symbols else [],
                    "news_fetched": news_fetched,
                    "market_data_fetched": market_data_fetched,
                    "synthesis_provided": news_fetched and market_data_fetched,  # Both tools used
                    "mcp_tools": True,
                    "news_sources": ["stock_news_api"] if news_fetched else [],
                    "time_range": "recent"  # Based on your news tool's typical range
                }
            )
            news_logger.info("📊 Evaluation queued (news agent)")
        except Exception as e:
            news_logger.error(f"❌ Evaluation queue failed: {e}")
    
    news_logger.info("⬅️ Exiting News Agent")
    
    return {"messages": state["messages"] + [response]}


def news_agent_should_continue(state: MessagesState):
    """Check if news agent needs to call tools"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        news_logger.info("🔧 Tool calls detected, routing to tools")
        return "news_tools"
    
    news_logger.info("✅ No tool calls, ending conversation")
    return "end"