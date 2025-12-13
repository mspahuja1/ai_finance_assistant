from langchain_core.messages import AIMessage
from mcp.mcp_client import call_mcp_tool
import asyncio

async def market_analysis_agent(state):
    # Extract symbol from user message
    symbol = state["messages"][-1].content.strip()

    # Call MCP tool
    mcp_result = await call_mcp_tool(
        "get_market_data",
        {"symbol": symbol}
    )

    # MCP returns a list of TextContent objects
    raw_json = mcp_result[0].text

    analysis = f"Fetched market data for {symbol}. Here is the raw JSON:\n{raw_json}"

    return {
        "messages": [AIMessage(content=analysis)],
        "next": "summary"
    }