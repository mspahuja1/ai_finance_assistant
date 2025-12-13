import asyncio
import json
import yfinance as yf

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Create the server
server = Server("finance-assistant")

# List available tools
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_market_data",
            description="Fetch 1-year market history for a stock symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["symbol"]
            }
        )
    ]

# Handle tool calls
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_market_data":
        symbol = arguments.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")
        
        try:    
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1y")
            
            if df.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            result = df.reset_index().to_dict(orient="records")
            
            # Return the result
            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )
            ]
        except Exception as e:
            # Return error message
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )
            ]
    else:
        raise ValueError(f"Unknown tool: {name}")

# Run the server with stdio transport
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())