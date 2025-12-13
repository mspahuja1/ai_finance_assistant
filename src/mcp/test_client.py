import asyncio
from mcp.client import Client

async def main():
    # Connect to your MCP server by name
    client = Client("finance-assistant")

    # Call your tool
    result = await client.call(
        "get_market_data",
        {"symbol": "AAPL"}
    )

    print("✅ MCP Tool Result:")
    print(result)

asyncio.run(main())