"""Market Data Tool"""
import json
import time
import sys
import os
from typing import Annotated

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langchain_core.tools import tool
from tools.cache import market_cache
from utils.logging_config import create_logger

# Add MCP directory to path
mcp_dir = os.path.join(src_dir, 'mcp')
sys.path.insert(0, mcp_dir)
from mcp_client import MCPClient

mcp_logger = create_logger("mcp_transactions", "mcp_transactions.log")

# Initialize MCP Client
server_path = os.path.join(mcp_dir, "server.py")
mcp_client = MCPClient(server_path)

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
mcp_dir = os.path.join(parent_dir, 'mcp')
sys.path.insert(0, mcp_dir)

from mcp_client import MCPClient

# Initialize MCP Client
server_path = os.path.join(mcp_dir, "server.py")
mcp_client = MCPClient(server_path)

@tool
def get_market_data(symbol: Annotated[str, "Stock ticker symbol (e.g., AAPL, GOOGL)"]) -> str:
    """Fetch 1-year market history and current data for a stock symbol via MCP"""
    
    request_id = f"mcp-{int(time.time() * 1000)}"
    mcp_logger.info(f"📥 REQUEST [{request_id}] - {symbol}")
    
    # Check cache first
    cached_result = market_cache.get("get_market_data", symbol=symbol)
    if cached_result:
        mcp_logger.info(f"⚡ CACHE HIT [{request_id}]")
        return cached_result
    
    start_time = time.time()
    
    try:
        response = mcp_client.call_tool("get_market_data", {"symbol": symbol})
        
        if "error" in response:
            error_msg = f"MCP Error: {response['error']}"
            mcp_logger.error(f"❌ {error_msg}")
            return error_msg
        
        if "result" in response:
            result_obj = response["result"]
            
            if result_obj.get("isError"):
                error_text = result_obj.get("content", [{}])[0].get("text", "Unknown error")
                mcp_logger.error(f"❌ {error_text}")
                return f"Error: {error_text}"
            
            content = result_obj.get("content", [])
            if content and len(content) > 0:
                text_content = content[0].get("text", "")
                market_data = json.loads(text_content)
                
                if not market_data:
                    return f"No data found for {symbol}"
                
                latest = market_data[-1]
                high_prices = [float(d.get('High', 0)) for d in market_data if 'High' in d]
                low_prices = [float(d.get('Low', 0)) for d in market_data if 'Low' in d]
                volumes = [float(d.get('Volume', 0)) for d in market_data if 'Volume' in d]
                
                year_high = max(high_prices) if high_prices else 0
                year_low = min(low_prices) if low_prices else 0
                avg_volume = sum(volumes) / len(volumes) if volumes else 0
                
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
                
                # Cache the result
                market_cache.set("get_market_data", result, symbol=symbol)
                
                elapsed = time.time() - start_time
                mcp_logger.info(f"✅ SUCCESS [{request_id}] - {elapsed:.3f}s")
                return result
        
        return "Unexpected response format"
        
    except Exception as e:
        elapsed = time.time() - start_time
        mcp_logger.error(f"❌ ERROR [{request_id}] - {str(e)}")
        return f"Error: {str(e)}"