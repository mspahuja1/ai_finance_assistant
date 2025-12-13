"""
MCP Client for Finance Assistant
Communicates with the MCP server to fetch market data
"""

import subprocess
import json
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_client")


class MCPClient:
    """Client to communicate with MCP server via stdio"""
    
    def __init__(self, server_script_path: str):
        """
        Initialize MCP Client
        
        Args:
            server_script_path: Path to the MCP server Python script
        """
        self.server_script_path = server_script_path
        self.request_id = 0
        self.process = None
        logger.info(f"Initialized MCP Client with server: {server_script_path}")
    
    def _create_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a JSON-RPC request"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params is not None:
            request["params"] = params
        return request
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the MCP server"""
        try:
            stdout, stderr = self.process.communicate(
                input=json.dumps(request) + "\n",
                timeout=30
            )
            
            if stderr:
                logger.warning(f"⚠️ MCP server stderr: {stderr}")
            
            if not stdout.strip():
                logger.error("❌ Empty response from MCP server")
                return {"error": "Empty response from MCP server"}
            
            # Try to parse each line as JSON (server might send multiple responses)
            lines = stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    try:
                        response = json.loads(line)
                        # Return the first valid response with matching ID
                        if response.get("id") == request.get("id"):
                            return response
                    except json.JSONDecodeError:
                        continue
            
            # If no matching response found, return the last parsed response
            if lines:
                return json.loads(lines[-1])
            
            return {"error": "No valid response received"}
            
        except subprocess.TimeoutExpired:
            logger.error("❌ MCP server timeout")
            return {"error": "MCP server timeout after 30 seconds"}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse MCP response: {e}")
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            logger.error(f"❌ MCP communication error: {e}")
            return {"error": str(e)}
    
    def initialize(self) -> bool:
        """
        Initialize the MCP server connection
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Start the MCP server process
            self.process = subprocess.Popen(
                ["python", self.server_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send initialization request
            init_request = self._create_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "finance-assistant-client",
                    "version": "1.0.0"
                }
            })
            
            logger.info("📤 Sending initialization request")
            response = self._send_request(init_request)
            
            if "error" in response:
                logger.error(f"❌ Initialization failed: {response['error']}")
                return False
            
            logger.info("✅ MCP server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP server: {e}")
            return False
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call (e.g., "get_market_data")
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            Dictionary containing the result or error
        """
        # Initialize if not already done
        if self.process is None:
            if not self.initialize():
                return {"error": "Failed to initialize MCP server"}
        
        request = self._create_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        logger.info(f"📤 Sending MCP request: {tool_name} with args: {arguments}")
        
        try:
            # Create a new process for each call (simpler approach)
            process = subprocess.Popen(
                ["python", self.server_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send initialize first
            init_request = self._create_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "finance-assistant-client",
                    "version": "1.0.0"
                }
            })
            
            # Send both requests
            requests = json.dumps(init_request) + "\n" + json.dumps(request) + "\n"
            stdout, stderr = process.communicate(input=requests, timeout=30)
            
            if stderr:
                logger.warning(f"⚠️ MCP server stderr: {stderr}")
            
            if not stdout.strip():
                return {"error": "Empty response from MCP server"}
            
            # Parse responses - look for the tool call response
            lines = stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    try:
                        response = json.loads(line)
                        # Look for the tool call response (not initialization)
                        if response.get("id") == request.get("id"):
                            if "error" in response:
                                logger.error(f"❌ MCP server returned error: {response['error']}")
                            else:
                                logger.info(f"✅ MCP call successful")
                            return response
                    except json.JSONDecodeError:
                        continue
            
            return {"error": "Tool response not found in server output"}
            
        except subprocess.TimeoutExpired:
            logger.error("❌ MCP server timeout")
            return {"error": "MCP server timeout after 30 seconds"}
        except Exception as e:
            logger.error(f"❌ MCP client error: {e}")
            return {"error": str(e)}
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Convenience method to fetch market data for a stock symbol
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            Dictionary containing market data or None on error
        """
        response = self.call_tool("get_market_data", {"symbol": symbol})
        
        if "error" in response:
            logger.error(f"Failed to get market data for {symbol}: {response['error']}")
            return None
        
        if "result" in response:
            return response["result"]
        
        return None
    
    def close(self):
        """Close the MCP server process"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            logger.info("🔌 MCP server connection closed")


# Convenience function for single-use calls
def get_market_data(symbol: str, server_path: str = "./server.py") -> Optional[Dict]:
    """
    Quick function to get market data without managing client lifecycle
    
    Args:
        symbol: Stock ticker symbol
        server_path: Path to MCP server script
        
    Returns:
        Market data dictionary or None
    """
    client = MCPClient(server_path)
    result = client.get_market_data(symbol)
    client.close()
    return result


if __name__ == "__main__":
    # Test the client
    import os
    
    # Get the path to server.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "server.py")
    
    print("🧪 Testing MCP Client")
    print("=" * 50)
    
    # Create client
    client = MCPClient(server_path)
    
    # Test: Get market data
    print("\n📊 Fetching market data for AAPL...")
    result = client.get_market_data("AAPL")
    if result:
        print(f"  ✅ Success!")
        if isinstance(result, list) and len(result) > 0:
            # Parse the text content
            text_data = result[0].get("text", "")
            import json
            data = json.loads(text_data)
            print(f"  Got {len(data)} data points")
            if data:
                latest = data[-1]
                print(f"  Latest close: ${latest.get('Close', 'N/A')}")
    else:
        print("  ❌ Failed to get data")
    
    client.close()
    
    print("\n" + "=" * 50)
    print("✅ Tests complete")