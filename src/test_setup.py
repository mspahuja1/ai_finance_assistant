"""Test all imports and dependencies"""
import sys

def test_imports():
    """Test all required imports"""
    tests = []
    
    # Test 1: LangGraph
    try:
        import langgraph
        from langgraph.graph import StateGraph
        tests.append(("✅ LangGraph", True))
    except Exception as e:
        tests.append((f"❌ LangGraph: {e}", False))
    
    # Test 2: LangChain
    try:
        import langchain
        from langchain_core.messages import HumanMessage
        tests.append(("✅ LangChain", True))
    except Exception as e:
        tests.append((f"❌ LangChain: {e}", False))
    
    # Test 3: Google GenAI
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        tests.append(("✅ Google GenAI", True))
    except Exception as e:
        tests.append((f"❌ Google GenAI: {e}", False))
    
    # Test 4: yfinance
    try:
        import yfinance as yf
        tests.append(("✅ yfinance", True))
    except Exception as e:
        tests.append((f"❌ yfinance: {e}", False))
    
    # Test 5: Streamlit
    try:
        import streamlit
        tests.append(("✅ Streamlit", True))
    except Exception as e:
        tests.append((f"❌ Streamlit: {e}", False))
    
    # Test 6: MCP
    try:
        import mcp
        tests.append(("✅ MCP", True))
    except Exception as e:
        tests.append((f"❌ MCP: {e}", False))
    
    # Test 7: Utils
    try:
        from utils.llm_config import llm
        tests.append(("✅ Utils (LLM)", True))
    except Exception as e:
        tests.append((f"❌ Utils: {e}", False))
    
    # Test 8: Tools
    try:
        from tools.cache import market_cache
        tests.append(("✅ Tools (Cache)", True))
    except Exception as e:
        tests.append((f"❌ Tools: {e}", False))
    
    # Test 9: Agents
    try:
        from agents.router import router_agent
        tests.append(("✅ Agents (Router)", True))
    except Exception as e:
        tests.append((f"❌ Agents: {e}", False))
    
    # Print results
    print("\n" + "="*50)
    print("DEPENDENCY TEST RESULTS")
    print("="*50)
    
    all_passed = True
    for test, passed in tests:
        print(test)
        if not passed:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to run Streamlit.")
    else:
        print("❌ Some tests failed. Install missing dependencies.")
    
    return all_passed

if __name__ == "__main__":
    test_imports()