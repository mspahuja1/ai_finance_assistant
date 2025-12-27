"""Portfolio Agent - Portfolio analysis and recommendations"""
# Designed and developed by Mandeep Pahuja

import sys
import os

# Add src directory to path FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

# Now import packages
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage
from utils.llm_config import llm
from utils.logging_config import create_logger

# ✨ ADD: LangSmith import
from config.langsmith_config import get_langsmith_config

portfolio_logger = create_logger("portfolio_agent", "portfolio_agent.log")

# ===================================================================
# EVALUATION INTEGRATION
# ===================================================================

# ✨ Add evaluation integration
try:
    from judge.evaluation_runner import EvaluationRunner
    EVALUATION_ENABLED = True
except ImportError:
    EVALUATION_ENABLED = False
    portfolio_logger.warning("⚠️ Evaluation system not available")

# Initialize evaluation runner for portfolio agent
if EVALUATION_ENABLED:
    portfolio_eval_runner = EvaluationRunner(eval_dir="evaluations/portfolio")
    portfolio_eval_runner.start()
    portfolio_logger.info("✅ Evaluation system started for portfolio agent")
else:
    portfolio_eval_runner = None


def portfolio_agent(state: MessagesState):
    """Portfolio analysis agent"""
    
    portfolio_logger.info("➡️ Entered Portfolio Agent")
    
    # Get user query for evaluation
    user_message = state["messages"][-1]
    user_query = user_message.content if hasattr(user_message, 'content') else str(user_message)
    
    system = SystemMessage(content="""You are a Portfolio Analysis Agent. 

Your expertise includes:
1. **Holdings Analysis**: Review asset allocation and position sizing
2. **Risk Assessment**: Evaluate portfolio volatility and concentration risk
3. **Diversification**: Assess sector, geography, and asset class spread
4. **Performance Review**: Analyze returns and benchmark comparisons
5. **Rebalancing**: Recommend adjustments to maintain target allocation

Provide actionable, data-driven recommendations.""")
    
    # ✨ NEW: Get LangSmith config
    langsmith_config = get_langsmith_config(
        agent_name="portfolio",
        tags=["analysis", "recommendations", "production"],
        metadata={"agent_version": "1.0"},
        reference_context="Portfolio analysis and risk assessment framework",  # ✨ NEW
        reference_outputs=""
    )
    
    # ✨ MODIFIED: Add config to invoke
    response = llm.invoke(
        [system] + state["messages"],
        config=langsmith_config  # ← Add this
    )
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    # ============================================================
    # QUEUE EVALUATION
    # ============================================================
    if EVALUATION_ENABLED and portfolio_eval_runner:
        try:
            # Detect what type of portfolio analysis was requested
            query_lower = user_query.lower()
            
            analysis_type = "general"
            if any(word in query_lower for word in ["risk", "volatility", "beta", "drawdown"]):
                analysis_type = "risk_assessment"
            elif any(word in query_lower for word in ["diversif", "allocation", "balance", "spread"]):
                analysis_type = "diversification"
            elif any(word in query_lower for word in ["performance", "return", "profit", "gain", "loss"]):
                analysis_type = "performance"
            elif any(word in query_lower for word in ["rebalance", "adjust", "reallocate", "optimize"]):
                analysis_type = "rebalancing"
            elif any(word in query_lower for word in ["holding", "position", "asset", "stock"]):
                analysis_type = "holdings_review"
            
            # Detect if user provided portfolio data
            portfolio_data_provided = any([
                "%" in user_query,
                "$" in user_query,
                any(word in query_lower for word in ["portfolio:", "holdings:", "assets:"]),
                len(user_query.split("\n")) > 3  # Multi-line likely contains data
            ])
            
            # Detect specific recommendations requested
            recommendation_keywords = {
                "rebalancing": any(word in query_lower for word in ["rebalance", "adjust"]),
                "risk_reduction": any(word in query_lower for word in ["reduce risk", "safer", "conservative"]),
                "growth": any(word in query_lower for word in ["growth", "aggressive", "increase return"]),
                "income": any(word in query_lower for word in ["income", "dividend", "yield"])
            }
            
            portfolio_eval_runner.queue_evaluation(
                user_query=user_query,
                agent_response=response_content,
                agent_type="portfolio",
                context={
                    "analysis_type": analysis_type,
                    "portfolio_data_provided": portfolio_data_provided,
                    "risk_assessment": "risk" in analysis_type or "risk" in query_lower,
                    "diversification_analysis": "diversif" in analysis_type,
                    "performance_review": "performance" in analysis_type,
                    "rebalancing_recommended": "rebalance" in analysis_type,
                    "recommendation_type": [k for k, v in recommendation_keywords.items() if v],
                    "actionable_advice": True
                }
            )
            portfolio_logger.info("📊 Evaluation queued (portfolio agent)")
        except Exception as e:
            portfolio_logger.error(f"❌ Evaluation queue failed: {e}")
    
    portfolio_logger.info("⬅️ Exiting Portfolio Agent")
    
    return {"messages": state["messages"] + [response]}