"""Tax Agent - Tax education and strategies"""
import sys
import os

# Add src directory to path FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage
from utils.llm_config import llm
from utils.logging_config import create_logger
from datetime import datetime

# ✨ ADD: LangSmith import
from config.langsmith_config import get_langsmith_config

tax_logger = create_logger("tax_agent", "tax_agent.log")

# ===================================================================
# EVALUATION INTEGRATION
# ===================================================================

# ✨ Add evaluation integration
try:
    from judge.evaluation_runner import EvaluationRunner
    EVALUATION_ENABLED = True
except ImportError:
    EVALUATION_ENABLED = False
    tax_logger.warning("⚠️ Evaluation system not available")

# Initialize evaluation runner for tax agent
if EVALUATION_ENABLED:
    tax_eval_runner = EvaluationRunner(eval_dir="evaluations/tax")
    tax_eval_runner.start()
    tax_logger.info("✅ Evaluation system started for tax agent")
else:
    tax_eval_runner = None


def tax_agent(state: MessagesState):
    """Tax education agent"""
    
    tax_logger.info("➡️ Entered Tax Agent")
    
    # Get user query for evaluation
    user_message = state["messages"][-1]
    user_query = user_message.content if hasattr(user_message, 'content') else str(user_message)
    
    system = SystemMessage(content="""You are a Tax Education Agent with expertise in US tax law and strategies.

Your knowledge areas:
1. **Tax Concepts**: Explain deductions, credits, brackets, AMT, capital gains
2. **Account Types**: 401(k), IRA, Roth IRA, HSA, 529, taxable accounts
3. **Tax Optimization**: Tax-loss harvesting, Roth conversions, bracket management
4. **Filing Status**: Single, MFJ, HOH, and their implications
5. **Business Taxes**: Self-employment, S-Corp, LLC structures

**CRITICAL DISCLAIMER**: Always remind users that you provide educational information only, 
not personalized tax advice. Users should consult a licensed CPA or tax professional for their specific situation.

Provide clear, accurate tax education with examples.""")
    
    # ✨ NEW: Get LangSmith config
    langsmith_config = get_langsmith_config(
        agent_name="tax",
        tags=["education", "tax-planning", "production"],
        metadata={"agent_version": "1.0", "requires_disclaimer": True},
        reference_context="Tax education and planning strategies",  # ✨ NEW
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
    if EVALUATION_ENABLED and tax_eval_runner:
        try:
            query_lower = user_query.lower()
            
            # Detect tax topic
            tax_topics = {
                "deductions": any(word in query_lower for word in ["deduction", "deduct", "write off", "expense"]),
                "credits": "credit" in query_lower,
                "capital_gains": any(word in query_lower for word in ["capital gain", "capital loss", "investment tax"]),
                "retirement_accounts": any(word in query_lower for word in ["401k", "ira", "roth", "retirement account"]),
                "hsa": "hsa" in query_lower or "health savings" in query_lower,
                "529": "529" in query_lower or "education savings" in query_lower,
                "tax_brackets": any(word in query_lower for word in ["bracket", "tax rate", "marginal"]),
                "tax_loss_harvesting": any(word in query_lower for word in ["tax loss", "harvest", "offset gains"]),
                "roth_conversion": "roth conversion" in query_lower,
                "estimated_tax": any(word in query_lower for word in ["estimated tax", "quarterly", "self employ"]),
                "business_tax": any(word in query_lower for word in ["business tax", "llc", "s corp", "self employ"])
            }
            
            # Identify primary topic
            primary_topic = "general"
            for topic, detected in tax_topics.items():
                if detected:
                    primary_topic = topic
                    break
            
            # Detect if optimization strategy was requested
            optimization_requested = any(word in query_lower for word in [
                "optimize", "minimize", "reduce tax", "save on tax", "strategy", "lower tax"
            ])
            
            # Detect tax year (current year by default)
            current_year = datetime.now().year
            tax_year = str(current_year)
            
            # Check if specific year mentioned
            for year in range(current_year - 2, current_year + 2):
                if str(year) in user_query:
                    tax_year = str(year)
                    break
            
            # Check if disclaimer was likely included in response
            disclaimer_keywords = ["consult", "professional", "cpa", "tax advisor", "not tax advice"]
            disclaimer_included = any(keyword in response_content.lower() for keyword in disclaimer_keywords)
            
            tax_eval_runner.queue_evaluation(
                user_query=user_query,
                agent_response=response_content,
                agent_type="tax",
                context={
                    "tax_topic": primary_topic,
                    "topics_covered": [topic for topic, detected in tax_topics.items() if detected],
                    "optimization_strategy": optimization_requested,
                    "tax_year": tax_year,
                    "disclaimer_included": disclaimer_included,
                    "account_types_discussed": tax_topics.get("retirement_accounts", False) or tax_topics.get("hsa", False),
                    "educational_content": True,
                    "requires_professional_advice": True  # Tax matters always benefit from professional advice
                }
            )
            tax_logger.info("📊 Evaluation queued (tax agent)")
        except Exception as e:
            tax_logger.error(f"❌ Evaluation queue failed: {e}")
    
    tax_logger.info("⬅️ Exiting Tax Agent")
    
    return {"messages": state["messages"] + [response]}