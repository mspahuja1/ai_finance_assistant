"""Goal Agent - Financial planning and budgeting"""
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

# ✨ ADD: LangSmith import
from config.langsmith_config import get_langsmith_config

goal_logger = create_logger("goal_agent", "goal_agent.log")

# ===================================================================
# EVALUATION INTEGRATION
# ===================================================================

# ✨ Add evaluation integration
try:
    from judge.evaluation_runner import EvaluationRunner
    EVALUATION_ENABLED = True
except ImportError:
    EVALUATION_ENABLED = False
    goal_logger.warning("⚠️ Evaluation system not available")

# Initialize evaluation runner for goal agent
if EVALUATION_ENABLED:
    goal_eval_runner = EvaluationRunner(eval_dir="evaluations/goal")
    goal_eval_runner.start()
    goal_logger.info("✅ Evaluation system started for goal agent")
else:
    goal_eval_runner = None


def goal_agent(state: MessagesState):
    """Financial goal planning agent"""
    
    goal_logger.info("➡️ Entered Goal Agent")
    
    # Get user query for evaluation
    user_message = state["messages"][-1]
    user_query = user_message.content if hasattr(user_message, 'content') else str(user_message)
    
    system = SystemMessage(content="""You are a Goal Planning Agent. 
Help users plan financial goals, create budgets, and develop timelines for achieving objectives.

Provide:
1. Specific actionable steps
2. Realistic timelines
3. Budget breakdowns
4. Progress tracking methods""")
    
    # ✨ NEW: Get LangSmith config
    langsmith_config = get_langsmith_config(
        agent_name="goal",
        tags=["planning", "budgeting", "production"],
        metadata={"agent_version": "1.0"},
        reference_context="Financial planning guidelines and best practices",  # ✨ NEW
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
    if EVALUATION_ENABLED and goal_eval_runner:
        try:
            # Extract goal-specific information from query
            goal_keywords = {
                "retirement": "retirement" in user_query.lower(),
                "house": any(word in user_query.lower() for word in ["house", "home", "property"]),
                "college": any(word in user_query.lower() for word in ["college", "education", "tuition"]),
                "emergency_fund": "emergency" in user_query.lower(),
                "debt": "debt" in user_query.lower(),
                "budget": "budget" in user_query.lower()
            }
            
            # Identify primary goal type
            goal_type = "general"
            for key, value in goal_keywords.items():
                if value:
                    goal_type = key
                    break
            
            goal_eval_runner.queue_evaluation(
                user_query=user_query,
                agent_response=response_content,
                agent_type="goal",
                context={
                    "goal_type": goal_type,
                    "planning_requested": True,
                    "budget_analysis": goal_keywords.get("budget", False),
                    "timeline_needed": any(word in user_query.lower() for word in ["when", "how long", "timeline", "years"])
                }
            )
            goal_logger.info("📊 Evaluation queued (goal agent)")
        except Exception as e:
            goal_logger.error(f"❌ Evaluation queue failed: {e}")
    
    goal_logger.info("⬅️ Exiting Goal Agent")
    
    return {"messages": state["messages"] + [response]}