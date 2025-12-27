"""
LangSmith Configuration for AI Finance Assistant

Provides centralized LangSmith setup with project-specific tags and metadata.
Aligned with LangGraph project configuration.
"""

import os
from typing import Dict, Any, Optional, List
from langsmith import Client
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Force disable LangSmith auto-evaluation (playground)
os.environ["LANGCHAIN_EVALUATION_ENABLED"] = "false"

# LangSmith configuration
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "AI-Finance-Assistant")

# Project metadata (from LangGraph configuration)
PROJECT_METADATA = {
    "project_name": "AI-Finance-Assistant",
    "description": "Multi-agent workflow for portfolio, market, tax, and goal planning.",
    "domain": "finance",
    "team": "AI Engineering",
    "owner": "mandeep",
    "framework": "langgraph",
    "version": "1.0.0"
}

# Base tags from LangGraph configuration
BASE_TAGS = ["langgraph", "mcp", "gateway", "rag", "agents", "finance-assistant"]

# Initialize client if enabled
langsmith_client = None
if LANGSMITH_ENABLED and LANGSMITH_API_KEY:
    try:
        langsmith_client = Client(api_key=LANGSMITH_API_KEY)
        logger.info(f"✅ LangSmith enabled - Project: {LANGSMITH_PROJECT}")
        logger.info(f"   Team: {PROJECT_METADATA['team']} | Owner: {PROJECT_METADATA['owner']}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LangSmith: {e}")
        LANGSMITH_ENABLED = False
else:
    logger.warning("⚠️ LangSmith not configured")


def get_langsmith_config(
    agent_name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    reference_context: Optional[str] = None,  # ✨ NEW
    reference_outputs: Optional[str] = None   # ✨ NEW
) -> Dict[str, Any]:
    """
    Get LangSmith configuration for an agent.
    
    Args:
        agent_name: Name of agent (finance, market, goal, news, portfolio, tax)
        tags: Additional tags for this run (appended to base tags)
        metadata: Additional metadata (merged with project metadata)
        
    Returns:
        Configuration dict for LangChain with tags and metadata
        
    Example:
        config = get_langsmith_config(
            agent_name="finance",
            tags=["rag-enabled", "cache-hit"],
            metadata={"num_chunks": 3}
        )
        response = llm.invoke(messages, config=config)
    """
    if not LANGSMITH_ENABLED:
        return {}
    
    # Build comprehensive tags
    agent_tags = [
        f"agent:{agent_name}",
        f"domain:{PROJECT_METADATA['domain']}",
        "environment:production"
    ]
    
    # Combine: BASE_TAGS + agent_tags + custom tags
    all_tags = BASE_TAGS + agent_tags + (tags or [])
    
    # Build comprehensive metadata
    run_metadata = {
        **PROJECT_METADATA,  # Include all project metadata
        "agent": agent_name,
        "agent_type": agent_name,
    }
    
    # ✨ NEW: Add LangSmith evaluator variables
    # These make your traces compatible with LangSmith's built-in evaluators
    if reference_context is not None:
        run_metadata["context"] = reference_context
    
    if reference_outputs is not None:
        run_metadata["reference_outputs"] = reference_outputs
    
    # Merge with additional metadata
    if metadata:
        run_metadata.update(metadata)
    
    return {
        "tags": all_tags,
        "metadata": run_metadata,
        "project_name": LANGSMITH_PROJECT
    }


def create_langsmith_callback(
    agent_name: str,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> List[BaseCallbackHandler]:
    """
    Create LangSmith callback handlers for explicit tracing.
    
    Args:
        agent_name: Name of agent
        run_name: Custom name for this run (defaults to agent_name)
        tags: Additional tags
        metadata: Additional metadata
        
    Returns:
        List of callback handlers (empty if LangSmith disabled)
        
    Example:
        callbacks = create_langsmith_callback(
            agent_name="finance",
            run_name="rag_retrieval",
            tags=["retrieval"]
        )
        docs = vector_store.similarity_search(query, callbacks=callbacks)
    """
    if not LANGSMITH_ENABLED:
        return []
    
    config = get_langsmith_config(agent_name, tags, metadata)
    
    try:
        tracer = LangChainTracer(
            project_name=config["project_name"],
            tags=config["tags"],
            metadata=config["metadata"]
        )
        return [tracer]
    except Exception as e:
        logger.error(f"Failed to create LangSmith callback: {e}")
        return []


def log_to_langsmith(
    run_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    agent_name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_type: str = "chain"
):
    """
    Manually log a run to LangSmith (for non-LangChain components).
    
    Args:
        run_name: Name of the run (e.g., "RAG Retrieval", "Cache Lookup")
        inputs: Input data
        outputs: Output data
        agent_name: Name of agent
        tags: Additional tags
        metadata: Additional metadata
        run_type: Type of run (chain, llm, tool, retriever, etc.)
        
    Example:
        log_to_langsmith(
            run_name="RAG Retrieval",
            inputs={"query": "What is compound interest?"},
            outputs={"num_docs": 3, "cache_hit": False},
            agent_name="finance",
            tags=["retrieval", "fresh"],
            run_type="retriever"
        )
    """
    if not LANGSMITH_ENABLED or not langsmith_client:
        return
    
    config = get_langsmith_config(agent_name, tags, metadata)
    
    try:
        langsmith_client.create_run(
            name=run_name,
            inputs=inputs,
            outputs=outputs,
            run_type=run_type,
            tags=config["tags"],
            extra=config["metadata"]
        )
    except Exception as e:
        logger.error(f"Failed to log to LangSmith: {e}")


def update_run_metadata(
    run_id: str,
    metadata: Dict[str, Any]
):
    """
    Update metadata for an existing run.
    
    Args:
        run_id: LangSmith run ID
        metadata: Metadata to update/add
        
    Example:
        update_run_metadata(
            run_id="abc-123",
            metadata={"cache_hit": True, "latency_ms": 150}
        )
    """
    if not LANGSMITH_ENABLED or not langsmith_client:
        return
    
    try:
        langsmith_client.update_run(
            run_id=run_id,
            extra=metadata
        )
    except Exception as e:
        logger.error(f"Failed to update run metadata: {e}")


# ============================================================================
# SPECIALIZED TRACING FUNCTIONS FOR AI FINANCE ASSISTANT
# ============================================================================

def trace_rag_retrieval(
    query: str,
    retrieved_docs: List[Any],
    agent_name: str,
    num_chunks: int,
    cache_hit: bool = False,
    similarity_scores: Optional[List[float]] = None
):
    """
    Log RAG retrieval to LangSmith with finance-specific metadata.
    
    Args:
        query: User query
        retrieved_docs: Retrieved documents
        agent_name: Agent name
        num_chunks: Number of chunks retrieved
        cache_hit: Whether result came from cache
        similarity_scores: Similarity scores for each doc
    """
    doc_titles = []
    doc_sources = []
    
    for doc in retrieved_docs:
        if hasattr(doc, 'metadata'):
            doc_titles.append(doc.metadata.get("title", "Unknown"))
            doc_sources.append(doc.metadata.get("source", "Unknown"))
    
    log_to_langsmith(
        run_name="RAG Retrieval",
        inputs={
            "query": query,
            "k": num_chunks,
            "cache_lookup": True
        },
        outputs={
            "num_docs_retrieved": len(retrieved_docs),
            "cache_hit": cache_hit,
            "doc_titles": doc_titles,
            "doc_sources": doc_sources,
            "similarity_scores": similarity_scores or []
        },
        agent_name=agent_name,
        tags=[
            "rag",
            "retrieval",
            "cache-hit" if cache_hit else "cache-miss",
            f"chunks:{num_chunks}"
        ],
        metadata={
            "retrieval_method": "faiss_similarity_search",
            "cache_hit": cache_hit,
            "num_chunks": num_chunks,
            "has_sources": len(doc_sources) > 0
        },
        run_type="retriever"
    )


def trace_cache_lookup(
    query: str,
    cache_type: str,
    cache_hit: bool,
    agent_name: str,
    similarity_score: Optional[float] = None,
    cache_size: Optional[int] = None
):
    """
    Log cache lookup to LangSmith.
    
    Args:
        query: User query
        cache_type: Type of cache (rag_context, llm_response)
        cache_hit: Whether cache hit occurred
        agent_name: Agent name
        similarity_score: Similarity score if hit
        cache_size: Current cache size
    """
    log_to_langsmith(
        run_name=f"Cache Lookup - {cache_type}",
        inputs={
            "query": query,
            "cache_type": cache_type,
            "similarity_threshold": 0.95 if cache_type == "rag_context" else 0.92
        },
        outputs={
            "cache_hit": cache_hit,
            "similarity_score": similarity_score,
            "cache_size": cache_size
        },
        agent_name=agent_name,
        tags=[
            "cache",
            cache_type,
            "hit" if cache_hit else "miss",
            "semantic-cache"
        ],
        metadata={
            "cache_type": cache_type,
            "cache_hit": cache_hit,
            "similarity_score": similarity_score,
            "cache_implementation": "two_level_semantic"
        },
        run_type="tool"
    )


def trace_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Any,
    agent_name: str,
    execution_time: Optional[float] = None,
    success: bool = True,
    error: Optional[str] = None
):
    """
    Log tool call to LangSmith (for MCP tools, market data, etc.).
    
    Args:
        tool_name: Name of tool (get_market_data, get_stock_news, etc.)
        tool_input: Tool input parameters
        tool_output: Tool output (truncated if too long)
        agent_name: Agent name
        execution_time: Execution time in seconds
        success: Whether tool call succeeded
        error: Error message if failed
    """
    # Truncate long outputs
    output_str = str(tool_output)
    if len(output_str) > 1000:
        output_str = output_str[:1000] + "... (truncated)"
    
    log_to_langsmith(
        run_name=f"Tool Call - {tool_name}",
        inputs={
            "tool": tool_name,
            "parameters": tool_input
        },
        outputs={
            "result": output_str,
            "success": success,
            "error": error,
            "execution_time_ms": execution_time * 1000 if execution_time else None
        },
        agent_name=agent_name,
        tags=[
            "tool",
            tool_name,
            "mcp" if "get_" in tool_name else "standard",
            "success" if success else "error"
        ],
        metadata={
            "tool_name": tool_name,
            "tool_type": "mcp_tool",
            "execution_time_ms": execution_time * 1000 if execution_time else None,
            "success": success
        },
        run_type="tool"
    )


def trace_evaluation(
    user_query: str,
    agent_response: str,
    evaluation_result: Dict[str, Any],
    agent_name: str
):
    """
    Log LLM-as-a-Judge evaluation result to LangSmith.
    
    Args:
        user_query: User query
        agent_response: Agent response (truncated)
        evaluation_result: Evaluation result dict
        agent_name: Agent name
    """
    # Truncate long content
    response_truncated = agent_response[:500] + "..." if len(agent_response) > 500 else agent_response
    
    log_to_langsmith(
        run_name="LLM-as-Judge Evaluation",
        inputs={
            "query": user_query,
            "response": response_truncated
        },
        outputs={
            "overall_score": evaluation_result.get("overall_score"),
            "composite_score": evaluation_result.get("composite_score"),
            "scores": {
                "accuracy": evaluation_result.get("accuracy_score"),
                "completeness": evaluation_result.get("completeness_score"),
                "clarity": evaluation_result.get("clarity_score"),
                "safety": evaluation_result.get("safety_score")
            },
            "strengths": evaluation_result.get("strengths", [])[:3],  # Top 3
            "weaknesses": evaluation_result.get("weaknesses", [])[:3],  # Top 3
            "judge_name": evaluation_result.get("judge_name", "unknown")
        },
        agent_name=agent_name,
        tags=[
            "evaluation",
            "llm-as-judge",
            f"score:{evaluation_result.get('overall_score', 0)}",
            "quality-assessment"
        ],
        metadata={
            "judge_type": evaluation_result.get("judge_name"),
            "technique": evaluation_result.get("technique", "unknown"),
            "overall_score": evaluation_result.get("overall_score"),
            "composite_score": evaluation_result.get("composite_score"),
            "evaluation_system": "llm_as_judge"
        },
        run_type="chain"
    )


def trace_agent_routing(
    user_query: str,
    selected_agent: str,
    router_reasoning: Optional[str] = None,
    confidence: Optional[float] = None
):
    """
    Log router agent decision to LangSmith.
    
    Args:
        user_query: User query
        selected_agent: Which agent was selected
        router_reasoning: Router's reasoning (if available)
        confidence: Confidence score (if available)
    """
    log_to_langsmith(
        run_name="Router Agent Decision",
        inputs={
            "query": user_query,
            "available_agents": ["finance", "market", "goal", "news", "portfolio", "tax"]
        },
        outputs={
            "selected_agent": selected_agent,
            "reasoning": router_reasoning,
            "confidence": confidence
        },
        agent_name="router",
        tags=[
            "router",
            "agent-selection",
            f"routed-to:{selected_agent}"
        ],
        metadata={
            "selected_agent": selected_agent,
            "confidence": confidence,
            "routing_method": "llm_based"
        },
        run_type="chain"
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_project_info() -> Dict[str, Any]:
    """
    Get current project configuration info.
    
    Returns:
        Project metadata dictionary
    """
    return {
        **PROJECT_METADATA,
        "langsmith_enabled": LANGSMITH_ENABLED,
        "langsmith_project": LANGSMITH_PROJECT,
        "base_tags": BASE_TAGS
    }


def is_langsmith_enabled() -> bool:
    """Check if LangSmith is enabled."""
    return LANGSMITH_ENABLED


def get_client() -> Optional[Client]:
    """Get LangSmith client (if available)."""
    return langsmith_client