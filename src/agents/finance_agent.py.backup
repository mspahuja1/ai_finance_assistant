"""Finance Agent - General financial education with RAG and Two-Level Semantic Caching"""
# Designed and developed by Mandeep Pahuja

import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage, AIMessage
from utils.llm_config import llm
from utils.logging_config import create_logger
from utils.semantic_cache import TwoLevelCache  # Import cache module

# RAG imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

finance_logger = create_logger("finance_agent", "finance_agent.log")

# ===================================================================
# EVALUATION INTEGRATION
# ===================================================================

# ✨ Add evaluation integration
try:
    from judge.evaluation_runner import EvaluationRunner
    EVALUATION_ENABLED = True
except ImportError:
    EVALUATION_ENABLED = False
    finance_logger.warning("⚠️ Evaluation system not available")

# ===================================================================
# TOKEN COUNTING UTILITIES
# ===================================================================

def estimate_gemini_tokens(text: str) -> int:
    """Estimate tokens for Gemini"""
    return len(text) // 4

# ===================================================================
# RAG WITH TWO-LEVEL CACHING
# ===================================================================

class FinanceRAG:
    """RAG system with two-level semantic caching"""
    
    def __init__(self, api_key: str):
        """Initialize RAG system with two-level semantic caching"""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.vector_store = None
        self.total_embedding_tokens = 0
        self.total_retrieval_tokens = 0
        
        # Initialize two-level cache
        cache_dir = os.path.join(src_dir, "finance_cache")
        self.cache = TwoLevelCache(
            embeddings=self.embeddings,
            cache_dir=cache_dir,
            rag_similarity_threshold=0.95,
            llm_similarity_threshold=0.92,
            logger=finance_logger
        )
    
    def load_vector_store(self, path: str) -> bool:
        """Load FAISS index from disk"""
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            finance_logger.info("✅ Vector store loaded from %s", path)
            return True
        except Exception as e:
            finance_logger.error("❌ Failed to load vector store: %s", e)
            return False
    
    def retrieve(self, query: str, k: int = 3) -> tuple[str, dict]:
        """
        Retrieve relevant context with Level 1 caching
        
        Returns:
            tuple: (context_string, token_stats)
        """
        if not self.vector_store:
            finance_logger.warning("⚠️ Vector store not initialized")
            return "", {"query_tokens": 0, "context_tokens": 0, "embedding_tokens": 0, "cache_hit": False}
        
        finance_logger.info("🔍 Retrieving context for query: %s", query[:100])
        
        # Check RAG cache first
        cache_hit, cached_data = self.cache.get_rag_context(query)
        
        if cache_hit:
            # Return cached result
            cached_stats = cached_data["token_stats"]
            cached_stats["cache_hit"] = True
            
            finance_logger.info("⚡ Using cached RAG result (saved embedding + retrieval)")
            
            return cached_data["context"], cached_stats
        
        # Cache miss - perform actual retrieval
        try:
            # Count query tokens for embedding
            query_tokens = estimate_gemini_tokens(query)
            self.total_embedding_tokens += query_tokens
            
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Format context with sources
            context_parts = []
            for i, doc in enumerate(docs, 1):
                title = doc.metadata.get('title', 'Financial Knowledge Base')
                context_parts.append(f"[Source {i}: {title}]\n{doc.page_content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Count context tokens
            context_tokens = estimate_gemini_tokens(context)
            self.total_retrieval_tokens += context_tokens
            
            token_stats = {
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "embedding_tokens": query_tokens,
                "num_chunks": len(docs),
                "cache_hit": False
            }
            
            finance_logger.info("✅ Retrieved %d chunks | Query: %d tokens | Context: %d tokens", 
                              len(docs), query_tokens, context_tokens)
            
            # Cache the RAG result
            self.cache.set_rag_context(query, context, token_stats)
            
            return context, token_stats
            
        except Exception as e:
            finance_logger.error("❌ Error during retrieval: %s", e)
            return "", {"query_tokens": 0, "context_tokens": 0, "embedding_tokens": 0, "cache_hit": False}
    
    def get_cached_response(self, query: str) -> tuple[bool, str]:
        """
        Check Level 2 cache for LLM response
        
        Returns:
            tuple: (cache_hit: bool, response: str or None)
        """
        return self.cache.get_llm_response(query)
    
    def cache_response(self, query: str, response: str, token_stats: dict):
        """Cache LLM response in Level 2"""
        self.cache.set_llm_response(query, response, token_stats)
    
    def get_cache_stats(self) -> dict:
        """Get combined cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self, cache_type: str = "all"):
        """Clear semantic cache(s)"""
        self.cache.clear(cache_type)


# ===================================================================
# INITIALIZE RAG SYSTEM
# Designed and developed by Mandeep Pahuja
# ===================================================================

# Get API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# ✨ DEFINE PATH FIRST (before using it!)
vector_store_path = os.path.join(src_dir, "finance_faiss_index")

# Initialize RAG with two-level caching
finance_rag = FinanceRAG(api_key)
rag_enabled = False

# Load vector store
if os.path.exists(vector_store_path):
    finance_logger.info("📂 Loading existing vector store from: %s", vector_store_path)
    success = finance_rag.load_vector_store(vector_store_path)
    if success:
        rag_enabled = True
        finance_logger.info("✅ RAG is ENABLED for Finance Agent")
        finance_logger.info("⚡ Two-level semantic caching is ENABLED")
    else:
        finance_logger.warning("⚠️ Failed to load vector store. Agent will work without RAG.")
else:
    finance_logger.warning("⚠️ Vector store not found at: %s", vector_store_path)
    finance_logger.warning("⚠️ Run 'python src/tools/setup_rag.py' to create it")

# ✨ Initialize evaluation runner (only once!)
if EVALUATION_ENABLED:
    eval_runner = EvaluationRunner(
        eval_dir="evaluations/finance",
    #    enable_logging=True
    )
    eval_runner.start()
    finance_logger.info("✅ Evaluation system started")
else:
    eval_runner = None

# Token tracking
total_llm_input_tokens = 0
total_llm_output_tokens = 0
total_requests = 0


# ===================================================================
# FINANCE AGENT FUNCTION
# ===================================================================

def finance_agent(state: MessagesState):
    """Finance education agent with RAG and two-level semantic caching"""
    
    global total_llm_input_tokens, total_llm_output_tokens, total_requests
    
    finance_logger.info("➡️ Entered FINANCE agent")
    finance_logger.info("📊 RAG Status: %s", 'ENABLED' if rag_enabled else 'DISABLED')
    
    # Get user query
    user_message = state["messages"][-1]
    user_query = user_message.content if hasattr(user_message, 'content') else str(user_message)
    finance_logger.info("📝 User query: %s", user_query)
    
    # ============================================================
    # LEVEL 2 CACHE CHECK: Check for cached LLM response FIRST
    # ============================================================
    if rag_enabled:
        finance_logger.info("=" * 80)
        finance_logger.info("🔍 LEVEL 2: Checking LLM Response Cache...")
        finance_logger.info("=" * 80)
        
        cache_hit, cached_response = finance_rag.get_cached_response(user_query)
        
        if cache_hit:
            finance_logger.info("🎯 LEVEL 2 CACHE HIT! Returning cached LLM response")
            finance_logger.info("💰 Savings: Skipped RAG retrieval + LLM generation")
            
            # ✨ NEW: Queue evaluation even for cached responses
        if EVALUATION_ENABLED and eval_runner:
            try:
                eval_runner.queue_evaluation(
                    user_query=user_query,
                    agent_response=cached_response,
                    agent_type="finance",
                    context={
                    "rag_used": rag_enabled,
                    "cache_hit": True,
                    "llm_cache_hit": True
                    }
                )
                finance_logger.info("📊 Evaluation queued (cached response)")
            except Exception as e:
                finance_logger.error(f"❌ Evaluation queue failed: {e}")
            
            # Log cache statistics
            cache_stats = finance_rag.get_cache_stats()
            finance_logger.info("=" * 80)
            finance_logger.info("⚡ CACHE STATISTICS:")
            finance_logger.info("   RAG Cache: %d entries, %.1f%% hit rate", 
                              cache_stats["rag_cache"]["cache_size"],
                              cache_stats["rag_cache"]["hit_rate"])
            finance_logger.info("   LLM Cache: %d entries, %.1f%% hit rate", 
                              cache_stats["llm_cache"]["cache_size"],
                              cache_stats["llm_cache"]["hit_rate"])
            finance_logger.info("=" * 80)
            
            # Return cached response
            response = AIMessage(content=cached_response)
            return {"messages": state["messages"] + [response]}
        
        # If cache_hit but cached_response is None, fall through to normal flow
        if cache_hit:
            finance_logger.warning("⚠️ Cache hit but response is None, proceeding normally")

        finance_logger.info("🔍 LEVEL 2 CACHE MISS - Proceeding to RAG retrieval...")

        #finance_logger.info("🔍 LEVEL 2 CACHE MISS - Proceeding to RAG retrieval...")
    
    # ============================================================
    # LEVEL 1: Retrieve context (with RAG caching)
    # ============================================================
    context = ""
    rag_token_stats = {}
    
    if rag_enabled:
        finance_logger.info("=" * 80)
        finance_logger.info("🔍 LEVEL 1: Retrieving Context...")
        finance_logger.info("=" * 80)
        
        context, rag_token_stats = finance_rag.retrieve(user_query, k=3)
        
        if context:
            cache_status = "⚡ CACHED" if rag_token_stats.get("cache_hit") else "🔍 FRESH"
            finance_logger.info("✅ Context retrieved [%s] (length: %d chars)", 
                              cache_status, len(context))
            
            if not rag_token_stats.get("cache_hit"):
                finance_logger.info("📊 RAG Tokens: Query=%d | Context=%d", 
                                  rag_token_stats.get("query_tokens", 0),
                                  rag_token_stats.get("context_tokens", 0))
            else:
                finance_logger.info("💰 Saved %d embedding tokens from cache",
                                  rag_token_stats.get("embedding_tokens", 0))
    
    # Create system prompt
    if context and rag_enabled:
        system = SystemMessage(content=f"""You are an expert Finance Q&A Agent with access to a comprehensive financial knowledge base.

**INSTRUCTIONS:**
1. Use the CONTEXT below as your PRIMARY source
2. Cite naturally (e.g., "According to financial principles...")
3. Supplement with general knowledge if needed
4. Explain clearly and simply with examples

**CONTEXT:**
{context}

Help users understand investing, personal finance, and money management.""")
        finance_logger.info("✅ Using RAG-enhanced system prompt")
    else:
        system = SystemMessage(content="""You are an expert Finance Q&A Agent. 
Explain financial concepts clearly and simply with examples.""")
        finance_logger.info("ℹ️ Using standard system prompt")
    
    # Count tokens
    system_tokens = estimate_gemini_tokens(system.content)
    user_tokens = estimate_gemini_tokens(user_query)
    history_tokens = sum(estimate_gemini_tokens(msg.content) 
                        for msg in state["messages"][:-1] 
                        if hasattr(msg, 'content'))
    
    total_input_tokens = system_tokens + user_tokens + history_tokens
    
    finance_logger.info("📊 LLM Input: %d tokens (system=%d, user=%d, history=%d)", 
                       total_input_tokens, system_tokens, user_tokens, history_tokens)
    
    # ============================================================
    # LLM GENERATION
    # ============================================================
    finance_logger.info("🤖 Invoking LLM...")
    response = llm.invoke([system] + state["messages"])
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    output_tokens = estimate_gemini_tokens(response_content)
    finance_logger.info("📊 LLM Output: %d tokens", output_tokens)
    
    # ============================================================
    # QUEUE EVALUATION (only once!)
    # ============================================================
    if EVALUATION_ENABLED and eval_runner:
        try:
            eval_runner.queue_evaluation(
                user_query=user_query,
                agent_response=response_content,
                agent_type="finance",
                context={
                    "rag_used": rag_enabled,
                    "rag_context": context if rag_enabled else None,
                    "sources": [context] if rag_enabled and context else [],
                    "num_chunks": rag_token_stats.get("num_chunks", 0) if rag_token_stats else 0,
                    "cache_hit": rag_token_stats.get("cache_hit", False) if rag_token_stats else False
                }
            )
            finance_logger.info("📊 Evaluation queued")
        except Exception as e:
            finance_logger.error(f"❌ Evaluation queue failed: {e}")
    
    # Cache the response
    if rag_enabled:
        finance_logger.info("💾 Caching LLM response...")
        finance_rag.cache_response(
            query=user_query,
            response=response_content,
            token_stats={
                "input_tokens": total_input_tokens,
                "output_tokens": output_tokens,
                "rag_tokens": rag_token_stats
            }
        )
    
    # Update totals
    total_llm_input_tokens += total_input_tokens
    total_llm_output_tokens += output_tokens
    total_requests += 1
    
    # Log statistics
    if rag_enabled:
        cache_stats = finance_rag.get_cache_stats()
        finance_logger.info("=" * 80)
        finance_logger.info("⚡ CACHE STATS:")
        finance_logger.info("   RAG: %d entries (%.1f%% hit rate)", 
                          cache_stats["rag_cache"]["cache_size"],
                          cache_stats["rag_cache"]["hit_rate"])
        finance_logger.info("   LLM: %d entries (%.1f%% hit rate)", 
                          cache_stats["llm_cache"]["cache_size"],
                          cache_stats["llm_cache"]["hit_rate"])
        finance_logger.info("=" * 80)
    
    finance_logger.info("⬅️ Exiting FINANCE agent")
    
    return {"messages": state["messages"] + [response]}


def get_token_stats():
    """Get token usage statistics"""
    cache_stats = finance_rag.get_cache_stats() if rag_enabled else {}
    
    return {
        "total_requests": total_requests,
        "llm_input_tokens": total_llm_input_tokens,
        "llm_output_tokens": total_llm_output_tokens,
        "rag_embedding_tokens": finance_rag.total_embedding_tokens,
        "rag_retrieval_tokens": finance_rag.total_retrieval_tokens,
        "grand_total": (total_llm_input_tokens + total_llm_output_tokens + 
                       finance_rag.total_embedding_tokens + finance_rag.total_retrieval_tokens),
        "cache_stats": cache_stats
    }

def clear_semantic_cache(cache_type: str = "all"):
    """Clear semantic cache(s)"""
    if rag_enabled:
        finance_rag.clear_cache(cache_type)
        finance_logger.info("🗑️ Semantic cache cleared: %s", cache_type)