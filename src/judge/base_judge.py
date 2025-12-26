"""
Base Judge class for AI Finance Assistant evaluation.

Compatible with LangGraph 1.0+ and LangChain 1.0+
Uses the same LangChain components as your main workflow.
Designed and developed by Mandee Pahuja
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import logging

# LangChain components (same as your workflow)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Configure logging
logger = logging.getLogger("judge")
logger.setLevel(logging.INFO)


class BaseJudge:
    """
    Base class for LLM-as-a-Judge evaluation.
    
    Compatible with LangGraph 1.0+ and LangChain 1.0+
    Uses ChatGoogleGenerativeAI (same as your workflow).
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize base judge.
        
        Args:
            model_name: Google Gemini model to use
            temperature: Sampling temperature (lower = more consistent)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get API key
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize ChatGoogleGenerativeAI (same as your workflow)
        # This works with both LangChain 0.1.x and 1.0+
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=self.api_key
            )
        except TypeError:
            # Fallback for older versions that might use different param names
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key
            )
        
        # Usage tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        
        # Judge metadata
        self.judge_name = "Base Judge"
        self.technique = "Basic Evaluation"
        
        logger.info(
            f"Initialized {self.judge_name} with {model_name} "
            f"(LangChain + LangGraph compatible)"
        )
    
    async def evaluate_async(
        self,
        user_query: str,
        agent_response: str,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously evaluate agent response.
        
        Args:
            user_query: User's original question
            agent_response: Agent's response to evaluate
            agent_type: Type of agent (finance, market, news, etc.)
            context: Additional context (RAG sources, tools used, etc.)
            
        Returns:
            Evaluation dictionary with scores and feedback
        """
        raise NotImplementedError("Subclasses must implement evaluate_async")
    
    def evaluate(
        self,
        user_query: str,
        agent_response: str,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_async.
        
        Args:
            user_query: User's original question
            agent_response: Agent's response to evaluate
            agent_type: Type of agent
            context: Additional context
            
        Returns:
            Evaluation dictionary
        """
        return asyncio.run(
            self.evaluate_async(user_query, agent_response, agent_type, context)
        )
    
    async def _call_llm_async(self, messages: list) -> str:
        """
        Asynchronously call LLM using LangChain.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Model response text
        """
        start_time = time.time()
        
        try:
            # Convert to LangChain message format
            lc_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "user":
                    lc_messages.append(HumanMessage(content=content))
            
            # Call LLM asynchronously (works with LangChain 0.1+ and 1.0+)
            response = await asyncio.to_thread(
                self.llm.invoke,
                lc_messages
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Update statistics
            self.total_calls += 1
            elapsed = time.time() - start_time
            self.total_time += elapsed
            
            # Estimate tokens
            estimated_tokens = sum(len(msg.content) for msg in lc_messages) // 4 + len(response_text) // 4
            self.total_tokens += estimated_tokens
            
            logger.debug(f"LLM call completed in {elapsed:.2f}s, ~{estimated_tokens} tokens")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"Error during evaluation: {str(e)}"
    
    def _call_llm(self, messages: list) -> str:
        """
        Synchronous wrapper for _call_llm_async.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Model response text
        """
        return asyncio.run(self._call_llm_async(messages))
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with call count, tokens, and timing stats
        """
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0
        avg_tokens = self.total_tokens / self.total_calls if self.total_calls > 0 else 0
        
        return {
            "judge_name": self.judge_name,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_time_per_call": avg_time,
            "avg_tokens_per_call": avg_tokens,
            "framework": "LangChain + LangGraph"
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        logger.info(f"{self.judge_name} statistics reset")