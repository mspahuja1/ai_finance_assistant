
"""
Multi-Judge Evaluator - Simplified version using FewShotJudge only.

Can be extended later to include hallucination detection.
"""

import asyncio
from typing import Dict, Any, Optional
import logging

from .few_shot_judge import FewShotJudge

logger = logging.getLogger("judge.multi")


class MultiJudgeEvaluator:
    """
    Multi-judge evaluator.
    
    Currently uses FewShotJudge only.
    Can be extended to include HallucinationJudge later.
    """
    
    def __init__(self, **kwargs):
        """Initialize judges."""
        self.few_shot_judge = FewShotJudge(**kwargs)
        logger.info("Initialized MultiJudgeEvaluator (FewShotJudge only)")
    
    async def evaluate_async(
        self,
        user_query: str,
        agent_response: str,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate using few-shot judge.
        
        Args:
            user_query: User's original question
            agent_response: Agent's response to evaluate
            agent_type: Type of agent
            context: Context including sources
            
        Returns:
            Evaluation result
        """
        logger.info(f"Running evaluation for {agent_type} agent")
        
        # Use few-shot judge
        result = await self.few_shot_judge.evaluate_async(
            user_query=user_query,
            agent_response=agent_response,
            agent_type=agent_type,
            context=context
        )
        
        # Add composite score (same as overall for now)
        result['composite_score'] = result.get('overall_score', 3)
        result['judge_type'] = 'multi_judge'
        result['judges_used'] = ['few_shot']
        
        logger.info(
            f"Evaluation complete - Overall: {result['overall_score']}/5"
        )
        
        return result
    
    def evaluate(
        self,
        user_query: str,
        agent_response: str,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for evaluate_async."""
        return asyncio.run(
            self.evaluate_async(user_query, agent_response, agent_type, context)
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics from judges."""
        few_shot_stats = self.few_shot_judge.get_usage_stats()
        
        return {
            "total_calls": few_shot_stats['total_calls'],
            "total_tokens": few_shot_stats['total_tokens'],
            "total_time": few_shot_stats['total_time'],
            "few_shot_judge": few_shot_stats
        }