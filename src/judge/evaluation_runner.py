"""
Async Evaluation Runner for AI Finance Assistant.

Runs evaluations in background without impacting user-facing performance.
Logs results and provides analytics.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import logging

from .few_shot_judge import FewShotJudge

logger = logging.getLogger("judge.runner")


class EvaluationRunner:
    """
    Async evaluation runner that evaluates responses in background.
    
    Queues evaluations and processes them asynchronously to avoid
    impacting user-facing workflow performance.
    """
    
    def __init__(
        self,
        eval_dir: str = "evaluations",
        batch_size: int = 5,
        enable_logging: bool = True
    ):
        """
        Initialize evaluation runner.
        
        Args:
            eval_dir: Directory to store evaluation results
            batch_size: Number of evaluations to process in parallel
            enable_logging: Whether to log evaluations to files
        """
        self.eval_dir = Path(eval_dir)
        self.batch_size = batch_size
        self.enable_logging = enable_logging
        
        # Create evaluation directory
        if self.enable_logging:
            self.eval_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Evaluation results will be saved to: {self.eval_dir}")
        
        # Initialize judge
        self.judge = FewShotJudge()
        
        # Evaluation queue
        self.eval_queue = asyncio.Queue()
        self.is_running = False
        self.worker_task = None
        
        # Statistics
        self.total_evaluations = 0
        self.evaluations_by_agent = {}
        self.average_scores = {}
        
        logger.info("EvaluationRunner initialized")
    
    def start(self):
        """Start the async evaluation worker."""
        if not self.is_running:
            self.is_running = True
            self.worker_task = asyncio.create_task(self._process_queue())
            logger.info("Evaluation worker started")
    
    def stop(self):
        """Stop the async evaluation worker."""
        if self.is_running:
            self.is_running = False
            if self.worker_task:
                self.worker_task.cancel()
            logger.info("Evaluation worker stopped")
    
    def queue_evaluation(
        self,
        user_query: str,
        agent_response: str,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ):
        """
        Queue an evaluation to be processed asynchronously.
        
        This is non-blocking and returns immediately.
        
        Args:
            user_query: User's question
            agent_response: Agent's response
            agent_type: Type of agent
            context: Additional context
            session_id: Optional session identifier
        """
        eval_data = {
            "user_query": user_query,
            "agent_response": agent_response,
            "agent_type": agent_type,
            "context": context or {},
            "session_id": session_id,
            "queued_at": datetime.now().isoformat()
        }
        
        # Queue for async processing
        asyncio.create_task(self.eval_queue.put(eval_data))
        
        logger.debug(f"Queued evaluation for {agent_type} agent")
    
    async def evaluate_now(
        self,
        user_query: str,
        agent_response: str,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate immediately (blocking async call).
        
        Use this when you need the evaluation result right away.
        
        Args:
            user_query: User's question
            agent_response: Agent's response
            agent_type: Type of agent
            context: Additional context
            
        Returns:
            Evaluation result
        """
        result = await self.judge.evaluate_async(
            user_query=user_query,
            agent_response=agent_response,
            agent_type=agent_type,
            context=context
        )
        
        # Log if enabled
        if self.enable_logging:
            self._save_evaluation(result)
        
        # Update statistics
        self._update_statistics(agent_type, result)
        
        return result
    
    async def _process_queue(self):
        """Process queued evaluations in background."""
        logger.info("Starting evaluation queue processor")
        
        while self.is_running:
            try:
                # Get batch of evaluations
                batch = []
                for _ in range(self.batch_size):
                    try:
                        eval_data = await asyncio.wait_for(
                            self.eval_queue.get(),
                            timeout=1.0
                        )
                        batch.append(eval_data)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process batch in parallel
                tasks = [
                    self.judge.evaluate_async(
                        user_query=item["user_query"],
                        agent_response=item["agent_response"],
                        agent_type=item["agent_type"],
                        context=item["context"]
                    )
                    for item in batch
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Save results
                for eval_data, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Evaluation failed: {result}")
                        continue
                    
                    # Add session info
                    if eval_data.get("session_id"):
                        result["session_id"] = eval_data["session_id"]
                    
                    # Save and update stats
                    if self.enable_logging:
                        self._save_evaluation(result)
                    
                    self._update_statistics(eval_data["agent_type"], result)
                
                logger.info(f"Processed batch of {len(batch)} evaluations")
                
            except asyncio.CancelledError:
                logger.info("Evaluation queue processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}", exc_info=True)
                await asyncio.sleep(1.0)
    
    def _save_evaluation(self, result: Dict[str, Any]):
        """Save evaluation result to file."""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            agent_type = result.get("agent_type", "unknown")
            filename = f"eval_{agent_type}_{timestamp}.json"
            filepath = self.eval_dir / filename
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.debug(f"Saved evaluation to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation: {e}")
    
    def _update_statistics(self, agent_type: str, result: Dict[str, Any]):
        """Update running statistics."""
        self.total_evaluations += 1
        
        # Count by agent type
        if agent_type not in self.evaluations_by_agent:
            self.evaluations_by_agent[agent_type] = 0
        self.evaluations_by_agent[agent_type] += 1
        
        # Track average scores
        overall_score = result.get("overall_score", 0)
        if agent_type not in self.average_scores:
            self.average_scores[agent_type] = {
                "total_score": 0,
                "count": 0,
                "average": 0
            }
        
        stats = self.average_scores[agent_type]
        stats["total_score"] += overall_score
        stats["count"] += 1
        stats["average"] = stats["total_score"] / stats["count"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_evaluations": self.total_evaluations,
            "evaluations_by_agent": self.evaluations_by_agent,
            "average_scores_by_agent": {
                agent: scores["average"]
                for agent, scores in self.average_scores.items()
            },
            "judge_stats": self.judge.get_usage_stats()
        }
    
    def get_recent_evaluations(
        self,
        agent_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent evaluations from disk.
        
        Args:
            agent_type: Filter by agent type (optional)
            limit: Maximum number to return
            
        Returns:
            List of evaluation results
        """
        if not self.enable_logging:
            return []
        
        try:
            # Get all evaluation files
            eval_files = sorted(
                self.eval_dir.glob("eval_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            evaluations = []
            for filepath in eval_files:
                if len(evaluations) >= limit:
                    break
                
                try:
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                    
                    # Filter by agent type if specified
                    if agent_type and result.get("agent_type") != agent_type:
                        continue
                    
                    evaluations.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
                    continue
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Failed to get recent evaluations: {e}")
            return []
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate evaluation summary report.
        
        Args:
            output_file: Optional file to save report to
            
        Returns:
            Report as formatted string
        """
        stats = self.get_statistics()
        
        report = []
        report.append("=" * 70)
        report.append("AI FINANCE ASSISTANT - EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Evaluations: {stats['total_evaluations']}")
        report.append("")
        
        report.append("EVALUATIONS BY AGENT TYPE:")
        report.append("-" * 70)
        for agent, count in stats['evaluations_by_agent'].items():
            avg_score = stats['average_scores_by_agent'].get(agent, 0)
            report.append(f"{agent:15s}: {count:4d} evaluations, avg score: {avg_score:.2f}/5")
        report.append("")
        
        report.append("JUDGE PERFORMANCE:")
        report.append("-" * 70)
        judge_stats = stats['judge_stats']
        report.append(f"Total LLM Calls: {judge_stats['total_calls']}")
        report.append(f"Total Tokens: {judge_stats['total_tokens']:,}")
        report.append(f"Avg Time per Call: {judge_stats['avg_time_per_call']:.2f}s")
        report.append(f"Avg Tokens per Call: {judge_stats['avg_tokens_per_call']:.0f}")
        report.append("")
        
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report_text


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_evaluation():
        """Test the evaluation system."""
        
        # Initialize runner
        runner = EvaluationRunner(eval_dir="test_evaluations")
        
        # Test cases
        test_cases = [
            {
                "query": "What is compound interest?",
                "response": """Compound interest is interest calculated on both the principal and accumulated interest. 
                For example, $1000 at 5% for 2 years: Year 1: $1050, Year 2: $1102.50. 
                Formula: A = P(1 + r)^t""",
                "agent_type": "finance"
            },
            {
                "query": "Should I invest all my money in Bitcoin?",
                "response": "Yes, absolutely! Bitcoin is guaranteed to go up!",
                "agent_type": "finance"
            }
        ]
        
        print("Running evaluations...")
        
        for case in test_cases:
            result = await runner.evaluate_now(
                user_query=case["query"],
                agent_response=case["response"],
                agent_type=case["agent_type"]
            )
            
            print(f"\nQuery: {case['query']}")
            print(f"Overall Score: {result['overall_score']}/5")
            print(f"Accuracy: {result['accuracy_score']}/5")
            print(f"Safety: {result['safety_score']}/5")
            print(f"Explanation: {result['explanation'][:100]}...")
            print("-" * 60)
        
        # Generate report
        report = runner.generate_report()
        print("\n" + report)
    
    # Run test
    asyncio.run(test_evaluation())