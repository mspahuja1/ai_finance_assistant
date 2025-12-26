"""
Core evaluation service with queue management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

# Import judge from parent project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from judge.few_shot_judge import FewShotJudge
from judge.hallucination_judge import HallucinationJudge
from judge.multi_judge_evaluator import MultiJudgeEvaluator

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Core evaluation service.
    
    Manages:
    - Multiple judge instances
    - Async evaluation queue
    - Result storage and retrieval
    """
    
    def __init__(self):
        """Initialize evaluation service."""
        # Initialize judges
        self.few_shot_judge = FewShotJudge()
        self.hallucination_judge = HallucinationJudge()
        self.multi_judge = MultiJudgeEvaluator()
        
        # Queue and storage
        self.evaluation_queue = asyncio.PriorityQueue()
        self.results_store: Dict[str, Dict[str, Any]] = {}
        self.processing_tasks = []
        
        # Worker control
        self.is_running = False
        self.num_workers = 3  # Number of parallel workers
        
        # Statistics
        self.total_evaluations = 0
        self.evaluations_by_agent = {}
        
        logger.info("EvaluationService initialized")
    
    async def start(self):
        """Start evaluation workers."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker(worker_id=i))
            self.processing_tasks.append(task)
        
        logger.info(f"Started {self.num_workers} evaluation workers")
    
    async def stop(self):
        """Stop evaluation workers."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all workers
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        logger.info("Stopped all evaluation workers")
    
    async def evaluate(
        self,
        user_query: str,
        agent_response: str,
        agent_type: str,
        judge_type: str = "multi",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform synchronous evaluation.
        
        Returns results immediately.
        """
        logger.info(f"Evaluating {agent_type} response with {judge_type} judge")
        
        # Select judge
        if judge_type == "few_shot":
            judge = self.few_shot_judge
        elif judge_type == "hallucination":
            judge = self.hallucination_judge
        else:  # multi
            judge = self.multi_judge
        
        # Perform evaluation
        result = await judge.evaluate_async(
            user_query=user_query,
            agent_response=agent_response,
            agent_type=agent_type,
            context=context
        )
        
        # Update statistics
        self.total_evaluations += 1
        self.evaluations_by_agent[agent_type] = \
            self.evaluations_by_agent.get(agent_type, 0) + 1
        
        return result
    
    async def queue_evaluation(
        self,
        evaluation_id: str,
        user_query: str,
        agent_response: str,
        agent_type: str,
        judge_type: str = "multi",
        context: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
        priority: int = 5
    ):
        """
        Queue an evaluation for async processing.
        
        Lower priority number = higher priority (1 is highest).
        """
        eval_data = {
            "evaluation_id": evaluation_id,
            "user_query": user_query,
            "agent_response": agent_response,
            "agent_type": agent_type,
            "judge_type": judge_type,
            "context": context,
            "callback_url": callback_url,
            "queued_at": datetime.now().isoformat()
        }
        
        # Add to priority queue (lower number = higher priority)
        await self.evaluation_queue.put((priority, evaluation_id, eval_data))
        
        # Mark as pending in results store
        self.results_store[evaluation_id] = {
            "status": "queued",
            "queued_at": eval_data["queued_at"]
        }
        
        logger.info(f"Queued evaluation {evaluation_id} with priority {priority}")
    
    async def _worker(self, worker_id: int):
        """Background worker to process evaluation queue."""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get next evaluation from queue (timeout to allow shutdown)
                priority, eval_id, eval_data = await asyncio.wait_for(
                    self.evaluation_queue.get(),
                    timeout=1.0
                )
                
                logger.info(f"Worker {worker_id} processing evaluation {eval_id}")
                
                # Update status
                self.results_store[eval_id]["status"] = "processing"
                self.results_store[eval_id]["started_at"] = datetime.now().isoformat()
                
                # Perform evaluation
                try:
                    result = await self.evaluate(
                        user_query=eval_data["user_query"],
                        agent_response=eval_data["agent_response"],
                        agent_type=eval_data["agent_type"],
                        judge_type=eval_data["judge_type"],
                        context=eval_data["context"]
                    )
                    
                    # Store result
                    self.results_store[eval_id] = {
                        "status": "completed",
                        **result,
                        "completed_at": datetime.now().isoformat()
                    }
                    
                    # Call callback if provided
                    if eval_data.get("callback_url"):
                        await self._send_callback(
                            eval_data["callback_url"],
                            eval_id,
                            result
                        )
                    
                    logger.info(f"Worker {worker_id} completed evaluation {eval_id}")
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error processing {eval_id}: {e}")
                    self.results_store[eval_id] = {
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat()
                    }
                
            except asyncio.TimeoutError:
                # No items in queue, continue loop
                continue
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} unexpected error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _send_callback(
        self,
        callback_url: str,
        evaluation_id: str,
        result: Dict[str, Any]
    ):
        """Send callback with evaluation result."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "evaluation_id": evaluation_id,
                    "result": result
                }
                async with session.post(callback_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Callback sent successfully for {evaluation_id}")
                    else:
                        logger.warning(f"Callback failed for {evaluation_id}: {response.status}")
        except Exception as e:
            logger.error(f"Error sending callback for {evaluation_id}: {e}")
    
    async def get_evaluation_result(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation result by ID."""
        return self.results_store.get(evaluation_id)
    
    async def delete_evaluation(self, evaluation_id: str) -> bool:
        """Delete evaluation result."""
        if evaluation_id in self.results_store:
            del self.results_store[evaluation_id]
            return True
        return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "evaluations_by_agent": self.evaluations_by_agent,
            "queue_size": self.evaluation_queue.qsize(),
            "pending_evaluations": len([
                r for r in self.results_store.values()
                if r.get("status") in ["queued", "processing"]
            ]),
            "completed_evaluations": len([
                r for r in self.results_store.values()
                if r.get("status") == "completed"
            ]),
            "processing_stats": {
                "num_workers": self.num_workers,
                "is_running": self.is_running
            }
        }
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.evaluation_queue.qsize()