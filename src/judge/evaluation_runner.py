"""
Evaluation Runner - Manages async evaluation queue.

Uses threading to work properly with Streamlit.
"""

import threading
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import queue

from .multi_judge_evaluator import MultiJudgeEvaluator

logger = logging.getLogger("judge.runner")


class EvaluationRunner:
    """
    Manages async evaluation queue using background thread.
    
    Compatible with Streamlit and other async environments.
    """
    
    def __init__(self, eval_dir: str = "evaluations/finance"):
        """
        Initialize evaluation runner.
        
        Args:
            eval_dir: Directory to save evaluation results
        """
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize judge
        self.judge = MultiJudgeEvaluator()
        
        # Use thread-safe queue (not asyncio.Queue)
        self.eval_queue = queue.Queue()
        
        # Background worker
        self.worker_thread = None
        self.is_running = False
        
        logger.info(f"EvaluationRunner initialized: {self.eval_dir}")
    
    def start(self):
        """Start background worker thread."""
        if self.is_running:
            logger.warning("Worker already running")
            return
        
        self.is_running = True
        
        # Start worker in background thread
        self.worker_thread = threading.Thread(
            target=self._worker_thread,
            daemon=True,
            name="EvaluationWorker"
        )
        self.worker_thread.start()
        
        logger.info("✅ Evaluation worker thread started")
    
    def stop(self):
        """Stop background worker."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        logger.info("✅ Evaluation worker stopped")
    
    def queue_evaluation(
        self,
        user_query: str,
        agent_response: str,
        agent_type: str = "finance",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Queue an evaluation (non-blocking).
        
        Args:
            user_query: User's question
            agent_response: Agent's response
            agent_type: Type of agent
            context: Additional context
        """
        eval_data = {
            "user_query": user_query,
            "agent_response": agent_response,
            "agent_type": agent_type,
            "context": context,
            "queued_at": datetime.now().isoformat()
        }
        
        try:
            self.eval_queue.put(eval_data, block=False)
            logger.info(f"📊 Evaluation queued (queue size: {self.eval_queue.qsize()})")
        except queue.Full:
            logger.error("Evaluation queue is full!")
        except Exception as e:
            logger.error(f"Failed to queue evaluation: {e}")
    
    def _worker_thread(self):
        """Background worker thread that processes evaluations."""
        logger.info("🔧 Worker thread started, entering processing loop...")
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.is_running:
            try:
                # Get next evaluation (with timeout to allow checking is_running)
                try:
                    eval_data = self.eval_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                logger.info(f"🔧 Worker processing evaluation...")
                
                # Perform evaluation
                try:
                    result = loop.run_until_complete(
                        self.judge.evaluate_async(
                            user_query=eval_data["user_query"],
                            agent_response=eval_data["agent_response"],
                            agent_type=eval_data["agent_type"],
                            context=eval_data["context"]
                        )
                    )
                    
                    # Save result
                    self._save_evaluation(result)
                    
                    logger.info(
                        f"✅ Evaluation complete - "
                        f"Score: {result.get('composite_score', result.get('overall_score', 'N/A'))}/5"
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Evaluation error: {e}", exc_info=True)
                
            except Exception as e:
                logger.error(f"❌ Worker error: {e}", exc_info=True)
        
        loop.close()
        logger.info("🔧 Worker thread stopped")
    
    def _save_evaluation(self, result: Dict[str, Any]):
        """Save evaluation result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"eval_{result.get('agent_type', 'unknown')}_{timestamp}.json"
        filepath = self.eval_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"💾 Saved evaluation: {filename}")
        except Exception as e:
            logger.error(f"Failed to save evaluation: {e}")
    
    def get_recent_evaluations(self, limit: int = 10) -> list:
        """Get recent evaluations from disk."""
        eval_files = sorted(
            self.eval_dir.glob("eval_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        results = []
        for filepath in eval_files[:limit]:
            try:
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        evaluations = self.get_recent_evaluations(limit=1000)
        
        if not evaluations:
            return {
                "total_evaluations": 0,
                "queue_size": self.eval_queue.qsize(),
                "is_running": self.is_running
            }
        
        # Calculate stats
        total = len(evaluations)
        scores = [e.get('composite_score', e.get('overall_score', 0)) for e in evaluations]
        avg_score = sum(scores) / total if scores else 0
        
        return {
            "total_evaluations": total,
            "average_score": round(avg_score, 2),
            "queue_size": self.eval_queue.qsize(),
            "is_running": self.is_running,
            "worker_alive": self.worker_thread.is_alive() if self.worker_thread else False
        }