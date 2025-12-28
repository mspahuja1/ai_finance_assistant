"""
Async Feedback System - Zero latency impact on user responses.

Processes quality checks and performance metrics in background thread.
"""

import threading
import queue
import time
from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AsyncFeedbackProcessor:
    """Process feedback operations in background thread."""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        
        # Create feedback directories
        self.feedback_dir = Path("feedback")
        self.quality_scores_dir = self.feedback_dir / "quality_scores"
        self.performance_metrics_dir = self.feedback_dir / "performance_metrics"
        
        for dir in [self.feedback_dir, self.quality_scores_dir, self.performance_metrics_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Load performance metrics
        self.performance_metrics = self._load_performance_metrics()
    
    def start(self):
        """Start background worker."""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("✅ Async feedback processor started")
    
    def stop(self):
        """Stop background worker."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def _worker(self):
        """Background worker thread."""
        logger.info("🔧 Feedback worker thread started")
        
        while self.is_running:
            try:
                task = self.queue.get(timeout=1.0)
                
                task_type = task["type"]
                
                if task_type == "quality_check":
                    self._process_quality_check(task)
                elif task_type == "performance_update":
                    self._process_performance_update(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Feedback processing error: {e}")
    
    # ========================================================================
    # QUALITY CHECK
    # ========================================================================
    
    def _process_quality_check(self, task: Dict[str, Any]):
        """Process quality check in background."""
        query = task["query"]
        response = task["response"]
        agent_type = task["agent_type"]
        
        # Simple quality scoring
        score = self._calculate_quality_score(response, agent_type)
        
        # Store quality score
        self._store_quality_score(
            query=query,
            response=response,
            agent_type=agent_type,
            quality_score=score,
            retry_count=task.get("retry_count", 0)
        )
        
        logger.debug(f"📊 Background quality check: {score:.2f}/5 for {agent_type}")
    
    def _calculate_quality_score(self, response: str, agent_type: str) -> float:
        """Fast quality scoring heuristic."""
        score = 5.0
        
        # Length check
        if len(response) < 50:
            score -= 2.0
        elif len(response) < 100:
            score -= 0.5
        
        # Unhelpful phrases
        response_lower = response.lower()
        if "i don't know" in response_lower and len(response) < 150:
            score -= 1.5
        
        # Has examples or numbers (good)
        if any(word in response_lower for word in ["example", "for instance", "such as"]):
            score += 0.3
        
        if any(char.isdigit() for char in response):
            score += 0.2
        
        # Safety check for tax/portfolio
        if agent_type in ["tax", "portfolio"]:
            has_disclaimer = any(
                word in response_lower 
                for word in ["consult", "professional", "advisor", "disclaimer"]
            )
            if not has_disclaimer:
                score -= 1.0
        
        return max(0.0, min(5.0, score))
    
    def _store_quality_score(
        self,
        query: str,
        response: str,
        agent_type: str,
        quality_score: float,
        retry_count: int = 0
    ):
        """Store quality score to disk."""
        timestamp = datetime.now()
        score_id = f"quality_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        score_data = {
            "score_id": score_id,
            "timestamp": timestamp.isoformat(),
            "query": query[:200],  # Truncate
            "response": response[:500],  # Truncate
            "agent_type": agent_type,
            "quality_score": round(quality_score, 2),
            "retry_count": retry_count,
            "date": timestamp.strftime("%Y-%m-%d")
        }
        
        filepath = self.quality_scores_dir / f"{score_id}.json"
        with open(filepath, 'w') as f:
            json.dump(score_data, f, indent=2)
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    def _load_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load performance metrics from disk."""
        metrics_file = self.performance_metrics_dir / "current_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
        
        # Default metrics
        return {
            "finance": {"success_rate": 0.95, "avg_latency": 2.3, "avg_score": 4.5},
            "market": {"success_rate": 0.90, "avg_latency": 5.1, "avg_score": 4.2},
            "goal": {"success_rate": 0.92, "avg_latency": 1.8, "avg_score": 4.3},
            "news": {"success_rate": 0.88, "avg_latency": 5.5, "avg_score": 4.0},
            "portfolio": {"success_rate": 0.91, "avg_latency": 2.0, "avg_score": 4.4},
            "tax": {"success_rate": 0.93, "avg_latency": 2.1, "avg_score": 4.3}
        }
    
    def _save_performance_metrics(self):
        """Save performance metrics to disk."""
        metrics_file = self.performance_metrics_dir / "current_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
    
    def _process_performance_update(self, task: Dict[str, Any]):
        """Update performance metrics in background."""
        agent_type = task["agent_type"]
        success = task["success"]
        latency = task["latency"]
        quality_score = task.get("quality_score")
        
        if agent_type not in self.performance_metrics:
            self.performance_metrics[agent_type] = {
                "success_rate": 0.90,
                "avg_latency": 2.0,
                "avg_score": 4.0
            }
        
        metrics = self.performance_metrics[agent_type]
        
        # Exponential moving average (alpha=0.1)
        alpha = 0.1
        
        metrics["success_rate"] = (metrics["success_rate"] * (1 - alpha)) + (float(success) * alpha)
        metrics["avg_latency"] = (metrics["avg_latency"] * (1 - alpha)) + (latency * alpha)
        
        if quality_score is not None:
            metrics["avg_score"] = (metrics.get("avg_score", 4.0) * (1 - alpha)) + (quality_score * alpha)
        
        self._save_performance_metrics()
        
        logger.debug(f"📈 Updated metrics for {agent_type}: "
                    f"success={metrics['success_rate']:.2f}, "
                    f"latency={metrics['avg_latency']:.2f}s, "
                    f"score={metrics.get('avg_score', 0):.2f}")
    
    # ========================================================================
    # PUBLIC API - Non-blocking
    # ========================================================================
    
    def queue_quality_check(
        self,
        query: str,
        response: str,
        agent_type: str,
        retry_count: int = 0
    ):
        """
        Queue quality check (non-blocking, <1ms).
        
        Args:
            query: User query
            response: Agent response
            agent_type: Which agent
            retry_count: Number of retries performed
        """
        self.queue.put({
            "type": "quality_check",
            "query": query,
            "response": response,
            "agent_type": agent_type,
            "retry_count": retry_count
        })
    
    def queue_performance_update(
        self,
        agent_type: str,
        success: bool,
        latency: float,
        quality_score: Optional[float] = None
    ):
        """
        Queue performance update (non-blocking, <1ms).
        
        Args:
            agent_type: Which agent
            success: Whether execution succeeded
            latency: Response latency in seconds
            quality_score: Optional quality score
        """
        self.queue.put({
            "type": "performance_update",
            "agent_type": agent_type,
            "success": success,
            "latency": latency,
            "quality_score": quality_score
        })
    
    def get_performance_metrics(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Args:
            agent_type: Specific agent or None for all
            
        Returns:
            Performance metrics
        """
        if agent_type:
            return self.performance_metrics.get(agent_type, {})
        
        return self.performance_metrics


# Global instance
async_feedback = AsyncFeedbackProcessor()
async_feedback.start()