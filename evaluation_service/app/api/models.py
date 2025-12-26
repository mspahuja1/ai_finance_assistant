"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class JudgeType(str, Enum):
    """Available judge types."""
    FEW_SHOT = "few_shot"
    HALLUCINATION = "hallucination"
    MULTI = "multi"


class EvaluationRequest(BaseModel):
    """Request model for evaluation."""
    user_query: str = Field(..., description="User's original question")
    agent_response: str = Field(..., description="Agent's response to evaluate")
    agent_type: str = Field(..., description="Type of agent (e.g., finance, market)")
    judge_type: JudgeType = Field(default=JudgeType.MULTI, description="Type of judge to use")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_query": "What is compound interest?",
                "agent_response": "Compound interest is interest on interest...",
                "agent_type": "finance",
                "judge_type": "multi",
                "context": {
                    "rag_used": True,
                    "sources": ["Wikipedia article on compound interest..."]
                }
            }
        }


class EvaluationResponse(BaseModel):
    """Response model for evaluation."""
    evaluation_id: str = Field(..., description="Unique evaluation ID")
    status: str = Field(..., description="Status: completed, pending, failed")
    
    # Scores
    composite_score: Optional[int] = None
    overall_score: Optional[int] = None
    hallucination_score: Optional[int] = None
    accuracy_score: Optional[int] = None
    completeness_score: Optional[int] = None
    clarity_score: Optional[int] = None
    safety_score: Optional[int] = None
    
    # Analysis
    grounding_rate: Optional[float] = None
    hallucination_rate: Optional[float] = None
    risk_level: Optional[str] = None
    
    # Details
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    explanation: Optional[str] = None
    
    # Metadata
    evaluation_timestamp: Optional[str] = None
    processing_time_ms: Optional[int] = None


class AsyncEvaluationRequest(EvaluationRequest):
    """Request model for async evaluation."""
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when complete"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority (1=lowest, 10=highest)"
    )


class AsyncEvaluationResponse(BaseModel):
    """Response for async evaluation request."""
    evaluation_id: str
    status: str = "queued"
    message: str
    estimated_time_seconds: int
    status_url: str


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation."""
    evaluations: List[EvaluationRequest] = Field(
        ...,
        description="List of evaluations to perform"
    )
    batch_id: Optional[str] = Field(
        default=None,
        description="Optional batch identifier"
    )


class BatchEvaluationResponse(BaseModel):
    """Response for batch evaluation."""
    batch_id: str
    total_evaluations: int
    status: str
    evaluation_ids: List[str]


class StatisticsResponse(BaseModel):
    """Response model for statistics."""
    total_evaluations: int
    evaluations_by_agent: Dict[str, int]
    average_scores: Dict[str, float]
    recent_evaluations: int
    queue_size: int
    processing_stats: Dict[str, Any]