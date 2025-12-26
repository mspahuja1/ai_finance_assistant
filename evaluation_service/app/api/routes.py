"""
API routes for evaluation microservice.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
import uuid
import time

from app.api.models import (
    EvaluationRequest,
    EvaluationResponse,
    AsyncEvaluationRequest,
    AsyncEvaluationResponse,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    StatisticsResponse,
    JudgeType
)
from app.services.evaluation_service import EvaluationService
from app.main import get_evaluation_service

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    request: EvaluationRequest,
    service: EvaluationService = Depends(get_evaluation_service)
):
    """
    Synchronous evaluation endpoint.
    
    Evaluates the response immediately and returns results.
    Use this for interactive/real-time evaluation.
    """
    evaluation_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Perform evaluation
        result = await service.evaluate(
            user_query=request.user_query,
            agent_response=request.agent_response,
            agent_type=request.agent_type,
            judge_type=request.judge_type,
            context=request.context
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return EvaluationResponse(
            evaluation_id=evaluation_id,
            status="completed",
            composite_score=result.get("composite_score"),
            overall_score=result.get("overall_score"),
            hallucination_score=result.get("hallucination_score"),
            accuracy_score=result.get("accuracy_score"),
            completeness_score=result.get("completeness_score"),
            clarity_score=result.get("clarity_score"),
            safety_score=result.get("safety_score"),
            grounding_rate=result.get("grounding_rate"),
            hallucination_rate=result.get("hallucination_rate"),
            risk_level=result.get("overall_risk"),
            strengths=result.get("strengths"),
            weaknesses=result.get("weaknesses"),
            recommendations=result.get("recommendations"),
            explanation=result.get("quality_explanation"),
            evaluation_timestamp=result.get("evaluation_timestamp"),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/evaluate/async", response_model=AsyncEvaluationResponse)
async def evaluate_async(
    request: AsyncEvaluationRequest,
    background_tasks: BackgroundTasks,
    service: EvaluationService = Depends(get_evaluation_service)
):
    """
    Asynchronous evaluation endpoint.
    
    Queues the evaluation for background processing.
    Use this for batch processing or when immediate results aren't needed.
    """
    evaluation_id = str(uuid.uuid4())
    
    # Queue the evaluation
    await service.queue_evaluation(
        evaluation_id=evaluation_id,
        user_query=request.user_query,
        agent_response=request.agent_response,
        agent_type=request.agent_type,
        judge_type=request.judge_type,
        context=request.context,
        callback_url=request.callback_url,
        priority=request.priority
    )
    
    return AsyncEvaluationResponse(
        evaluation_id=evaluation_id,
        status="queued",
        message="Evaluation queued for processing",
        estimated_time_seconds=5,
        status_url=f"/api/v1/evaluation/{evaluation_id}"
    )


@router.post("/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(
    request: BatchEvaluationRequest,
    service: EvaluationService = Depends(get_evaluation_service)
):
    """
    Batch evaluation endpoint.
    
    Evaluates multiple requests in parallel.
    """
    batch_id = request.batch_id or str(uuid.uuid4())
    evaluation_ids = []
    
    for eval_request in request.evaluations:
        eval_id = str(uuid.uuid4())
        evaluation_ids.append(eval_id)
        
        await service.queue_evaluation(
            evaluation_id=eval_id,
            user_query=eval_request.user_query,
            agent_response=eval_request.agent_response,
            agent_type=eval_request.agent_type,
            judge_type=eval_request.judge_type,
            context=eval_request.context
        )
    
    return BatchEvaluationResponse(
        batch_id=batch_id,
        total_evaluations=len(evaluation_ids),
        status="queued",
        evaluation_ids=evaluation_ids
    )


@router.get("/evaluation/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: str,
    service: EvaluationService = Depends(get_evaluation_service)
):
    """
    Get evaluation result by ID.
    
    Returns the evaluation result if completed, or status if still processing.
    """
    result = await service.get_evaluation_result(evaluation_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return EvaluationResponse(
        evaluation_id=evaluation_id,
        status=result.get("status", "completed"),
        **result
    )


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    service: EvaluationService = Depends(get_evaluation_service)
):
    """
    Get evaluation statistics.
    
    Returns aggregate statistics about evaluations performed.
    """
    stats = await service.get_statistics()
    
    return StatisticsResponse(**stats)


@router.delete("/evaluation/{evaluation_id}")
async def delete_evaluation(
    evaluation_id: str,
    service: EvaluationService = Depends(get_evaluation_service)
):
    """Delete an evaluation result."""
    success = await service.delete_evaluation(evaluation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return {"message": "Evaluation deleted successfully"}