"""
Evaluation Microservice - FastAPI Application

Provides REST API for LLM-as-a-Judge evaluation.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api import routes
from app.core.config import settings
from app.services.evaluation_service import EvaluationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global evaluation service instance
evaluation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global evaluation_service
    
    # Startup
    logger.info("Starting Evaluation Microservice...")
    evaluation_service = EvaluationService()
    await evaluation_service.start()
    logger.info("✅ Evaluation service started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Evaluation Microservice...")
    await evaluation_service.stop()
    logger.info("✅ Evaluation service stopped")


# Create FastAPI app
app = FastAPI(
    title="LLM Evaluation Microservice",
    description="Multi-judge LLM evaluation service with hallucination detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LLM Evaluation Microservice",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "evaluation",
        "queue_size": evaluation_service.get_queue_size() if evaluation_service else 0
    }


def get_evaluation_service() -> EvaluationService:
    """Dependency to get evaluation service."""
    if evaluation_service is None:
        raise HTTPException(status_code=503, detail="Evaluation service not initialized")
    return evaluation_service