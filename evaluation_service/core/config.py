"""
Configuration settings for evaluation microservice.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "LLM Evaluation Microservice"
    VERSION: str = "1.0.0"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]  # Configure for production
    
    # Google API
    GOOGLE_API_KEY: str
    
    # Service Settings
    NUM_WORKERS: int = 3
    MAX_QUEUE_SIZE: int = 1000
    
    # Storage
    RESULTS_STORAGE_PATH: str = "./evaluation_results"
    
    # Redis (optional)
    REDIS_URL: str = "redis://localhost:6379"
    USE_REDIS: bool = False
    
    # Database (optional)
    DATABASE_URL: str = "postgresql://user:password@localhost/evaluations"
    USE_DATABASE: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()