"""Request and response Pydantic models for the LLM API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to summarize")
    max_length: int = Field(..., gt=0, le=2000, description="Max summary length in characters")


class SummarizeResponse(BaseModel):
    summary: str


class AnalyzeSentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze for sentiment")


class AnalyzeSentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence_score: float = Field(..., ge=0, le=1)
    explanation: str
