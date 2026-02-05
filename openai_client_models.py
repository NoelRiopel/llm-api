"""Result types and response validators for OpenAI client prompt outputs."""

from typing import Literal

from pydantic import BaseModel, Field


class SummarizeResult(BaseModel):
    """Result of summarization; matches the prompt's JSON output structure."""

    summary: str = Field(..., description="Summary text")


class SentimentResult(BaseModel):
    """Result of sentiment analysis; matches the prompt's JSON output structure."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ..., description="Sentiment label"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score between 0 and 1",
    )
    explanation: str = Field(..., description="Brief explanation")
