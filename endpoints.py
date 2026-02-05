"""API route handlers for health, summarize, and analyze-sentiment."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from models import (
    AnalyzeSentimentRequest,
    AnalyzeSentimentResponse,
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from openai_client import LLMClient

router = APIRouter()
llm = LLMClient()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Return service status and current timestamp."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest):
    """Summarize the given text within the specified max length."""
    try:
        result = llm.summarize(body.text, body.max_length)
        return SummarizeResponse(summary=result.summary)
    except (KeyError, ValueError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid model response: {str(e)} Please try again.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {str(e)} Please try again.")


@router.post("/analyze-sentiment", response_model=AnalyzeSentimentResponse)
async def analyze_sentiment(body: AnalyzeSentimentRequest):
    """Analyze sentiment of the given text; returns label, confidence, and explanation."""
    try:
        result = llm.analyze_sentiment(body.text)
        return AnalyzeSentimentResponse(
            sentiment=result.sentiment,
            confidence_score=result.confidence_score,
            explanation=result.explanation,
        )
    except (KeyError, ValueError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid model response: {str(e)} Please try again.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {str(e)} Please try again.")
