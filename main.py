"""
FastAPI LLM API — health, summarize, and sentiment analysis endpoints.
Uses OpenAI API for summarization and sentiment; no local prompt storage.
"""

import json
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="LLM API", version="0.1.0")

# OpenAI client; set OPENAI_API_KEY in environment (e.g. on Render)
client = OpenAI()


# --- Request/Response models ---


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


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return service status and current timestamp."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest):
    """Summarize the given text within the specified max length."""
    prompt = (
        f"Summarize the following text in at most {body.max_length} characters. "
        "Return only the summary, no preamble.\n\n"
        f"Text:\n{body.text}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        summary = response.choices[0].message.content or ""
        return SummarizeResponse(summary=summary.strip())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {str(e)}")


@app.post("/analyze-sentiment", response_model=AnalyzeSentimentResponse)
async def analyze_sentiment(body: AnalyzeSentimentRequest):
    """Analyze sentiment of the given text; returns label, confidence, and explanation."""
    prompt = (
        "Analyze the sentiment of the following text. "
        "Respond in exactly this JSON format, with no other text:\n"
        '{"sentiment": "positive" or "negative" or "neutral", '
        '"confidence_score": number between 0 and 1, '
        '"explanation": "brief explanation"}\n\n'
        f"Text:\n{body.text}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        content = (response.choices[0].message.content or "").strip()
        # Parse JSON from response (handle markdown code blocks if present)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        data = json.loads(content)
        return AnalyzeSentimentResponse(
            sentiment=data["sentiment"],
            confidence_score=float(data["confidence_score"]),
            explanation=data["explanation"],
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid model response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {str(e)}")
