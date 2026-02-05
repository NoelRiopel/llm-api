"""OpenAI client wrapper for summarization and sentiment analysis.

Uses the Responses API with prompt IDs from the dashboard.
Set in environment: SUMMARIZE_PROMPT_ID, SENTIMENT_PROMPT_ID.
Optional: SUMMARIZE_PROMPT_VERSION, SENTIMENT_PROMPT_VERSION.
"""

import json
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

# Env var names (values read at runtime via os.environ).
SUMMARIZE_PROMPT_ID = "SUMMARIZE_PROMPT_ID"
SENTIMENT_PROMPT_ID = "SENTIMENT_PROMPT_ID"
SUMMARIZE_PROMPT_VERSION = "SUMMARIZE_PROMPT_VERSION"
SENTIMENT_PROMPT_VERSION = "SENTIMENT_PROMPT_VERSION"


@dataclass
class SentimentResult:
    sentiment: Literal["positive", "negative", "neutral"]
    confidence_score: float
    explanation: str


def _get_output_text(response) -> str:
    """Extract aggregated text from Responses API output."""
    if getattr(response, "output_text", None) is not None:
        return response.output_text or ""
    # Fallback: walk output for output_text items
    text_parts = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "content", None):
            for block in item.content:
                if getattr(block, "type", None) == "output_text" and getattr(block, "text", None):
                    text_parts.append(block.text)
    return "".join(text_parts)


class LLMClient:
    """Thin wrapper around OpenAI Responses API using dashboard prompt IDs."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        summarize_prompt_id: str | None = None,
        sentiment_prompt_id: str | None = None,
        summarize_prompt_version: str | None = None,
        sentiment_prompt_version: str | None = None,
    ):
        self._client = OpenAI()
        self._model = model
        self._summarize_prompt_id = summarize_prompt_id or os.environ.get(SUMMARIZE_PROMPT_ID)
        self._sentiment_prompt_id = sentiment_prompt_id or os.environ.get(SENTIMENT_PROMPT_ID)
        self._summarize_version = summarize_prompt_version or os.environ.get(SUMMARIZE_PROMPT_VERSION)
        self._sentiment_version = sentiment_prompt_version or os.environ.get(SENTIMENT_PROMPT_VERSION)

    def summarize(self, text: str, max_length: int) -> str:
        """Summarize text in at most max_length characters."""
        if not self._summarize_prompt_id:
            raise ValueError(f"{SUMMARIZE_PROMPT_ID} is not set")
        prompt_param: dict = {
            "id": self._summarize_prompt_id,
            "variables": {"text": text, "max_length": max_length},
        }
        if self._summarize_version:
            prompt_param["version"] = self._summarize_version
        response = self._client.responses.create(
            model=self._model,
            prompt=prompt_param,
            max_output_tokens=500,
        )
        return _get_output_text(response).strip()

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment; returns sentiment label, confidence, and explanation."""
        if not self._sentiment_prompt_id:
            raise ValueError(f"{SENTIMENT_PROMPT_ID} is not set")
        prompt_param: dict = {
            "id": self._sentiment_prompt_id,
            "variables": {"text": text},
        }
        if self._sentiment_version:
            prompt_param["version"] = self._sentiment_version
        response = self._client.responses.create(
            model=self._model,
            prompt=prompt_param,
            max_output_tokens=300,
        )
        content = _get_output_text(response).strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        data = json.loads(content)
        return SentimentResult(
            sentiment=data["sentiment"],
            confidence_score=float(data["confidence_score"]),
            explanation=data["explanation"],
        )
