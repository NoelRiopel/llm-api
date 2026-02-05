"""ASGI entry point for the LLM API."""

from fastapi import FastAPI

from endpoints import router

app = FastAPI(title="LLM API")
app.include_router(router)
