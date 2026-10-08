"""API package: FastAPI server (pure components importable without fastapi)."""

from .server import KeyAuth, Metrics, RateLimiter, create_app

__all__ = ["KeyAuth", "Metrics", "RateLimiter", "create_app"]
