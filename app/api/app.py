"""
FastAPI application factory.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    setup_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="Reliable, testable AI claim extraction and fact verification service.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    if settings.API_V1_PREFIX:
        app.include_router(router, prefix=settings.API_V1_PREFIX)
    return app


app = create_app()
