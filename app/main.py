"""FastAPI entry point for the RAGLint app."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from app.errors import RAGLintError, error_response
from app.routers.report import router as report_router
from app.routers.upload import router as upload_router


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAGLint",
        description="Audit document corpora for RAG readiness.",
        version="0.1.0",
    )

    static_path = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    app.state.session_api_keys = {}
    app.state.session_reports = {}
    app.state.analysis_jobs = {}

    @app.exception_handler(RAGLintError)
    async def raglint_error_handler(request: Request, exc: RAGLintError) -> Response:
        logger.exception("Handled RAGLint error for %s", request.url.path, exc_info=exc)
        return error_response(
            request,
            title="RAGLint error",
            message=exc.message,
            status_code=exc.http_status_code,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
        logger.exception("Unhandled application error for %s", request.url.path, exc_info=exc)
        return error_response(
            request,
            title="Something went wrong",
            message="Something went wrong during analysis. Please try again or reduce the number of documents.",
            status_code=500,
        )

    app.include_router(upload_router)
    app.include_router(report_router)

    return app


app = create_app()
