"""Shared exceptions and user-facing error responses for RAGLint."""

from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates


logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


class RAGLintError(Exception):
    """Base exception with a user-facing message and HTTP status code."""

    status_code = 400

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.http_status_code = status_code or self.status_code


class UploadValidationError(RAGLintError):
    """Raised when the current upload request should be rejected."""


class UnsupportedFileTypeError(UploadValidationError):
    """Raised when a provided document or chunk export uses an unsupported extension."""


class FileSizeLimitError(UploadValidationError):
    """Raised when an uploaded file exceeds the per-file size limit."""


class TotalUploadSizeLimitError(UploadValidationError):
    """Raised when the total upload payload exceeds the request limit."""


class DocumentLimitError(UploadValidationError):
    """Raised when too many documents are selected in one analysis."""


class ChunkLimitError(UploadValidationError):
    """Raised when an imported chunk export exceeds the maximum allowed size."""


class DocumentParsingError(RAGLintError):
    """Raised when a document cannot be parsed into text."""


class EmptyDocumentError(DocumentParsingError):
    """Raised when a document produces no meaningful text."""


class ChunkImportError(UploadValidationError):
    """Raised when an imported chunk export cannot be parsed or normalized."""


class MissingAPIKeyError(RAGLintError):
    """Raised when an OpenAI-backed pass is requested without an API key."""


class InvalidAPIKeyError(RAGLintError):
    """Raised when the configured OpenAI API key is rejected."""


class RateLimitReachedError(RAGLintError):
    """Raised when the OpenAI API rejects a request due to rate limiting."""

    status_code = 429


class QuotaExceededError(RAGLintError):
    """Raised when the configured OpenAI account has no remaining quota."""

    status_code = 429


class OpenAIConnectionFailedError(RAGLintError):
    """Raised when the OpenAI API cannot be reached reliably."""

    status_code = 503


class OpenAIServiceError(RAGLintError):
    """Raised for other OpenAI service failures."""

    status_code = 502


class AnalysisPassError(RAGLintError):
    """Raised when a single analysis pass cannot complete."""

    status_code = 500

    def __init__(self, pass_name: str, message: str) -> None:
        super().__init__(message, status_code=500)
        self.pass_name = pass_name


def classify_openai_error(exc: Exception) -> RAGLintError:
    """Map an OpenAI SDK or transport exception to a user-facing RAGLint error."""
    class_name = exc.__class__.__name__
    status_code = getattr(exc, "status_code", None)
    message = str(exc)
    lowered = message.lower()

    if status_code == 401 or "authentication" in class_name.lower() or "invalid api key" in lowered:
        return InvalidAPIKeyError("The provided API key is invalid. Please check your key in settings.")
    if status_code == 402:
        return QuotaExceededError("Your OpenAI API quota has been exceeded. Check your billing at platform.openai.com.")
    if status_code == 429 and any(token in lowered for token in ("quota", "insufficient_quota", "billing", "credit")):
        return QuotaExceededError("Your OpenAI API quota has been exceeded. Check your billing at platform.openai.com.")
    if status_code == 429:
        return RateLimitReachedError(
            "OpenAI rate limit reached. Please wait a moment and try again, or use an API key with higher limits."
        )
    if isinstance(exc, TimeoutError) or any(
        token in class_name for token in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout")
    ):
        return OpenAIConnectionFailedError(
            "Could not connect to OpenAI. Please check your internet connection and try again."
        )
    if any(token in lowered for token in ("timeout", "timed out", "connection", "network")):
        return OpenAIConnectionFailedError(
            "Could not connect to OpenAI. Please check your internet connection and try again."
        )
    return OpenAIServiceError("OpenAI could not complete the request. Please try again.")


def request_prefers_json(request: Request) -> bool:
    """Return whether the current request expects a JSON-style error response."""
    accept_header = request.headers.get("accept", "").lower()
    path = request.url.path
    return (
        "application/json" in accept_header
        or path.startswith("/report/analyze")
        or path.startswith("/report/status/")
    )


def error_response(
    request: Request,
    *,
    title: str,
    message: str,
    status_code: int,
    retry_url: str = "/",
) -> Response:
    """Render a consistent user-facing error response as HTML or JSON."""
    if request_prefers_json(request):
        return JSONResponse({"detail": message}, status_code=status_code)

    context = {
        "request": request,
        "title": title,
        "message": message,
        "retry_url": retry_url,
    }
    try:
        response = templates.TemplateResponse(request, "error.html", context, status_code=status_code)
        _ = response.body
        return response
    except Exception:  # pragma: no cover - only used if the error template itself fails
        logger.exception("Error template rendering failed.")
        return HTMLResponse(
            content=(
                "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
                "<title>RAGLint | Error</title></head><body>"
                f"<main><h1>{escape(title)}</h1><p>{escape(message)}</p>"
                f"<p><a href='{escape(retry_url)}'>Try again</a></p></main></body></html>"
            ),
            status_code=status_code,
        )


def report_render_fallback_response(
    request: Request,
    *,
    report_payload: dict[str, Any],
    export_json_url: str,
    upload_url: str = "/",
) -> HTMLResponse:
    """Render a plain fallback report view when the Jinja report template fails."""
    pretty_json = escape(json.dumps(report_payload, indent=2))
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
        "<title>RAGLint | Report fallback</title>"
        "<style>body{font-family:ui-sans-serif,system-ui,sans-serif;background:#0f1720;color:#e7edf5;"
        "margin:0;padding:32px;}main{max-width:1100px;margin:0 auto;}a,button{color:#0f1720;background:#f6c177;"
        "padding:12px 18px;border-radius:10px;border:none;text-decoration:none;font-weight:600;}pre{background:#111927;"
        "padding:20px;border-radius:14px;overflow:auto;border:1px solid #243041;} .actions{display:flex;gap:12px;"
        "flex-wrap:wrap;margin:20px 0;}</style></head><body><main>"
        "<h1>RAGLint report fallback</h1>"
        "<p>The full report view could not be rendered. The analysis completed successfully, and the raw results are available below.</p>"
        f"<div class='actions'><form action='{escape(export_json_url)}' method='post'><button type='submit'>Download JSON</button></form>"
        f"<a href='{escape(upload_url)}'>Try again</a></div><pre>{pretty_json}</pre></main></body></html>"
    )
    return HTMLResponse(content=html, status_code=200)
