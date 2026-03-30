"""Routes for the upload workflow."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.passes.metadata import DEFAULT_METADATA_FIELD_INPUT
from app.services.samples import list_sample_corpora


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
SESSION_COOKIE_NAME = "raglint_session_id"

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    """Render the Phase 1 upload page."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME) or uuid4().hex
    session_store: dict[str, str] = request.app.state.session_api_keys
    response = templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "metadata_fields_input": DEFAULT_METADATA_FIELD_INPUT,
            "api_key_configured": bool(session_store.get(session_id)),
            "error": None,
            "message": None,
            "upload_action": "/report/preview",
            "analysis_action": "/report/analyze",
            "api_key_action": "/report/api-key",
            "sample_corpora": list_sample_corpora(),
        },
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
    )
    return response
