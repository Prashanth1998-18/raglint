"""Routes for parsing preview and the Phase 2 through Phase 5 analysis passes."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.errors import (
    AnalysisPassError,
    ChunkImportError,
    DocumentLimitError,
    DocumentParsingError,
    EmptyDocumentError,
    FileSizeLimitError,
    MissingAPIKeyError,
    RAGLintError,
    TotalUploadSizeLimitError,
    UnsupportedFileTypeError,
    UploadValidationError,
    error_response,
    report_render_fallback_response,
)
from app.models.document import (
    Chunk,
    ContradictionFinding,
    ContradictionRunStats,
    CorpusHealthScore,
    Document,
    DuplicationFinding,
    MetadataAuditSummary,
    MetadataFinding,
    ROTFinding,
    StalenessFinding,
)
from app.services.chunker import RecursiveCharacterChunker
from app.services.embeddings import EmbeddingRunResult, OpenAIEmbeddingService
from app.services.metadata import parse_client_modified_map
from app.services.parser import ChunkExportParser, DocumentParser
from app.services.passes.contradiction import ContradictionDetectionPass
from app.services.passes.duplication import DuplicationDetectionPass
from app.services.passes.metadata import (
    DEFAULT_METADATA_FIELD_INPUT,
    MetadataAuditPass,
    parse_metadata_field_input,
)
from app.services.passes.rot import ROTClassificationPass
from app.services.passes.staleness import StalenessScoringPass
from app.services.scoring import CorpusHealthScorer
from app.services.similarity import SimilarityPair, SimilarityService
from app.services.samples import (
    SampleCorpusDefinition,
    get_sample_definition,
    list_sample_corpora,
    load_sample_report,
)


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
SESSION_COOKIE_NAME = "raglint_session_id"
MAX_DOCUMENTS = 50
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 50 * 1024 * 1024
ANALYSIS_STEPS = (
    ("parsing_documents", "Parsing documents"),
    ("analyzing_text_segments", "Analyzing text segments"),
    ("checking_duplicates", "Checking for duplicates"),
    ("detecting_outdated_content", "Detecting outdated content"),
    ("scanning_contradictions", "Scanning for contradictions"),
    ("auditing_document_information", "Checking document information"),
    ("classifying_content_quality", "Evaluating content quality"),
    ("generating_report", "Generating report"),
)

router = APIRouter(prefix="/report", tags=["report"])
logger = logging.getLogger(__name__)


@router.post("/api-key")
async def store_api_key(request: Request) -> Response:
    """Store or clear the OpenAI API key for the current browser session."""
    session_id = _get_or_create_session_id(request)
    payload = await _request_payload(request)
    openai_api_key = str(payload.get("openai_api_key", "")).strip()

    if openai_api_key:
        _set_session_api_key(request.app, session_id, openai_api_key)
    else:
        _clear_session_api_key(request.app, session_id)

    response = JSONResponse(
        {
            "configured": bool(openai_api_key),
        }
    )
    return _apply_session_cookie(response, session_id)


@router.post("/analyze")
async def start_analysis_job(
    request: Request,
    documents: list[UploadFile] = File(default=[]),
    chunks_export: UploadFile | None = File(default=None),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
    client_modified_map: str | None = Form(default=None),
    metadata_fields: str | None = Form(default=DEFAULT_METADATA_FIELD_INPUT),
) -> Response:
    """Start an asynchronous analysis job and return a pollable status URL."""
    try:
        session_id, job_id = await _queue_analysis_job(
            request=request,
            documents=documents,
            chunks_export=chunks_export,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            client_modified_map=client_modified_map,
            metadata_fields=metadata_fields or DEFAULT_METADATA_FIELD_INPUT,
        )
    except RAGLintError as exc:
        response = JSONResponse({"detail": str(exc)}, status_code=400)
        return _apply_session_cookie(response, _get_or_create_session_id(request))

    response = JSONResponse(
        {
            "job_id": job_id,
            "status_url": f"/report/status/{job_id}",
            "report_url": f"/report/view/{job_id}",
            "api_key_configured": _session_has_api_key(request.app, session_id),
        }
    )
    return _apply_session_cookie(response, session_id)


@router.post("/analyze/stream")
async def start_analysis_stream(
    request: Request,
    documents: list[UploadFile] = File(default=[]),
    chunks_export: UploadFile | None = File(default=None),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
    client_modified_map: str | None = Form(default=None),
    metadata_fields: str | None = Form(default=DEFAULT_METADATA_FIELD_INPUT),
) -> Response:
    """Create an analysis job and return the SSE stream URL for live progress updates."""
    try:
        session_id, job_id = await _queue_analysis_job(
            request=request,
            documents=documents,
            chunks_export=chunks_export,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            client_modified_map=client_modified_map,
            metadata_fields=metadata_fields or DEFAULT_METADATA_FIELD_INPUT,
        )
    except RAGLintError as exc:
        response = JSONResponse({"detail": str(exc)}, status_code=400)
        return _apply_session_cookie(response, _get_or_create_session_id(request))

    response = JSONResponse(
        {
            "job_id": job_id,
            "stream_url": f"/report/analyze/stream/{job_id}",
            "status_url": f"/report/status/{job_id}",
            "report_url": f"/report/view/{job_id}",
        }
    )
    return _apply_session_cookie(response, session_id)


@router.get("/analyze/stream/{job_id}")
async def stream_analysis_progress(request: Request, job_id: str) -> Response:
    """Stream live analysis progress updates over Server-Sent Events."""
    job = _get_analysis_job(request, job_id)
    if job is None:
        return JSONResponse({"detail": "Analysis job not found."}, status_code=404)

    session_id = job["session_id"]

    async def event_stream() -> AsyncIterator[str]:
        yield _sse_message(
            "snapshot",
            {
                "status": job["status"],
                "steps": job["steps"],
                "report_url": f"/report/view/{job_id}",
            },
        )
        if job["status"] == "completed":
            yield _sse_message("complete", {"report_url": f"/report/view/{job_id}"})
            return
        if job["status"] == "error":
            yield _sse_message(
                "job-error",
                {"message": job.get("error") or "Analysis could not complete."},
            )
            return

        queue: asyncio.Queue[dict[str, Any]] = job["event_queue"]
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except TimeoutError:
                if job["status"] == "completed":
                    yield _sse_message("complete", {"report_url": f"/report/view/{job_id}"})
                    return
                if job["status"] == "error":
                    yield _sse_message(
                        "job-error",
                        {"message": job.get("error") or "Analysis could not complete."},
                    )
                    return
                yield ": ping\n\n"
                continue

            yield _sse_message(event["event"], event["data"])
            if event["event"] in {"complete", "job-error"}:
                return

    response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    return _apply_session_cookie(response, session_id)


@router.post("/samples/{sample_id}/start")
async def start_sample_analysis_stream(request: Request, sample_id: str) -> Response:
    """Start the cached sample analysis experience using the existing SSE progress flow."""
    session_id = _get_or_create_session_id(request)
    try:
        sample_definition = get_sample_definition(sample_id)
    except ValueError as exc:
        response = JSONResponse({"detail": str(exc)}, status_code=404)
        return _apply_session_cookie(response, session_id)

    report_payload = load_sample_report(sample_id)
    if report_payload is None:
        response = JSONResponse(
            {
                "detail": (
                    f"The cached report for {sample_definition.title} is not available yet. "
                    "Run app/samples/precompute.py to generate it."
                )
            },
            status_code=503,
        )
        return _apply_session_cookie(response, session_id)

    try:
        report_result = _build_report_result_from_payload(
            report_payload,
            sample_definition=sample_definition,
        )
    except Exception:
        logger.exception("Sample report %s could not be loaded.", sample_definition.report_path.name)
        response = JSONResponse(
            {"detail": "The cached sample report could not be loaded. Please regenerate the sample reports."},
            status_code=500,
        )
        return _apply_session_cookie(response, session_id)

    job_id = uuid4().hex
    jobs_store: dict[str, dict[str, Any]] = request.app.state.analysis_jobs
    jobs_store[job_id] = _new_analysis_job(
        job_id=job_id,
        session_id=session_id,
        metadata_fields_input=str(report_result["template_context"].get("metadata_fields_input", DEFAULT_METADATA_FIELD_INPUT)),
    )
    job = jobs_store[job_id]
    for step in job["steps"]:
        step["status"] = "completed"
        step["detail"] = _sample_step_detail(str(step["key"]), sample_definition)
    job["status"] = "completed"
    job["report_payload"] = report_result["report_payload"]
    job["template_context"] = report_result["template_context"]

    response = JSONResponse(
        {
            "job_id": job_id,
            "stream_url": f"/report/analyze/stream/{job_id}",
            "status_url": f"/report/status/{job_id}",
            "report_url": f"/report/view/{job_id}",
        }
    )
    return _apply_session_cookie(response, session_id)


@router.get("/status/{job_id}")
async def analysis_status(request: Request, job_id: str) -> Response:
    """Return the current analysis job progress for polling."""
    job = _get_analysis_job(request, job_id)
    if job is None:
        return JSONResponse({"detail": "Analysis job not found."}, status_code=404)

    return JSONResponse(
        {
            "status": job["status"],
            "steps": job["steps"],
            "error": job.get("error"),
            "report_url": f"/report/view/{job_id}",
        }
    )


@router.get("/view/{job_id}", response_class=HTMLResponse)
async def view_analysis_report(request: Request, job_id: str) -> Response:
    """Render the completed report for an asynchronous analysis job."""
    job = _get_analysis_job(request, job_id)
    if job is None:
        return JSONResponse({"detail": "Analysis job not found."}, status_code=404)
    if job["status"] != "completed":
        return JSONResponse({"detail": "Analysis is not complete yet."}, status_code=409)

    session_id = job["session_id"]
    request.app.state.session_reports[session_id] = job["report_payload"]
    response = _render_report_response(
        request,
        template_context=job["template_context"],
        report_payload=job["report_payload"],
    )
    return _apply_session_cookie(response, session_id)


@router.post("/preview", response_class=HTMLResponse)
async def preview_report(
    request: Request,
    documents: list[UploadFile] = File(default=[]),
    chunks_export: UploadFile | None = File(default=None),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
    client_modified_map: str | None = Form(default=None),
    openai_api_key: str | None = Form(default=None),
    metadata_fields: str | None = Form(default=DEFAULT_METADATA_FIELD_INPUT),
) -> HTMLResponse:
    """Parse uploads, run analysis passes, and render the report page."""
    session_id = _get_or_create_session_id(request)
    try:
        serialized_documents, serialized_chunks_export = await _prepare_upload_payloads(
            documents=documents,
            chunks_export=chunks_export,
        )
        report_result = await _prepare_report_result(
            app=request.app,
            session_id=session_id,
            documents=_deserialize_uploads(serialized_documents),
            chunks_export=_deserialize_upload(serialized_chunks_export),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            client_modified_map=client_modified_map,
            openai_api_key=openai_api_key,
            metadata_fields=metadata_fields or DEFAULT_METADATA_FIELD_INPUT,
        )
    except UploadValidationError as exc:
        return _apply_session_cookie(
            templates.TemplateResponse(request, "index.html", _upload_page_context(
                request=request,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata_fields=metadata_fields or DEFAULT_METADATA_FIELD_INPUT,
                error=str(exc),
            ), status_code=400),
            session_id,
        )

    request.app.state.session_reports[session_id] = report_result["report_payload"]
    response = _render_report_response(
        request,
        template_context=report_result["template_context"],
        report_payload=report_result["report_payload"],
    )
    return _apply_session_cookie(response, session_id)


@router.post("/export/json")
async def export_json_report(request: Request) -> Response:
    """Return the cached session report as a downloadable JSON file."""
    payload = _get_session_report(request)
    if payload is None:
        return JSONResponse(
            {"detail": "No report is available for this session. Generate a report first."},
            status_code=400,
        )

    try:
        content = json.dumps(payload, indent=2)
    except Exception:
        logger.exception("JSON export failed.")
        return error_response(
            request,
            title="Export failed",
            message="Export failed. Please try again.",
            status_code=500,
        )

    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="raglint-report.json"',
        },
    )


@router.post("/export/markdown")
async def export_markdown_report(request: Request) -> Response:
    """Return the cached session report as a downloadable markdown summary."""
    payload = _get_session_report(request)
    if payload is None:
        return JSONResponse(
            {"detail": "No report is available for this session. Generate a report first."},
            status_code=400,
        )

    try:
        content = _build_markdown_report(payload)
    except Exception:
        logger.exception("Markdown export failed.")
        return error_response(
            request,
            title="Export failed",
            message="Export failed. Please try again.",
            status_code=500,
        )

    return Response(
        content=content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": 'attachment; filename="raglint-report.md"',
        },
    )


async def _prepare_report_result(
    *,
    app,
    session_id: str,
    documents: list[UploadFile],
    chunks_export: UploadFile | None,
    chunk_size: int,
    chunk_overlap: int,
    client_modified_map: str | None,
    openai_api_key: str | None,
    metadata_fields: str,
    progress_callback: Callable[[str, str, str | None], Awaitable[None]] | None = None,
) -> dict[str, object]:
    """Run the full analysis pipeline and return the report payload plus template context."""
    requested_metadata_fields = parse_metadata_field_input(metadata_fields)
    modified_map = parse_client_modified_map(client_modified_map)

    document_parser = DocumentParser()
    chunk_export_parser = ChunkExportParser()
    chunker = RecursiveCharacterChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    await _emit_progress(progress_callback, "parsing_documents", "active", "Reading uploaded documents.")

    parsed_documents: list[Document] = []
    generated_chunks: list[Chunk] = []
    imported_chunks: list[Chunk] = []
    report_warnings: list[str] = []
    parse_failures: list[tuple[str, str]] = []
    empty_documents: list[str] = []

    def add_warning(message: str | None) -> None:
        if not message:
            return
        if message not in report_warnings:
            report_warnings.append(message)

    for upload in documents:
        if not upload.filename:
            continue
        try:
            document = await document_parser.parse_upload(
                upload,
                client_modified_at=modified_map.get(upload.filename),
            )
        except EmptyDocumentError:
            empty_documents.append(upload.filename)
            logger.warning("Uploaded file %s produced no extractable text.", upload.filename)
            continue
        except DocumentParsingError as exc:
            parse_failures.append((upload.filename, str(exc)))
            logger.warning("Uploaded file %s could not be parsed.", upload.filename, exc_info=exc)
            continue

        parsed_documents.append(document)
        generated_chunks.extend(chunker.chunk_document(document))

    if chunks_export is not None and chunks_export.filename:
        imported_chunks = await chunk_export_parser.parse_upload(chunks_export)

    if parse_failures:
        if len(parse_failures) == 1:
            filename, reason = parse_failures[0]
            add_warning(f"1 file could not be parsed: {filename} (reason: {reason}).")
        else:
            details = "; ".join(f"{filename} (reason: {reason})" for filename, reason in parse_failures)
            add_warning(f"{len(parse_failures)} files could not be parsed: {details}.")
    for filename in empty_documents:
        add_warning(f"File {filename} produced no extractable text and was excluded from analysis.")
    if not parsed_documents:
        raise UploadValidationError("None of the uploaded documents could be analyzed. Please upload at least one readable document.")

    parsing_detail_parts = [
        f"{len(parsed_documents)} document{'s' if len(parsed_documents) != 1 else ''} parsed",
    ]
    if generated_chunks:
        parsing_detail_parts.append(
            f"{len(generated_chunks)} text segment{'s' if len(generated_chunks) != 1 else ''} created"
        )
    if imported_chunks:
        parsing_detail_parts.append(
            f"{len(imported_chunks)} existing segment{'s' if len(imported_chunks) != 1 else ''} loaded"
        )
    await _emit_progress(progress_callback, "parsing_documents", "completed", ", ".join(parsing_detail_parts))

    duplication_error: str | None = None
    duplication_message: str | None = None
    staleness_error: str | None = None
    staleness_message: str | None = None
    metadata_error: str | None = None
    metadata_message: str | None = None
    contradiction_message: str | None = None
    contradiction_error: str | None = None
    rot_error: str | None = None
    rot_message: str | None = None

    pass_availability = {
        "duplication": True,
        "staleness": True,
        "contradictions": True,
        "metadata": True,
        "rot": True,
    }

    duplication_findings: list[DuplicationFinding] = []
    contradiction_candidate_pairs: list[SimilarityPair] = []
    supersession_similarity_pairs: list[SimilarityPair] = []
    session_api_key = _resolve_session_api_key(app, session_id, openai_api_key)
    embedded_generated_chunks = generated_chunks
    embedded_imported_chunks = imported_chunks

    if generated_chunks:
        if not session_api_key:
            duplication_message = "Duplication analysis skipped because no OpenAI API key was provided."
            add_warning("An OpenAI API key is required for analysis. Click the settings icon to add your key.")
            pass_availability["duplication"] = False
            await _emit_progress(
                progress_callback,
                "analyzing_text_segments",
                "completed",
                "Add an API key to compare text segments against duplicates and contradictions.",
            )
            await _emit_progress(
                progress_callback,
                "checking_duplicates",
                "completed",
                "Skipped until an API key is added.",
            )
        else:
            await _emit_progress(
                progress_callback,
                "analyzing_text_segments",
                "active",
                f"Preparing {len(generated_chunks)} text segment{'s' if len(generated_chunks) != 1 else ''}.",
            )
            try:
                embedding_service = OpenAIEmbeddingService()
                similarity_service = SimilarityService()
                duplication_pass = DuplicationDetectionPass()

                generated_embedding_result = _coerce_embedding_run_result(
                    await embedding_service.embed_chunks(generated_chunks, session_api_key),
                    generated_chunks,
                )
                if imported_chunks:
                    imported_embedding_result = _coerce_embedding_run_result(
                        await embedding_service.embed_chunks(imported_chunks, session_api_key),
                        imported_chunks,
                    )
                else:
                    imported_embedding_result = EmbeddingRunResult(chunks=imported_chunks, embedded_chunk_count=0)

                if generated_embedding_result.failed_batches:
                    add_warning(_embedding_partial_failure_warning(generated_embedding_result))
                if imported_embedding_result.failed_batches:
                    add_warning(_embedding_partial_failure_warning(imported_embedding_result, label="existing index"))

                embedded_generated_chunks = [chunk for chunk in generated_chunks if chunk.embedding is not None]
                embedded_imported_chunks = [chunk for chunk in imported_chunks if chunk.embedding is not None]
                if not embedded_generated_chunks:
                    raise AnalysisPassError(
                        "duplication",
                        "The duplication analysis encountered an error and was skipped. The remaining passes completed successfully.",
                    )

                await _emit_progress(
                    progress_callback,
                    "analyzing_text_segments",
                    "completed",
                    f"{len(generated_chunks)} text segment{'s' if len(generated_chunks) != 1 else ''} prepared.",
                )
                await _emit_progress(progress_callback, "checking_duplicates", "active", "Comparing related passages.")

                similarity_pairs = similarity_service.find_similar_chunks(
                    embedded_generated_chunks,
                    embedded_imported_chunks or None,
                )
                duplication_findings = duplication_pass.build_findings(similarity_pairs)
                contradiction_candidate_pairs = similarity_service.find_pairs_in_range(
                    embedded_generated_chunks,
                    embedded_imported_chunks or None,
                    min_similarity=0.7,
                    max_similarity=0.95,
                )

                if embedded_imported_chunks:
                    supersession_similarity_pairs = SimilarityService(
                        exact_threshold=0.98,
                        near_duplicate_threshold=0.7,
                    ).find_similar_chunks(
                        embedded_generated_chunks,
                        embedded_imported_chunks,
                    )

                if duplication_findings:
                    duplicates_detail = f"{len(duplication_findings)} duplicate match{'es' if len(duplication_findings) != 1 else ''} found."
                else:
                    duplicates_detail = "No duplicate matches found."
                    duplication_message = "No duplicate or near-duplicate matches were found above the default thresholds."
                await _emit_progress(progress_callback, "checking_duplicates", "completed", duplicates_detail)
            except MissingAPIKeyError:
                duplication_message = "Duplication analysis skipped because no OpenAI API key was provided."
                add_warning("An OpenAI API key is required for analysis. Click the settings icon to add your key.")
                pass_availability["duplication"] = False
                await _emit_progress(progress_callback, "analyzing_text_segments", "completed", duplication_message)
                await _emit_progress(progress_callback, "checking_duplicates", "completed", duplication_message)
            except RAGLintError as exc:
                duplication_error = exc.message
                pass_availability["duplication"] = False
                await _emit_progress(progress_callback, "analyzing_text_segments", "error", duplication_error)
                await _emit_progress(progress_callback, "checking_duplicates", "error", duplication_error)
            except Exception as exc:  # pragma: no cover - defensive API wrapper
                logger.exception("Duplication analysis failed unexpectedly.", exc_info=exc)
                duplication_error = (
                    "The duplication analysis encountered an error and was skipped. "
                    "The remaining passes completed successfully."
                )
                pass_availability["duplication"] = False
                await _emit_progress(progress_callback, "analyzing_text_segments", "error", duplication_error)
                await _emit_progress(progress_callback, "checking_duplicates", "error", duplication_error)
    else:
        duplication_message = "Duplication analysis needs at least one uploaded document so RAGLint can generate new chunks."
        pass_availability["duplication"] = False
        await _emit_progress(progress_callback, "analyzing_text_segments", "completed", "No uploaded documents to prepare.")
        await _emit_progress(progress_callback, "checking_duplicates", "completed", "No uploaded documents to compare.")

    staleness_pass = StalenessScoringPass()
    metadata_pass = MetadataAuditPass(expected_fields=requested_metadata_fields)

    await _emit_progress(progress_callback, "detecting_outdated_content", "active", "Checking dates and freshness signals.")
    staleness_findings: list[StalenessFinding] = []
    supersession_findings: list[StalenessFinding] = []
    if generated_chunks:
        try:
            staleness_findings, supersession_findings = staleness_pass.build_findings(
                generated_chunks,
                parsed_documents,
                existing_chunks=embedded_imported_chunks,
                similarity_pairs=supersession_similarity_pairs,
            )
            stale_detail_parts = [
                f"{len(staleness_findings)} stale item{'s' if len(staleness_findings) != 1 else ''} flagged",
            ]
            if supersession_findings:
                stale_detail_parts.append(
                    f"{len(supersession_findings)} possible replacement{'s' if len(supersession_findings) != 1 else ''}"
                )
            await _emit_progress(progress_callback, "detecting_outdated_content", "completed", ", ".join(stale_detail_parts))
        except Exception as exc:
            logger.exception("Staleness analysis failed unexpectedly.", exc_info=exc)
            staleness_error = (
                "The staleness analysis encountered an error and was skipped. "
                "The remaining passes completed successfully."
            )
            pass_availability["staleness"] = False
            await _emit_progress(progress_callback, "detecting_outdated_content", "error", staleness_error)
    else:
        staleness_message = "Staleness scoring needs at least one uploaded document so RAGLint can analyze new chunks."
        pass_availability["staleness"] = False
        await _emit_progress(progress_callback, "detecting_outdated_content", "completed", "No uploaded documents to review.")

    await _emit_progress(progress_callback, "auditing_document_information", "active", "Checking titles, dates, and ownership.")
    metadata_findings: list[MetadataFinding] = []
    metadata_summary_model = MetadataAuditSummary(average_completeness=0.0, expected_fields=requested_metadata_fields)
    try:
        metadata_findings, metadata_summary_model = metadata_pass.audit_documents(parsed_documents)
        await _emit_progress(
            progress_callback,
            "auditing_document_information",
            "completed",
            f"{len(metadata_findings)} document information result{'s' if len(metadata_findings) != 1 else ''} prepared.",
        )
    except Exception as exc:
        logger.exception("Metadata audit failed unexpectedly.", exc_info=exc)
        metadata_error = (
            "The metadata analysis encountered an error and was skipped. "
            "The remaining passes completed successfully."
        )
        pass_availability["metadata"] = False
        await _emit_progress(progress_callback, "auditing_document_information", "error", metadata_error)

    contradiction_findings: list[ContradictionFinding] = []
    contradiction_stats = ContradictionRunStats()

    if not generated_chunks:
        contradiction_message = "Contradiction detection needs at least one uploaded document so RAGLint can analyze new chunks."
        pass_availability["contradictions"] = False
        await _emit_progress(progress_callback, "scanning_contradictions", "completed", "No uploaded documents to compare.")
    elif not session_api_key:
        contradiction_message = "Contradiction detection requires an OpenAI API key. Provide one to enable this analysis."
        add_warning("An OpenAI API key is required for analysis. Click the settings icon to add your key.")
        pass_availability["contradictions"] = False
        await _emit_progress(progress_callback, "scanning_contradictions", "completed", "Requires API key.")
    elif not pass_availability["duplication"]:
        contradiction_message = "Contradiction detection could not run because the embedding analysis did not complete successfully."
        pass_availability["contradictions"] = False
        await _emit_progress(progress_callback, "scanning_contradictions", "completed", "Skipped because text comparison did not complete.")
    else:
        await _emit_progress(
            progress_callback,
            "scanning_contradictions",
            "active",
            f"{len(contradiction_candidate_pairs)} candidate pair{'s' if len(contradiction_candidate_pairs) != 1 else ''} to check.",
        )
        contradiction_pass = ContradictionDetectionPass()
        try:
            contradiction_findings, contradiction_stats = await contradiction_pass.run(
                contradiction_candidate_pairs,
                api_key=session_api_key,
            )
            if contradiction_stats.llm_calls_made > 0 and contradiction_stats.failed_calls == contradiction_stats.llm_calls_made:
                contradiction_error = (
                    "Contradiction detection failed due to API errors. The remaining passes "
                    "(duplication, staleness, metadata, ROT) completed successfully."
                )
                pass_availability["contradictions"] = False
                contradiction_findings = []
                await _emit_progress(progress_callback, "scanning_contradictions", "error", contradiction_error)
            else:
                if contradiction_stats.failed_calls:
                    contradiction_message = (
                        f"{contradiction_stats.failed_calls} contradiction check"
                        f"{'s' if contradiction_stats.failed_calls != 1 else ''} failed and were skipped."
                    )
                    add_warning(contradiction_message)
                elif not contradiction_findings:
                    contradiction_message = "No contradiction findings were detected in the candidate pair set."
                await _emit_progress(
                    progress_callback,
                    "scanning_contradictions",
                    "completed",
                    f"{len(contradiction_findings)} contradiction{'s' if len(contradiction_findings) != 1 else ''} found.",
                )
        except MissingAPIKeyError:
            contradiction_message = "Contradiction detection requires an OpenAI API key. Provide one to enable this analysis."
            add_warning("An OpenAI API key is required for analysis. Click the settings icon to add your key.")
            pass_availability["contradictions"] = False
            await _emit_progress(progress_callback, "scanning_contradictions", "completed", contradiction_message)
        except RAGLintError as exc:
            contradiction_error = exc.message
            pass_availability["contradictions"] = False
            await _emit_progress(progress_callback, "scanning_contradictions", "error", contradiction_error)
        except Exception as exc:  # pragma: no cover - defensive API wrapper
            logger.exception("Contradiction detection failed unexpectedly.", exc_info=exc)
            contradiction_error = (
                "The contradiction analysis encountered an error and was skipped. "
                "The remaining passes completed successfully."
            )
            pass_availability["contradictions"] = False
            await _emit_progress(progress_callback, "scanning_contradictions", "error", contradiction_error)

    await _emit_progress(progress_callback, "classifying_content_quality", "active", "Combining the analysis into cleanup recommendations.")
    rot_pass = ROTClassificationPass()
    rot_findings: list[ROTFinding] = []
    try:
        rot_findings = await rot_pass.run(
            parsed_documents,
            generated_chunks,
            duplication_findings=duplication_findings,
            staleness_findings=staleness_findings,
            supersession_findings=supersession_findings,
            contradiction_findings=contradiction_findings,
            metadata_findings=metadata_findings,
            api_key=session_api_key,
        )
        await _emit_progress(
            progress_callback,
            "classifying_content_quality",
            "completed",
            f"{len(rot_findings)} content quality result{'s' if len(rot_findings) != 1 else ''} prepared.",
        )
    except Exception as exc:
        logger.exception("ROT classification failed unexpectedly.", exc_info=exc)
        rot_error = (
            "The ROT analysis encountered an error and was skipped. "
            "The remaining passes completed successfully."
        )
        pass_availability["rot"] = False
        await _emit_progress(progress_callback, "classifying_content_quality", "error", rot_error)

    await _emit_progress(progress_callback, "generating_report", "active", "Preparing the final dashboard.")
    duplication_score_available = pass_availability["duplication"]
    staleness_score_available = pass_availability["staleness"]
    contradiction_score_available = pass_availability["contradictions"]
    metadata_score_available = pass_availability["metadata"]
    rot_score_available = pass_availability["rot"]
    health_score = CorpusHealthScorer().calculate(
        documents=parsed_documents,
        chunks=generated_chunks,
        duplication_findings=duplication_findings,
        staleness_findings=staleness_findings,
        contradiction_findings=contradiction_findings,
        contradiction_stats=contradiction_stats,
        metadata_findings=metadata_findings,
        metadata_summary=metadata_summary_model,
        rot_findings=rot_findings,
        duplication_score_available=duplication_score_available,
        staleness_score_available=staleness_score_available,
        contradiction_score_available=contradiction_score_available,
        metadata_score_available=metadata_score_available,
        rot_score_available=rot_score_available,
    )

    visible_metadata_findings = [finding for finding in metadata_findings if _metadata_has_issue(finding)] if metadata_score_available else []
    visible_rot_findings = [finding for finding in rot_findings if _rot_has_issue(finding)] if rot_score_available else []
    report_payload = _build_report_payload(
        documents=parsed_documents,
        generated_chunks=generated_chunks,
        imported_chunks=imported_chunks,
        duplication_findings=duplication_findings,
        staleness_findings=staleness_findings,
        supersession_findings=supersession_findings,
        metadata_findings=metadata_findings,
        metadata_summary=metadata_summary_model,
        contradiction_findings=contradiction_findings,
        contradiction_stats=contradiction_stats,
        rot_findings=rot_findings,
        health_score=health_score,
        messages={
            "warnings": report_warnings,
            "duplication_error": duplication_error,
            "duplication_message": duplication_message,
            "staleness_error": staleness_error,
            "staleness_message": staleness_message,
            "metadata_error": metadata_error,
            "metadata_message": metadata_message,
            "contradiction_error": contradiction_error,
            "contradiction_message": contradiction_message,
            "rot_error": rot_error,
            "rot_message": rot_message,
            "pass_availability": pass_availability,
        },
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata_fields=metadata_summary_model.expected_fields,
    )
    template_context = _build_report_template_context(
        health_score=health_score,
        documents=parsed_documents,
        generated_chunks=generated_chunks,
        imported_chunks=imported_chunks,
        duplication_findings=duplication_findings,
        staleness_findings=staleness_findings,
        supersession_findings=supersession_findings,
        metadata_findings=metadata_findings,
        metadata_summary_model=metadata_summary_model,
        contradiction_findings=contradiction_findings,
        contradiction_stats=contradiction_stats,
        rot_findings=rot_findings,
        visible_metadata_findings=visible_metadata_findings,
        visible_rot_findings=visible_rot_findings,
        duplication_error=duplication_error,
        duplication_message=duplication_message,
        staleness_error=staleness_error,
        staleness_message=staleness_message,
        metadata_error=metadata_error,
        metadata_message=metadata_message,
        contradiction_message=contradiction_message,
        contradiction_error=contradiction_error,
        rot_error=rot_error,
        rot_message=rot_message,
        report_warnings=report_warnings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata_fields=metadata_fields,
    )
    await _emit_progress(progress_callback, "generating_report", "completed", "Report ready.")
    return {
        "report_payload": report_payload,
        "template_context": template_context,
    }


def _build_report_template_context(
    *,
    health_score: CorpusHealthScore,
    documents: list[Document],
    generated_chunks: list[Chunk],
    imported_chunks: list[Chunk],
    duplication_findings: list[DuplicationFinding],
    staleness_findings: list[StalenessFinding],
    supersession_findings: list[StalenessFinding],
    metadata_findings: list[MetadataFinding],
    metadata_summary_model: MetadataAuditSummary,
    contradiction_findings: list[ContradictionFinding],
    contradiction_stats: ContradictionRunStats,
    rot_findings: list[ROTFinding],
    visible_metadata_findings: list[MetadataFinding],
    visible_rot_findings: list[ROTFinding],
    duplication_error: str | None,
    duplication_message: str | None,
    staleness_error: str | None,
    staleness_message: str | None,
    metadata_error: str | None,
    metadata_message: str | None,
    contradiction_message: str | None,
    contradiction_error: str | None,
    rot_error: str | None,
    rot_message: str | None,
    report_warnings: list[str],
    chunk_size: int,
    chunk_overlap: int,
    metadata_fields: str,
    sample_definition: SampleCorpusDefinition | None = None,
) -> dict[str, object]:
    """Build the template context used by the report page."""
    document_views = [_document_view(document) for document in documents]
    generated_chunk_views = [_chunk_view(chunk) for chunk in generated_chunks]
    imported_chunk_views = [_chunk_view(chunk) for chunk in imported_chunks]
    finding_views = [_duplication_finding_view(finding) for finding in duplication_findings]
    staleness_views = [_staleness_finding_view(finding) for finding in staleness_findings]
    supersession_views = [_staleness_finding_view(finding) for finding in supersession_findings]
    metadata_views = [_metadata_finding_view(finding) for finding in visible_metadata_findings]
    contradiction_views = [_contradiction_finding_view(finding) for finding in contradiction_findings]
    rot_views = [_rot_finding_view(finding) for finding in visible_rot_findings]
    pass_overviews = _pass_overviews(
        duplication_findings,
        staleness_findings,
        supersession_findings,
        visible_metadata_findings,
        contradiction_findings,
        visible_rot_findings,
        duplication_error=duplication_error,
        duplication_message=duplication_message,
        staleness_error=staleness_error,
        staleness_message=staleness_message,
        metadata_error=metadata_error,
        metadata_message=metadata_message,
        contradiction_error=contradiction_error,
        contradiction_message=contradiction_message,
        rot_error=rot_error,
        rot_message=rot_message,
    )

    return {
        "health_dashboard": _health_dashboard_view(health_score),
        "pass_overviews": pass_overviews,
        "pass_overviews_by_key": {item["key"]: item for item in pass_overviews},
        "documents": document_views,
        "generated_chunks": generated_chunk_views,
        "existing_chunks": imported_chunk_views,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "metadata_fields_input": metadata_fields,
        "upload_url": "/",
        "export_json_url": "/report/export/json",
        "export_markdown_url": "/report/export/markdown",
        "sample_report": _sample_report_context(sample_definition),
        "report_warnings": report_warnings,
        "duplication_findings": finding_views,
        "duplication_error": duplication_error,
        "duplication_message": duplication_message,
        "finding_counts": _finding_counts(duplication_findings),
        "staleness_findings": staleness_views,
        "supersession_findings": supersession_views,
        "staleness_summary": _staleness_summary(staleness_findings, supersession_findings),
        "staleness_error": staleness_error,
        "staleness_message": staleness_message,
        "metadata_findings": metadata_views,
        "metadata_summary": _metadata_summary_view(metadata_summary_model, bool(documents)),
        "metadata_error": metadata_error,
        "metadata_message": metadata_message,
        "contradiction_findings": contradiction_views,
        "contradiction_message": contradiction_message,
        "contradiction_error": contradiction_error,
        "contradiction_summary": _contradiction_summary_view(contradiction_findings, contradiction_stats),
        "rot_findings": rot_views,
        "rot_error": rot_error,
        "rot_message": rot_message,
        "section_counts": _section_counts(
            duplication_findings,
            staleness_findings,
            supersession_findings,
            contradiction_findings,
            visible_metadata_findings,
            visible_rot_findings,
        ),
    }


def _build_report_result_from_payload(
    report_payload: dict[str, object],
    *,
    sample_definition: SampleCorpusDefinition | None = None,
) -> dict[str, object]:
    """Rebuild template context from a cached report payload."""
    settings = report_payload.get("settings", {})
    settings_payload = settings if isinstance(settings, dict) else {}
    messages = report_payload.get("messages", {})
    message_payload = messages if isinstance(messages, dict) else {}

    metadata_fields_list = [
        item.strip()
        for item in settings_payload.get("metadata_fields", [])
        if isinstance(item, str) and item.strip()
    ]
    metadata_fields_input = ", ".join(metadata_fields_list) or DEFAULT_METADATA_FIELD_INPUT
    report_warnings = [item for item in message_payload.get("warnings", []) if isinstance(item, str)]

    documents = [
        Document.model_validate(item)
        for item in report_payload.get("documents", [])
        if isinstance(item, dict)
    ]
    generated_chunks = [
        Chunk.model_validate(item)
        for item in report_payload.get("generated_chunks", [])
        if isinstance(item, dict)
    ]
    imported_chunks = [
        Chunk.model_validate(item)
        for item in report_payload.get("existing_chunks", [])
        if isinstance(item, dict)
    ]
    duplication_findings = [
        DuplicationFinding.model_validate(item)
        for item in report_payload.get("duplication_findings", [])
        if isinstance(item, dict)
    ]
    staleness_findings = [
        StalenessFinding.model_validate(item)
        for item in report_payload.get("staleness_findings", [])
        if isinstance(item, dict)
    ]
    supersession_findings = [
        StalenessFinding.model_validate(item)
        for item in report_payload.get("supersession_findings", [])
        if isinstance(item, dict)
    ]
    metadata_findings = [
        MetadataFinding.model_validate(item)
        for item in report_payload.get("metadata_findings", [])
        if isinstance(item, dict)
    ]
    metadata_summary_model = MetadataAuditSummary.model_validate(report_payload.get("metadata_summary", {}))
    contradiction_findings = [
        ContradictionFinding.model_validate(item)
        for item in report_payload.get("contradiction_findings", [])
        if isinstance(item, dict)
    ]
    contradiction_stats = ContradictionRunStats.model_validate(report_payload.get("contradiction_stats", {}))
    rot_findings = [
        ROTFinding.model_validate(item)
        for item in report_payload.get("rot_findings", [])
        if isinstance(item, dict)
    ]
    health_score = CorpusHealthScore.model_validate(report_payload.get("health_score", {}))

    visible_metadata_findings = [finding for finding in metadata_findings if _metadata_has_issue(finding)]
    visible_rot_findings = [finding for finding in rot_findings if _rot_has_issue(finding)]

    template_context = _build_report_template_context(
        health_score=health_score,
        documents=documents,
        generated_chunks=generated_chunks,
        imported_chunks=imported_chunks,
        duplication_findings=duplication_findings,
        staleness_findings=staleness_findings,
        supersession_findings=supersession_findings,
        metadata_findings=metadata_findings,
        metadata_summary_model=metadata_summary_model,
        contradiction_findings=contradiction_findings,
        contradiction_stats=contradiction_stats,
        rot_findings=rot_findings,
        visible_metadata_findings=visible_metadata_findings,
        visible_rot_findings=visible_rot_findings,
        duplication_error=_optional_message(message_payload, "duplication_error"),
        duplication_message=_optional_message(message_payload, "duplication_message"),
        staleness_error=_optional_message(message_payload, "staleness_error"),
        staleness_message=_optional_message(message_payload, "staleness_message"),
        metadata_error=_optional_message(message_payload, "metadata_error"),
        metadata_message=_optional_message(message_payload, "metadata_message"),
        contradiction_message=_optional_message(message_payload, "contradiction_message"),
        contradiction_error=_optional_message(message_payload, "contradiction_error"),
        rot_error=_optional_message(message_payload, "rot_error"),
        rot_message=_optional_message(message_payload, "rot_message"),
        report_warnings=report_warnings,
        chunk_size=_int_setting(settings_payload, "chunk_size", 500),
        chunk_overlap=_int_setting(settings_payload, "chunk_overlap", 50),
        metadata_fields=metadata_fields_input,
        sample_definition=sample_definition,
    )
    return {
        "report_payload": report_payload,
        "template_context": template_context,
    }


def _sample_report_context(sample_definition: SampleCorpusDefinition | None) -> dict[str, object]:
    """Return the optional sample report banner copy for the report template."""
    if sample_definition is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "title": sample_definition.title,
        "eyebrow": "Sample report",
        "top_title": "This is a sample analysis. Ready to check your own documents?",
        "top_message": "This is a pre-computed analysis of sample data. Upload your own documents to run a live analysis.",
        "top_button_label": "Try your own documents",
        "bottom_title": "See what RAGLint finds in your documents.",
        "bottom_message": "Upload your own files to run a live analysis with your content.",
        "bottom_button_label": "Upload documents",
    }


def _optional_message(messages: dict[str, object], key: str) -> str | None:
    """Read an optional message string from the serialized report payload."""
    value = messages.get(key)
    return value if isinstance(value, str) and value else None


def _int_setting(settings: dict[str, object], key: str, default: int) -> int:
    """Read an integer setting from serialized payload data with a safe fallback."""
    value = settings.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _document_view(document: Document) -> dict[str, object]:
    """Prepare a document model for display in the template."""
    payload = document.model_dump(mode="json")
    metadata = payload["metadata"]

    return {
        "id": payload["id"],
        "source_path": payload["source_path"],
        "text": payload["text"],
        "metadata": {
            "filename": metadata["filename"],
            "file_extension": metadata["extension"],
            "file_size": metadata["file_size"],
            "created_at": metadata["created_at"],
            "modified_at": metadata["modified_at"],
            "embedded_metadata": metadata["embedded"],
            "frontmatter": metadata["frontmatter"],
            "metadata_json": json.dumps(metadata, indent=2, sort_keys=True),
        },
    }


def _chunk_view(chunk: Chunk) -> dict[str, object]:
    """Prepare a chunk model for display in the template."""
    payload = chunk.model_dump(mode="json")
    position = payload["position"]
    return {
        "id": payload["id"],
        "text": payload["text"],
        "parent_document_id": payload["parent_document_id"],
        "parent_document_name": payload["parent_document_name"],
        "position": position["chunk_index"],
        "start_index": position["start_char"],
        "end_index": position["end_char"],
        "metadata": payload["metadata"],
        "metadata_json": json.dumps(payload["metadata"], indent=2, sort_keys=True),
        "embedding_dimensions": len(payload["embedding"]) if payload["embedding"] else 0,
    }


def _duplication_finding_view(finding: DuplicationFinding) -> dict[str, object]:
    """Prepare duplication findings for the report template."""
    payload = finding.model_dump(mode="json")
    finding_type = payload["finding_type"]

    return {
        "id": payload["id"],
        "finding_type": finding_type,
        "finding_label": _finding_label(finding_type),
        "severity_class": "severity-exact" if finding_type == "exact" else "severity-near",
        "similarity_score": payload["similarity_score"],
        "similarity_label": f"{payload['similarity_score']:.3f}",
        "explanation": payload["explanation"],
        "recommendation": payload["recommendation"],
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source_filename": chunk["source_filename"] or "Unknown source",
                "parent_document_id": chunk["parent_document_id"],
                "source_scope": chunk["source_scope"],
            }
            for chunk in payload["chunks_involved"]
        ],
    }


def _finding_counts(findings: list[DuplicationFinding]) -> dict[str, int]:
    """Summarize duplication findings for report badges."""
    counts = {"exact": 0, "near_duplicate": 0, "already_in_index": 0}
    for finding in findings:
        counts[finding.finding_type] += 1
    return counts


def _staleness_finding_view(finding: StalenessFinding) -> dict[str, object]:
    """Prepare staleness findings for the report template."""
    payload = finding.model_dump(mode="json")
    score = payload["staleness_score"]
    signals = payload["signals"]
    notes = _staleness_notes(signals)
    metadata_dates = _staleness_metadata_labels(signals.get("metadata_dates", {}))
    signal_count = len(metadata_dates) + len(signals.get("detected_dates", [])) + len(signals.get("temporal_language", []))
    signal_count += len(signals.get("version_references", [])) + len(notes)

    return {
        "id": payload["id"],
        "finding_type": payload["finding_type"],
        "finding_label": (
            "Potentially superseded" if payload["finding_type"] == "potentially_superseded" else "Staleness finding"
        ),
        "severity_class": _score_severity_class(score, high_threshold=0.8, medium_threshold=0.5),
        "source_filename": payload["chunk"]["source_filename"],
        "score": score,
        "score_label": f"{score:.2f}",
        "signal_count": signal_count,
        "chunk_text": payload["chunk"]["text"],
        "signals": {
            "metadata_dates": metadata_dates,
            "detected_dates": signals.get("detected_dates", []),
            "temporal_language": signals.get("temporal_language", []),
            "version_references": signals.get("version_references", []),
            "notes": notes,
        },
        "explanation": payload["explanation"],
        "recommendation": payload["recommendation"],
        "related_chunk": (
            {
                "source_filename": payload["related_chunk"]["source_filename"],
                "text": payload["related_chunk"]["text"],
            }
            if payload.get("related_chunk")
            else None
        ),
        "related_similarity_label": (
            f"{signals['matched_similarity_score']:.3f}"
            if signals.get("matched_similarity_score") is not None
            else None
        ),
    }


def _metadata_finding_view(finding: MetadataFinding) -> dict[str, object]:
    """Prepare metadata audit findings for the report template."""
    payload = finding.model_dump(mode="json")
    score = payload["completeness_score"]
    return {
        "id": payload["id"],
        "document_filename": payload["document_filename"],
        "score": score,
        "score_label": f"{score * 100:.0f}%",
        "severity_class": _metadata_severity_class(score),
        "missing_fields": payload["missing_fields"],
        "consistency_issues": payload["consistency_issues"],
        "explanation": payload["explanation"],
        "recommendation": payload["recommendation"],
    }


def _metadata_summary_view(
    summary: MetadataAuditSummary,
    has_documents: bool,
) -> dict[str, object] | None:
    """Prepare the metadata summary block for the report template."""
    if not has_documents:
        return None

    return {
        "average_completeness": summary.average_completeness,
        "average_completeness_label": f"{summary.average_completeness * 100:.0f}%",
        "most_common_missing_fields": [
            {"field": field_name, "count": count}
            for field_name, count in summary.most_common_missing_fields.items()
        ],
        "consistency_issues": summary.consistency_issues,
        "expected_fields": summary.expected_fields,
        "expected_fields_label": ", ".join(summary.expected_fields),
    }


def _contradiction_finding_view(finding: ContradictionFinding) -> dict[str, object]:
    """Prepare contradiction findings for the report template."""
    payload = finding.model_dump(mode="json")
    left_chunk = payload["chunks_involved"][0]
    right_chunk = payload["chunks_involved"][1]
    return {
        "id": payload["id"],
        "severity": payload["severity"],
        "severity_class": "severity-red" if payload["severity"] == "high" else "severity-orange",
        "severity_label": "High severity" if payload["severity"] == "high" else "Medium severity",
        "similarity_label": f"{payload['similarity_score']:.3f}",
        "source_a": left_chunk["source_filename"],
        "source_b": right_chunk["source_filename"],
        "claim_a": payload["claim_a"],
        "claim_b": payload["claim_b"],
        "explanation": payload["explanation"],
        "why_it_matters": payload["why_it_matters"],
        "recommendation": payload["recommendation"],
        "chunk_text_a": left_chunk["text"],
        "chunk_text_b": right_chunk["text"],
        "chunk_a_label": left_chunk["source_scope"].replace("_", " "),
        "chunk_b_label": right_chunk["source_scope"].replace("_", " "),
    }


def _contradiction_summary_view(
    findings: list[ContradictionFinding],
    stats: ContradictionRunStats,
) -> dict[str, object]:
    """Prepare contradiction summary statistics for the report template."""
    high = sum(1 for finding in findings if finding.severity == "high")
    medium = sum(1 for finding in findings if finding.severity == "medium")
    return {
        "high": high,
        "medium": medium,
        "llm_calls": stats.llm_calls_made,
        "failed_calls": stats.failed_calls,
        "candidate_pairs_considered": stats.candidate_pairs_considered,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "estimated_cost": stats.estimated_cost_usd,
        "estimated_cost_label": f"${stats.estimated_cost_usd:.4f}",
    }


def _rot_finding_view(finding: ROTFinding) -> dict[str, object]:
    """Prepare ROT findings for the report template."""
    payload = finding.model_dump(mode="json")
    classifications = payload["classifications"]
    if classifications == ["healthy"]:
        severity_class = "severity-green"
    elif len(classifications) > 1 or "redundant" in classifications:
        severity_class = "severity-red"
    else:
        severity_class = "severity-orange"

    signals = payload["signals"]
    duplication = signals.get("duplication", {})
    staleness = signals.get("staleness", {})
    triviality = signals.get("triviality", {})
    contradictions = signals.get("contradictions", {})

    return {
        "id": payload["id"],
        "document_id": payload["document_id"],
        "document_filename": payload["document_filename"],
        "classifications": [item.replace("_", " ").title() for item in classifications],
        "severity_class": severity_class,
        "is_healthy": classifications == ["healthy"],
        "explanation": payload["explanation"],
        "impact_on_rag_quality": payload["impact_on_rag_quality"],
        "recommendation": payload["recommendation"],
        "signals": {
            "duplicate_ratio": duplication.get("duplicate_ratio"),
            "duplicated_chunks": duplication.get("duplicated_chunks"),
            "average_staleness_score": staleness.get("average_staleness_score"),
            "stale_chunks_above_0_7": staleness.get("stale_chunks_above_0_7"),
            "average_chunk_word_count": triviality.get("average_chunk_word_count"),
            "average_unique_term_ratio": triviality.get("average_unique_term_ratio"),
            "boilerplate_patterns": triviality.get("boilerplate_patterns", []),
            "contradiction_count": contradictions.get("count", 0),
            "metadata": signals.get("metadata"),
        },
    }


def _metadata_has_issue(finding: MetadataFinding) -> bool:
    """Return whether a metadata finding represents a real issue worth showing."""
    return bool(finding.missing_fields or finding.consistency_issues or finding.completeness_score < 1.0)


def _rot_has_issue(finding: ROTFinding) -> bool:
    """Return whether a ROT finding should appear in the issue-focused report view."""
    return finding.classifications != ["healthy"]


def _health_dashboard_view(score: CorpusHealthScore) -> dict[str, object]:
    """Prepare the Phase 5 health dashboard view model."""
    overall = score.overall_score
    projected = score.projected_overall_score
    summary = score.summary

    return {
        "overall_score": overall,
        "overall_score_label": f"{overall:.1f}",
        "overall_severity_class": _score_severity_class(overall, high_threshold=80, medium_threshold=60),
        "projected_score": projected,
        "projected_score_label": f"{projected:.1f}",
        "projected_delta_label": f"{projected - overall:+.1f}",
        "skipped_dimensions": score.skipped_dimensions,
        "score_note": (
            "Overall score excludes skipped dimensions: "
            + ", ".join(item.title() for item in score.skipped_dimensions)
            if score.skipped_dimensions
            else None
        ),
        "dimension_scores": [
            {
                "key": dimension.key,
                "label": dimension.label,
                "score": dimension.score,
                "score_label": "N/A" if dimension.score is None else f"{dimension.score:.1f}",
                "severity_class": (
                    "severity-neutral"
                    if dimension.score is None
                    else _score_severity_class(dimension.score, high_threshold=80, medium_threshold=60)
                ),
            }
            for dimension in score.dimension_scores
        ],
        "summary": {
            "total_documents": summary.total_documents,
            "total_chunks": summary.total_chunks_analyzed,
            "unique_documents_with_issues": summary.unique_documents_with_issues,
            "unique_documents_with_issues_label": (
                f"{summary.unique_documents_with_issues} of {summary.total_documents}"
            ),
            "duplicate_clusters": summary.duplicate_clusters,
            "stale_chunks": summary.stale_chunks,
            "contradictions_found": summary.contradictions_found,
            "average_metadata_completeness": summary.average_metadata_completeness,
            "average_metadata_completeness_label": f"{summary.average_metadata_completeness * 100:.0f}%",
            "documents_recommended_for_removal": summary.documents_recommended_for_removal,
        },
    }


def _pass_overviews(
    duplication_findings: list[DuplicationFinding],
    staleness_findings: list[StalenessFinding],
    supersession_findings: list[StalenessFinding],
    metadata_findings: list[MetadataFinding],
    contradiction_findings: list[ContradictionFinding],
    rot_findings: list[ROTFinding],
    *,
    duplication_error: str | None,
    duplication_message: str | None,
    staleness_error: str | None,
    staleness_message: str | None,
    metadata_error: str | None,
    metadata_message: str | None,
    contradiction_error: str | None,
    contradiction_message: str | None,
    rot_error: str | None,
    rot_message: str | None,
) -> list[dict[str, object]]:
    """Build concise per-pass summary rows for the drill-down report layout."""
    duplication_counts = _finding_counts(duplication_findings)
    duplication_total = len(duplication_findings)
    duplication_parts = []
    if duplication_counts["exact"]:
        duplication_parts.append(
            f"{duplication_counts['exact']} exact duplicate {_pluralize('cluster', duplication_counts['exact'])}"
        )
    if duplication_counts["near_duplicate"]:
        duplication_parts.append(
            f"{duplication_counts['near_duplicate']} near-duplicate {_pluralize('pair', duplication_counts['near_duplicate'])}"
        )
    if duplication_counts["already_in_index"]:
        duplication_parts.append(
            f"{duplication_counts['already_in_index']} already-in-index {_pluralize('match', duplication_counts['already_in_index'])}"
        )
    if duplication_parts:
        duplication_summary = ", ".join(duplication_parts) + " found."
    elif duplication_error:
        duplication_summary = duplication_error
    else:
        duplication_summary = duplication_message or "No duplication issues found."

    stale_total = len(staleness_findings)
    superseded_total = len(supersession_findings)
    staleness_parts = []
    if stale_total:
        staleness_parts.append(f"{stale_total} stale {_pluralize('chunk', stale_total)} flagged")
    if superseded_total:
        staleness_parts.append(
            f"{superseded_total} existing {_pluralize('chunk', superseded_total)} may be superseded"
        )
    if staleness_parts:
        staleness_summary = ", ".join(staleness_parts) + "."
    elif staleness_error:
        staleness_summary = staleness_error
    else:
        staleness_summary = staleness_message or "No staleness concerns found."

    contradiction_high = sum(1 for finding in contradiction_findings if finding.severity == "high")
    contradiction_medium = sum(1 for finding in contradiction_findings if finding.severity == "medium")
    contradiction_total = len(contradiction_findings)
    if contradiction_total:
        contradiction_summary = f"{contradiction_total} {_pluralize('contradiction', contradiction_total)} detected."
    elif contradiction_error:
        contradiction_summary = contradiction_error
    else:
        contradiction_summary = contradiction_message or "No contradictions detected."

    metadata_total = len(metadata_findings)
    if metadata_total:
        metadata_summary = (
            f"{metadata_total} uploaded {_pluralize('document', metadata_total)} missing configured metadata."
        )
    elif metadata_error:
        metadata_summary = metadata_error
    else:
        metadata_summary = metadata_message or "No metadata gaps found in uploaded documents."

    rot_total = len(rot_findings)
    if rot_total:
        rot_summary = f"{rot_total} {_pluralize('document', rot_total)} classified as redundant, outdated, or trivial."
    elif rot_error:
        rot_summary = rot_error
    else:
        rot_summary = rot_message or "No ROT flags on uploaded documents."

    return [
        {
            "key": "duplication",
            "label": "Duplications",
            "target_id": "duplication-section",
            "count": duplication_total,
            "summary": duplication_summary,
            "severity_class": (
                "severity-red"
                if duplication_counts["exact"] > 0
                else "severity-orange" if duplication_total > 0 or duplication_error else "severity-green"
            ),
        },
        {
            "key": "staleness",
            "label": "Staleness",
            "target_id": "staleness-section",
            "count": stale_total + superseded_total,
            "summary": staleness_summary,
            "severity_class": (
                "severity-red"
                if any(finding.staleness_score > 0.7 for finding in staleness_findings)
                else "severity-orange" if stale_total + superseded_total > 0 or staleness_error else "severity-green"
            ),
        },
        {
            "key": "contradictions",
            "label": "Contradictions",
            "target_id": "contradiction-section",
            "count": contradiction_total,
            "summary": contradiction_summary,
            "severity_class": (
                "severity-red"
                if contradiction_high > 0
                else "severity-orange"
                if contradiction_medium > 0 or contradiction_error or (contradiction_message and "requires" in contradiction_message.lower())
                else "severity-green"
            ),
        },
        {
            "key": "metadata",
            "label": "Metadata",
            "target_id": "metadata-section",
            "count": metadata_total,
            "summary": metadata_summary,
            "severity_class": (
                "severity-red"
                if any(finding.completeness_score < 0.3 for finding in metadata_findings)
                else "severity-orange" if metadata_total > 0 or metadata_error else "severity-green"
            ),
        },
        {
            "key": "rot",
            "label": "ROT",
            "target_id": "rot-section",
            "count": rot_total,
            "summary": rot_summary,
            "severity_class": (
                "severity-red"
                if any("redundant" in finding.classifications or len(finding.classifications) > 1 for finding in rot_findings)
                else "severity-orange" if rot_total > 0 or rot_error else "severity-green"
            ),
        },
    ]


def _section_counts(
    duplication_findings: list[DuplicationFinding],
    staleness_findings: list[StalenessFinding],
    supersession_findings: list[StalenessFinding],
    contradiction_findings: list[ContradictionFinding],
    metadata_findings: list[MetadataFinding],
    rot_findings: list[ROTFinding],
) -> dict[str, int]:
    """Count visible findings per report section."""
    return {
        "duplication": len(duplication_findings),
        "staleness": len(staleness_findings) + len(supersession_findings),
        "contradictions": len(contradiction_findings),
        "metadata": len(metadata_findings),
        "rot": len(rot_findings),
    }


def _staleness_summary(
    findings: list[StalenessFinding],
    supersession_findings: list[StalenessFinding],
) -> dict[str, int]:
    """Summarize staleness severity buckets for the report header."""
    summary = {"high": 0, "review": 0, "fresh": 0, "superseded": len(supersession_findings)}
    for finding in findings:
        if finding.staleness_score > 0.8:
            summary["high"] += 1
        elif finding.staleness_score >= 0.5:
            summary["review"] += 1
        else:
            summary["fresh"] += 1
    return summary


def _pluralize(word: str, count: int) -> str:
    """Return a basic pluralized label for short summary copy."""
    if count == 1:
        return word
    if word.endswith(("ch", "sh", "s", "x", "z")):
        return f"{word}es"
    return f"{word}s"


def _staleness_notes(signals: dict[str, object]) -> list[str]:
    """Convert staleness signal details into short human-readable notes."""
    notes: list[str] = []

    metadata_reference_date = signals.get("metadata_reference_date")
    if metadata_reference_date:
        notes.append(f"Metadata reference date: {str(metadata_reference_date)[:10]}")

    content_reference_date = signals.get("content_reference_date")
    if content_reference_date:
        notes.append(f"Latest text date: {str(content_reference_date)[:10]}")

    matched_newer_chunk = signals.get("matched_newer_chunk_filename")
    matched_newer_reference_date = signals.get("matched_newer_reference_date")
    if matched_newer_chunk and matched_newer_reference_date:
        notes.append(
            f"Matched newer content from {matched_newer_chunk} dated {str(matched_newer_reference_date)[:10]}"
        )

    metadata_age = signals.get("metadata_age_months")
    if metadata_age is not None:
        notes.append(f"Metadata age: {metadata_age} months")

    content_age = signals.get("content_age_months")
    if content_age is not None:
        notes.append(f"Text reference age: {content_age} months")

    return notes


def _staleness_metadata_labels(metadata_dates: object) -> list[str]:
    """Render staleness metadata dates as short labels for the template."""
    if not isinstance(metadata_dates, dict):
        return []

    labels: list[str] = []
    created_at = metadata_dates.get("created_at")
    modified_at = metadata_dates.get("modified_at")

    if created_at:
        labels.append(f"Created {str(created_at)[:10]}")
    if modified_at:
        labels.append(f"Modified {str(modified_at)[:10]}")

    return labels


def _score_severity_class(score: float, *, high_threshold: float, medium_threshold: float) -> str:
    """Map a score to a red, orange, or green severity class."""
    if score > high_threshold:
        return "severity-red"
    if score >= medium_threshold:
        return "severity-orange"
    return "severity-green"


def _metadata_severity_class(score: float) -> str:
    """Map a metadata completeness score to a severity class."""
    if score < 0.3:
        return "severity-red"
    if score < 0.6:
        return "severity-orange"
    return "severity-green"


def _finding_label(finding_type: str) -> str:
    """Map finding type values to display labels."""
    labels = {
        "exact": "Exact duplicate",
        "near_duplicate": "Near duplicate",
        "already_in_index": "Already in your index",
    }
    return labels[finding_type]


def _get_or_create_session_id(request: Request) -> str:
    """Reuse the current session cookie or create a new one."""
    return request.cookies.get(SESSION_COOKIE_NAME) or uuid4().hex


def _resolve_session_api_key(
    app,
    session_id: str,
    openai_api_key: str | None,
) -> str | None:
    """Persist the API key in memory for the current browser session only."""
    session_store: dict[str, str] = app.state.session_api_keys
    cleaned_key = (openai_api_key or "").strip()
    if cleaned_key:
        session_store[session_id] = cleaned_key
        return cleaned_key
    return session_store.get(session_id)


def _set_session_api_key(app, session_id: str, openai_api_key: str) -> None:
    """Persist a cleaned OpenAI API key in memory for the current session."""
    cleaned_key = openai_api_key.strip()
    if cleaned_key:
        app.state.session_api_keys[session_id] = cleaned_key


def _clear_session_api_key(app, session_id: str) -> None:
    """Remove any stored API key for the current browser session."""
    app.state.session_api_keys.pop(session_id, None)


def _session_has_api_key(app, session_id: str) -> bool:
    """Return whether an API key is configured for the current session."""
    return bool(app.state.session_api_keys.get(session_id))


def _get_session_report(request: Request) -> dict[str, object] | None:
    """Return the cached report payload for the current browser session."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        return None
    session_reports: dict[str, dict[str, object]] = request.app.state.session_reports
    return session_reports.get(session_id)


def _new_analysis_job(
    *,
    job_id: str,
    session_id: str,
    metadata_fields_input: str,
) -> dict[str, Any]:
    """Create the initial polling state for an asynchronous analysis run."""
    return {
        "id": job_id,
        "session_id": session_id,
        "status": "running",
        "error": None,
        "metadata_fields_input": metadata_fields_input,
        "steps": [
            {
                "key": key,
                "label": label,
                "status": "pending",
                "detail": None,
            }
            for key, label in ANALYSIS_STEPS
        ],
        "current_step": None,
        "report_payload": None,
        "template_context": None,
        "event_queue": asyncio.Queue(),
    }


def _get_analysis_job(request: Request, job_id: str) -> dict[str, Any] | None:
    """Return an analysis job only if it belongs to the current session."""
    jobs_store: dict[str, dict[str, Any]] = request.app.state.analysis_jobs
    job = jobs_store.get(job_id)
    if job is None:
        return None
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id and job["session_id"] != session_id:
        return None
    return job


def _update_analysis_job_step(
    job: dict[str, Any],
    step_key: str,
    status: str,
    detail: str | None = None,
) -> None:
    """Update a single step inside the job progress payload."""
    for step in job["steps"]:
        if step["key"] != step_key:
            continue
        step["status"] = status
        if detail is not None:
            step["detail"] = detail
        if status == "active":
            job["current_step"] = step_key
        elif job.get("current_step") == step_key:
            job["current_step"] = None
        return


async def _run_analysis_job(
    *,
    app,
    job_id: str,
    session_id: str,
    serialized_documents: list[dict[str, Any]],
    serialized_chunks_export: dict[str, Any] | None,
    chunk_size: int,
    chunk_overlap: int,
    client_modified_map: str | None,
    metadata_fields: str,
) -> None:
    """Execute an asynchronous analysis job and persist its final report data."""
    jobs_store: dict[str, dict[str, Any]] = app.state.analysis_jobs
    job = jobs_store[job_id]

    async def progress(step_key: str, status: str, detail: str | None = None) -> None:
        _update_analysis_job_step(job, step_key, status, detail)
        await _publish_analysis_job_event(
            job,
            "step",
            {
                "key": step_key,
                "status": status,
                "detail": detail,
            },
        )

    try:
        report_result = await _prepare_report_result(
            app=app,
            session_id=session_id,
            documents=_deserialize_uploads(serialized_documents),
            chunks_export=_deserialize_upload(serialized_chunks_export),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            client_modified_map=client_modified_map,
            openai_api_key=None,
            metadata_fields=metadata_fields,
            progress_callback=progress,
        )
    except RAGLintError as exc:
        _mark_analysis_job_error(job, str(exc))
        await _publish_analysis_job_event(job, "job-error", {"message": str(exc)})
        return
    except Exception as exc:  # pragma: no cover - defensive async job wrapper
        logger.exception("Background analysis job %s failed unexpectedly.", job_id, exc_info=exc)
        message = f"Analysis could not complete: {exc}"
        _mark_analysis_job_error(job, message)
        await _publish_analysis_job_event(job, "job-error", {"message": message})
        return

    job["status"] = "completed"
    job["report_payload"] = report_result["report_payload"]
    job["template_context"] = report_result["template_context"]
    await _publish_analysis_job_event(
        job,
        "complete",
        {
            "report_url": f"/report/view/{job_id}",
        },
    )


def _sample_step_detail(step_key: str, sample_definition: SampleCorpusDefinition) -> str:
    """Return the completed progress copy for cached sample jobs."""
    details = {
        "parsing_documents": f"{sample_definition.document_count} sample documents loaded.",
        "analyzing_text_segments": "Cached text segments loaded.",
        "checking_duplicates": "Cached duplication findings loaded.",
        "detecting_outdated_content": "Cached staleness findings loaded.",
        "scanning_contradictions": "Cached contradiction findings loaded.",
        "auditing_document_information": "Cached metadata audit results loaded.",
        "classifying_content_quality": "Cached content quality classifications loaded.",
        "generating_report": "Sample report ready.",
    }
    return details[step_key]


def _mark_analysis_job_error(job: dict[str, Any], message: str) -> None:
    """Mark the current job as failed and attach the error to the active step."""
    job["status"] = "error"
    job["error"] = message
    current_step = job.get("current_step") or "generating_report"
    _update_analysis_job_step(job, current_step, "error", message)


async def _emit_progress(
    callback: Callable[[str, str, str | None], Awaitable[None]] | None,
    step_key: str,
    status: str,
    detail: str | None = None,
) -> None:
    """Safely send a progress update when a callback is present."""
    if callback is None:
        return
    await callback(step_key, status, detail)


async def _publish_analysis_job_event(
    job: dict[str, Any],
    event: str,
    data: dict[str, Any],
) -> None:
    """Append an SSE-friendly event payload to the current analysis job queue."""
    queue: asyncio.Queue[dict[str, Any]] = job["event_queue"]
    await queue.put({"event": event, "data": data})


async def _request_payload(request: Request) -> dict[str, Any]:
    """Read either a JSON or form request body into a plain dictionary."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        return payload if isinstance(payload, dict) else {}
    form = await request.form()
    return dict(form)


def _render_report_response(
    request: Request,
    *,
    template_context: dict[str, object],
    report_payload: dict[str, object],
) -> Response:
    """Render the report template or fall back to a raw JSON report view."""
    try:
        response = templates.TemplateResponse(
            request,
            "report.html",
            {
                "request": request,
                **template_context,
            },
        )
        _ = response.body
        return response
    except Exception:
        logger.exception("Report template rendering failed.")
        return report_render_fallback_response(
            request,
            report_payload=report_payload,
            export_json_url="/report/export/json",
            upload_url="/",
        )


async def _prepare_upload_payloads(
    *,
    documents: list[UploadFile],
    chunks_export: UploadFile | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Validate the incoming uploads and return serialized payloads."""
    selected_documents = [upload for upload in documents if upload.filename]
    if not selected_documents:
        raise UploadValidationError("Please select at least one document to analyze.")
    if len(selected_documents) > MAX_DOCUMENTS:
        raise DocumentLimitError(
            f"You selected {len(selected_documents)} documents. The maximum is 50 per analysis. Please reduce your selection."
        )

    total_size = 0
    serialized_documents: list[dict[str, Any]] = []
    for upload in selected_documents:
        filename = upload.filename or "uploaded-file"
        extension = Path(filename).suffix.lower()
        if extension not in DocumentParser.supported_extensions:
            raise UnsupportedFileTypeError(
                f"File {filename} is not supported. RAGLint accepts PDF, DOCX, Markdown, and TXT files."
            )
        serialized = await _serialize_upload(upload)
        if serialized is None:
            continue
        total_size += int(serialized["size"])
        if serialized["size"] > MAX_FILE_BYTES:
            raise FileSizeLimitError(f"File {filename} exceeds the 10MB limit.")
        serialized_documents.append(serialized)

    serialized_chunks = None
    if chunks_export is not None and chunks_export.filename:
        filename = chunks_export.filename
        extension = Path(filename).suffix.lower()
        if extension not in ChunkExportParser.supported_extensions:
            raise ChunkImportError("The chunks export file could not be read. Please ensure it is valid JSON or CSV.")
        serialized_chunks = await _serialize_upload(chunks_export)
        if serialized_chunks is not None:
            total_size += int(serialized_chunks["size"])
            if serialized_chunks["size"] > MAX_FILE_BYTES:
                raise FileSizeLimitError(f"File {filename} exceeds the 10MB limit.")

    if total_size > MAX_TOTAL_UPLOAD_BYTES:
        raise TotalUploadSizeLimitError("Total upload size exceeds 50MB.")

    return serialized_documents, serialized_chunks


async def _serialize_upload(upload: UploadFile | None) -> dict[str, Any] | None:
    """Read an uploaded file into an in-memory payload for background processing."""
    if upload is None or not upload.filename:
        return None
    content = await upload.read()
    return {
        "filename": upload.filename,
        "content_type": upload.content_type,
        "content": content,
        "size": len(content),
    }


async def _queue_analysis_job(
    *,
    request: Request,
    documents: list[UploadFile],
    chunks_export: UploadFile | None,
    chunk_size: int,
    chunk_overlap: int,
    client_modified_map: str | None,
    metadata_fields: str,
) -> tuple[str, str]:
    """Serialize the request payload, create a background job, and start analysis."""
    session_id = _get_or_create_session_id(request)
    documents_payload, chunks_payload = await _prepare_upload_payloads(
        documents=documents,
        chunks_export=chunks_export,
    )

    job_id = uuid4().hex
    jobs_store: dict[str, dict[str, Any]] = request.app.state.analysis_jobs
    jobs_store[job_id] = _new_analysis_job(
        job_id=job_id,
        session_id=session_id,
        metadata_fields_input=metadata_fields,
    )

    asyncio.create_task(
        _run_analysis_job(
            app=request.app,
            job_id=job_id,
            session_id=session_id,
            serialized_documents=documents_payload,
            serialized_chunks_export=chunks_payload,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            client_modified_map=client_modified_map,
            metadata_fields=metadata_fields,
        )
    )

    return session_id, job_id


def _sse_message(event: str, data: dict[str, Any]) -> str:
    """Format a single Server-Sent Events message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _deserialize_uploads(payloads: list[dict[str, Any]]) -> list[UploadFile]:
    """Rebuild UploadFile objects from serialized request data."""
    return [
        UploadFile(
            filename=str(payload["filename"]),
            file=io.BytesIO(payload["content"]),
        )
        for payload in payloads
    ]


def _deserialize_upload(payload: dict[str, Any] | None) -> UploadFile | None:
    """Rebuild a single UploadFile from serialized request data."""
    if payload is None:
        return None
    return UploadFile(
        filename=str(payload["filename"]),
        file=io.BytesIO(payload["content"]),
    )


def _upload_page_context(
    *,
    request: Request,
    chunk_size: int,
    chunk_overlap: int,
    metadata_fields: str,
    error: str | None = None,
    message: str | None = None,
) -> dict[str, object]:
    """Build the upload page context used for fallback rendering."""
    session_id = _get_or_create_session_id(request)
    return {
        "request": request,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "metadata_fields_input": metadata_fields,
        "api_key_configured": _session_has_api_key(request.app, session_id),
        "error": error,
        "message": message,
        "upload_action": "/report/preview",
        "analysis_action": "/report/analyze",
        "api_key_action": "/report/api-key",
        "sample_corpora": list_sample_corpora(),
    }


def _build_report_payload(
    *,
    documents: list[Document],
    generated_chunks: list[Chunk],
    imported_chunks: list[Chunk],
    duplication_findings: list[DuplicationFinding],
    staleness_findings: list[StalenessFinding],
    supersession_findings: list[StalenessFinding],
    metadata_findings: list[MetadataFinding],
    metadata_summary: MetadataAuditSummary,
    contradiction_findings: list[ContradictionFinding],
    contradiction_stats: ContradictionRunStats,
    rot_findings: list[ROTFinding],
    health_score: CorpusHealthScore,
    messages: dict[str, object],
    chunk_size: int,
    chunk_overlap: int,
    metadata_fields: list[str],
) -> dict[str, object]:
    """Build the cached report payload used by exports."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "settings": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "metadata_fields": metadata_fields,
        },
        "health_score": health_score.model_dump(mode="json"),
        "documents": [document.model_dump(mode="json") for document in documents],
        "generated_chunks": [chunk.model_dump(mode="json") for chunk in generated_chunks],
        "existing_chunks": [chunk.model_dump(mode="json") for chunk in imported_chunks],
        "duplication_findings": [finding.model_dump(mode="json") for finding in duplication_findings],
        "staleness_findings": [finding.model_dump(mode="json") for finding in staleness_findings],
        "supersession_findings": [finding.model_dump(mode="json") for finding in supersession_findings],
        "metadata_findings": [finding.model_dump(mode="json") for finding in metadata_findings],
        "metadata_summary": metadata_summary.model_dump(mode="json"),
        "contradiction_findings": [finding.model_dump(mode="json") for finding in contradiction_findings],
        "contradiction_stats": contradiction_stats.model_dump(mode="json"),
        "rot_findings": [finding.model_dump(mode="json") for finding in rot_findings],
        "messages": messages,
    }


def _build_markdown_report(payload: dict[str, object]) -> str:
    """Render the cached report payload as a readable markdown summary."""
    health = payload.get("health_score", {})
    summary = health.get("summary", {}) if isinstance(health, dict) else {}
    rot_findings = payload.get("rot_findings", [])
    duplication_findings = payload.get("duplication_findings", [])
    staleness_findings = payload.get("staleness_findings", [])
    contradiction_findings = payload.get("contradiction_findings", [])
    metadata_findings = payload.get("metadata_findings", [])

    lines = [
        "# RAGLint Report",
        "",
        f"Generated at: {payload.get('generated_at', 'n/a')}",
        "",
        "## Health Score",
        "",
        f"- Overall score: {health.get('overall_score', 0):.1f}/100" if isinstance(health, dict) else "- Overall score: n/a",
        (
            f"- Projected score after removals: {health.get('projected_overall_score', 0):.1f}/100"
            if isinstance(health, dict)
            else "- Projected score after removals: n/a"
        ),
        "",
        "## Dimension Scores",
        "",
    ]

    if isinstance(health, dict):
        duplication_score = health.get("duplication_score")
        staleness_score = health.get("staleness_score")
        contradiction_score = health.get("contradiction_score")
        metadata_score = health.get("metadata_score")
        rot_score = health.get("rot_score")
        lines.extend(
            [
                f"- Duplication: {duplication_score:.1f}" if isinstance(duplication_score, (int, float)) else "- Duplication: N/A",
                f"- Staleness: {staleness_score:.1f}" if isinstance(staleness_score, (int, float)) else "- Staleness: N/A",
                (
                    f"- Contradictions: {contradiction_score:.1f}"
                    if isinstance(contradiction_score, (int, float))
                    else "- Contradictions: N/A"
                ),
                f"- Metadata: {metadata_score:.1f}" if isinstance(metadata_score, (int, float)) else "- Metadata: N/A",
                f"- ROT: {rot_score:.1f}" if isinstance(rot_score, (int, float)) else "- ROT: N/A",
            ]
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Total documents: {summary.get('total_documents', 0)}",
            (
                f"- Documents with issues: {summary.get('unique_documents_with_issues', 0)} of {summary.get('total_documents', 0)}"
            ),
            f"- Total chunks analyzed: {summary.get('total_chunks_analyzed', 0)}",
            f"- Duplicate clusters: {summary.get('duplicate_clusters', 0)}",
            f"- Stale chunks: {summary.get('stale_chunks', 0)}",
            f"- Contradictions found: {summary.get('contradictions_found', 0)}",
            f"- Average metadata completeness: {summary.get('average_metadata_completeness', 0) * 100:.0f}%",
            f"- Documents recommended for removal: {summary.get('documents_recommended_for_removal', 0)}",
            "",
            "## Key Findings",
            "",
        ]
    )

    unhealthy_rot = [
        finding
        for finding in rot_findings
        if isinstance(finding, dict) and finding.get("classifications") != ["healthy"]
    ]
    if unhealthy_rot:
        lines.append("### ROT")
        lines.append("")
        for finding in unhealthy_rot[:5]:
            classes = ", ".join(item.title() for item in finding.get("classifications", []))
            lines.append(
                f"- **{finding.get('document_filename', 'Unknown')}**: {classes}. {finding.get('recommendation', '')}"
            )
        lines.append("")

    if duplication_findings:
        lines.append("### Duplications")
        lines.append("")
        for finding in duplication_findings[:5]:
            if not isinstance(finding, dict):
                continue
            chunks = finding.get("chunks_involved", [])
            sources = ", ".join(
                chunk.get("source_filename", "Unknown")
                for chunk in chunks
                if isinstance(chunk, dict)
            )
            lines.append(
                f"- **{finding.get('finding_type', 'duplicate')}**: {sources}. {finding.get('recommendation', '')}"
            )
        lines.append("")

    stale_examples = [
        finding
        for finding in staleness_findings[:5]
        if isinstance(finding, dict)
    ]
    if stale_examples:
        lines.append("### Staleness")
        lines.append("")
        for finding in stale_examples:
            chunk = finding.get("chunk", {})
            lines.append(
                f"- **{chunk.get('source_filename', 'Unknown')}**: score {finding.get('staleness_score', 0):.2f}. {finding.get('recommendation', '')}"
            )
        lines.append("")

    if contradiction_findings:
        lines.append("### Contradictions")
        lines.append("")
        for finding in contradiction_findings[:5]:
            if not isinstance(finding, dict):
                continue
            chunks = finding.get("chunks_involved", [])
            left = chunks[0].get("source_filename", "Unknown") if len(chunks) > 0 and isinstance(chunks[0], dict) else "Unknown"
            right = chunks[1].get("source_filename", "Unknown") if len(chunks) > 1 and isinstance(chunks[1], dict) else "Unknown"
            lines.append(
                f"- **{left} vs {right}**: {finding.get('explanation', '')} Recommendation: {finding.get('recommendation', '')}"
            )
        lines.append("")

    if metadata_findings:
        lines.append("### Metadata")
        lines.append("")
        for finding in metadata_findings[:5]:
            if not isinstance(finding, dict):
                continue
            lines.append(
                f"- **{finding.get('document_filename', 'Unknown')}**: completeness {finding.get('completeness_score', 0) * 100:.0f}%. {finding.get('recommendation', '')}"
            )

    return "\n".join(lines).strip() + "\n"


def _coerce_embedding_run_result(
    result: EmbeddingRunResult | list[Chunk],
    chunks: list[Chunk],
) -> EmbeddingRunResult:
    """Normalize legacy test doubles or service results into one embedding result shape."""
    if isinstance(result, EmbeddingRunResult):
        return result
    return EmbeddingRunResult(
        chunks=result,
        embedded_chunk_count=sum(1 for chunk in chunks if chunk.embedding is not None),
    )


def _embedding_partial_failure_warning(result: EmbeddingRunResult, *, label: str = "uploaded documents") -> str:
    """Describe an embedding run that completed with some failed batches."""
    failed_chunk_count = sum(item.chunk_count for item in result.failed_batches)
    return (
        f"Embedding generation failed for {failed_chunk_count} text segment"
        f"{'s' if failed_chunk_count != 1 else ''} from the {label} after one retry. "
        "RAGLint continued with the embeddings that succeeded, so the analysis is partial but still useful."
    )


def _apply_session_cookie(response: Response, session_id: str) -> Response:
    """Attach the in-memory session identifier as a browser session cookie."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
    )
    return response
