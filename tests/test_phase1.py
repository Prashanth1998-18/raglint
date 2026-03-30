"""Phase 1 tests for parsing, chunking, normalization, and the FastAPI flow."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from fastapi import UploadFile

from app.main import app
from app.errors import DocumentParsingError
from app.models.document import (
    ContradictionFinding,
    ContradictionRunStats,
    Document,
    FindingChunk,
)
from app.services.chunker import RecursiveCharacterChunker
from app.services.metadata import build_document_metadata
from app.services.parser import ChunkExportParser, DocumentParser


def test_markdown_parser_extracts_frontmatter_and_modified_date() -> None:
    """Markdown uploads should preserve body text and parsed frontmatter."""
    upload = UploadFile(
        filename="guide.md",
        file=io.BytesIO(
            b"---\ntitle: Guide\nteam: Search\n---\n\nRAGLint checks the corpus before indexing."
        ),
    )
    modified_at = datetime(2026, 3, 15, 12, 30, tzinfo=timezone.utc)

    document = _run(DocumentParser().parse_upload(upload, client_modified_at=modified_at))

    assert document.metadata.filename == "guide.md"
    assert document.metadata.modified_at == modified_at
    assert document.metadata.frontmatter["title"] == "Guide"
    assert document.text.startswith("RAGLint checks")


def test_chunker_generates_overlapping_chunks() -> None:
    """Recursive chunking should keep parent references and overlapping windows."""
    document = Document(
        text="Paragraph one.\n\nParagraph two has more detail.\n\nParagraph three closes the loop.",
        metadata=build_document_metadata(filename="notes.txt", file_size=88),
        source_path="notes.txt",
    )

    chunks = RecursiveCharacterChunker(chunk_size=35, chunk_overlap=8).chunk_document(document)

    assert len(chunks) >= 2
    assert all(chunk.parent_document_id == document.id for chunk in chunks)
    assert chunks[1].position.start_char < chunks[0].position.end_char


def test_chunk_export_parser_normalizes_json_and_csv() -> None:
    """Chunk export uploads should normalize both JSON and CSV formats."""
    json_upload = UploadFile(
        filename="chunks.json",
        file=io.BytesIO(
            json.dumps(
                [
                    {
                        "text": "Existing JSON chunk",
                        "document_id": "doc-json",
                        "chunk_index": 2,
                        "start_char": 10,
                        "end_char": 28,
                        "embedding": [0.1, 0.2, 0.3],
                        "metadata": {"source": "json"},
                    }
                ]
            ).encode("utf-8")
        ),
    )
    csv_upload = UploadFile(
        filename="chunks.csv",
        file=io.BytesIO(
            b"text,document_name,index,start,end,topic\nExisting CSV chunk,guide.md,4,100,118,onboarding\n"
        ),
    )

    parser = ChunkExportParser()
    json_chunks = _run(parser.parse_upload(json_upload))
    csv_chunks = _run(parser.parse_upload(csv_upload))

    assert json_chunks[0].parent_document_id == "doc-json"
    assert json_chunks[0].embedding == [0.1, 0.2, 0.3]
    assert json_chunks[0].metadata["source"] == "json"
    assert csv_chunks[0].parent_document_name == "guide.md"
    assert csv_chunks[0].position.chunk_index == 4
    assert csv_chunks[0].metadata["topic"] == "onboarding"


def test_preview_route_renders_documents_and_chunks() -> None:
    """The preview route should render parsed documents and imported chunks."""
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={
            "chunk_size": "40",
            "chunk_overlap": "10",
            "client_modified_map": json.dumps(
                {"sample.md": datetime(2026, 3, 15, 8, 0, tzinfo=timezone.utc).isoformat()}
            ),
        },
        files=[
            (
                "documents",
                ("sample.md", b"# Heading\n\nSome markdown body for preview rendering.", "text/markdown"),
            ),
            (
                "chunks_export",
                ("chunks.json", b'[{"text":"Imported chunk","document_name":"existing.md"}]', "application/json"),
            ),
        ],
    )

    assert response.status_code == 200
    assert "sample.md" in response.text
    assert "Imported chunk" in response.text
    assert "Generated text segments" in response.text
    assert "Staleness" in response.text
    assert "Document information" in response.text
    assert "Contradictions" in response.text
    assert "Content quality" in response.text


def test_upload_form_exposes_metadata_field_configuration() -> None:
    """The upload form should show the configurable metadata field input with defaults."""
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Expected metadata fields" in response.text
    assert "Analyze documents" in response.text
    assert "title, author, date modified" in response.text


def test_preview_route_handles_missing_api_key_gracefully() -> None:
    """The report should render even when duplication analysis is skipped."""
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={"chunk_size": "80", "chunk_overlap": "10"},
        files=[
            ("documents", ("one.txt", b"same text", "text/plain")),
            ("documents", ("two.txt", b"same text", "text/plain")),
        ],
    )

    assert response.status_code == 200
    assert "An OpenAI API key is required for analysis. Click the settings icon to add your key." in response.text
    assert "Duplication analysis skipped because no OpenAI API key was provided." in response.text
    assert "Staleness" in response.text
    assert "Document information" in response.text
    assert "Contradictions" in response.text
    assert "Content quality" in response.text
    assert "Contradiction detection requires an OpenAI API key. Provide one to enable this analysis." in response.text


def test_preview_route_uses_custom_metadata_fields_in_audit_output() -> None:
    """The report should reflect user-configured metadata fields in the metadata section."""
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={
            "chunk_size": "80",
            "chunk_overlap": "10",
            "metadata_fields": "title, owner, jurisdiction",
        },
        files=[
            (
                "documents",
                (
                    "policy.md",
                    b"---\nowner: Legal Ops\n---\n\nPolicy body.",
                    "text/markdown",
                ),
            ),
        ],
    )

    assert response.status_code == 200
    assert "Configured fields" in response.text
    assert "title, owner, jurisdiction" in response.text
    assert "jurisdiction" in response.text


def test_preview_route_renders_duplication_findings_with_mocked_embeddings(monkeypatch) -> None:
    """Mocked embeddings should drive a full duplication finding through the report route."""

    async def fake_embed_chunks(self, chunks, api_key):
        for chunk in chunks:
            chunk.embedding = [1.0, 0.0] if chunk.text == "same text" else [0.0, 1.0]
        return chunks

    async def fake_contradiction_run(self, candidate_pairs, api_key):
        return [], ContradictionRunStats(candidate_pairs_considered=len(candidate_pairs))

    monkeypatch.setattr("app.routers.report.OpenAIEmbeddingService.embed_chunks", fake_embed_chunks)
    monkeypatch.setattr("app.routers.report.ContradictionDetectionPass.run", fake_contradiction_run)
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={
            "chunk_size": "80",
            "chunk_overlap": "10",
            "openai_api_key": "sk-test",
        },
        files=[
            ("documents", ("one.txt", b"same text", "text/plain")),
            ("documents", ("two.txt", b"same text", "text/plain")),
        ],
    )

    assert response.status_code == 200
    assert "Exact duplicate" in response.text
    assert "one.txt" in response.text
    assert "two.txt" in response.text
    assert "Staleness" in response.text
    assert "Document information" in response.text
    assert "Contradictions" in response.text
    assert "Content quality" in response.text


def test_preview_route_renders_contradiction_findings_with_mocked_analysis(monkeypatch) -> None:
    """Mocked contradiction analysis should render the contradiction section without live API calls."""

    async def fake_embed_chunks(self, chunks, api_key):
        for chunk in chunks:
            if "allowed" in chunk.text:
                chunk.embedding = [1.0, 0.0]
            else:
                chunk.embedding = [0.8, 0.6]
        return chunks

    async def fake_contradiction_run(self, candidate_pairs, api_key):
        finding = ContradictionFinding(
            chunks_involved=[
                FindingChunk(
                    chunk_id="a",
                    text="Remote work is allowed two days per week.",
                    source_filename="policy-a.txt",
                    parent_document_id="doc-a",
                    source_scope="new",
                ),
                FindingChunk(
                    chunk_id="b",
                    text="Remote work is not allowed for this team.",
                    source_filename="policy-b.txt",
                    parent_document_id="doc-b",
                    source_scope="new",
                ),
            ],
            similarity_score=0.8,
            severity="high",
            claim_a="Remote work is allowed two days per week.",
            claim_b="Remote work is not allowed for this team.",
            explanation="The passages make opposite claims about whether remote work is allowed.",
            why_it_matters="Conflicting chunks can cause the model to answer with the wrong policy.",
            recommendation="Resolve the policy conflict before indexing both passages.",
        )
        stats = ContradictionRunStats(
            llm_calls_made=len(candidate_pairs),
            candidate_pairs_considered=len(candidate_pairs),
            prompt_tokens=120,
            completion_tokens=40,
            estimated_cost_usd=0.000084,
        )
        return [finding], stats

    monkeypatch.setattr("app.routers.report.OpenAIEmbeddingService.embed_chunks", fake_embed_chunks)
    monkeypatch.setattr("app.routers.report.ContradictionDetectionPass.run", fake_contradiction_run)
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={
            "chunk_size": "120",
            "chunk_overlap": "10",
            "openai_api_key": "sk-test",
        },
        files=[
            ("documents", ("policy-a.txt", b"Remote work is allowed two days per week.", "text/plain")),
            ("documents", ("policy-b.txt", b"Remote work is not allowed for this team.", "text/plain")),
        ],
    )

    assert response.status_code == 200
    assert "Contradictions" in response.text
    assert "Remote work is allowed two days per week." in response.text
    assert "Remote work is not allowed for this team." in response.text
    assert "Estimated cost" in response.text


def test_preview_route_rejects_empty_submission() -> None:
    """Submitting without any documents should return a helpful validation error."""
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={"chunk_size": "80", "chunk_overlap": "10"},
    )

    assert response.status_code == 400
    assert "Please select at least one document to analyze." in response.text


def test_preview_route_rejects_unsupported_document_type() -> None:
    """Unsupported uploads should fail before parsing begins."""
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={"chunk_size": "80", "chunk_overlap": "10"},
        files=[
            ("documents", ("spreadsheet.xlsx", b"not-a-supported-document", "application/vnd.ms-excel")),
        ],
    )

    assert response.status_code == 400
    assert "File spreadsheet.xlsx is not supported. RAGLint accepts PDF, DOCX, Markdown, and TXT files." in response.text


def test_preview_route_shows_warning_when_one_file_cannot_be_parsed(monkeypatch) -> None:
    """One unreadable file should be skipped while the rest of the analysis completes."""
    original_parse_upload = DocumentParser.parse_upload

    async def fake_parse_upload(self, upload_file, *, client_modified_at=None):
        if upload_file.filename == "broken.pdf":
            raise DocumentParsingError("unable to extract text")
        return await original_parse_upload(self, upload_file, client_modified_at=client_modified_at)

    monkeypatch.setattr("app.routers.report.DocumentParser.parse_upload", fake_parse_upload)
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={"chunk_size": "80", "chunk_overlap": "10"},
        files=[
            ("documents", ("good.txt", b"Readable text for the report.", "text/plain")),
            ("documents", ("broken.pdf", b"%PDF-bad", "application/pdf")),
        ],
    )

    assert response.status_code == 200
    assert "1 file could not be parsed: broken.pdf (reason: unable to extract text)." in response.text
    assert "good.txt" in response.text


def test_preview_route_rejects_chunk_export_without_text_field() -> None:
    """Chunk imports without a text field should show a specific validation error."""
    client = TestClient(app)

    response = client.post(
        "/report/preview",
        data={"chunk_size": "80", "chunk_overlap": "10"},
        files=[
            ("documents", ("guide.txt", b"Guide text.", "text/plain")),
            ("chunks_export", ("chunks.json", b'[{"document_name":"existing.md"}]', "application/json")),
        ],
    )

    assert response.status_code == 400
    assert "Chunks export is missing required" in response.text
    assert "text" in response.text


def _run(awaitable):
    """Run a single coroutine without introducing a pytest async dependency."""
    import asyncio

    return asyncio.run(awaitable)
