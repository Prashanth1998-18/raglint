"""Tests for the built-in sample corpus experience."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app
from app.services.samples import load_sample_corpus


def test_sample_loading_service_builds_documents_and_chunks() -> None:
    """Sample corpus JSON should normalize into shared document and chunk models."""
    hr_sample = load_sample_corpus("hr-policies", chunk_size=220, chunk_overlap=30)
    tech_sample = load_sample_corpus("tech-docs", chunk_size=220, chunk_overlap=30)
    product_sample = load_sample_corpus("product-kb", chunk_size=220, chunk_overlap=30)

    assert len(hr_sample.documents) == 9
    assert len(tech_sample.documents) == 11
    assert len(product_sample.documents) == 11
    assert hr_sample.documents[0].metadata.embedded["title"] == "Travel Policy"
    assert all(document.source_path.startswith("sample://hr-policies/") for document in hr_sample.documents)
    assert hr_sample.chunks
    assert {chunk.parent_document_name for chunk in hr_sample.chunks} <= {
        document.metadata.filename for document in hr_sample.documents
    }


def test_upload_page_shows_sample_cards() -> None:
    """The upload page should present the built-in sample corpora ahead of the live upload form."""
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Try with sample data" in response.text
    assert "HR Policies" in response.text
    assert "Technical Docs" in response.text
    assert "Product Knowledge Base" in response.text
    assert "Analyze your documents" in response.text


def test_sample_route_uses_cached_report_payload(monkeypatch) -> None:
    """Starting a sample run should reuse a cached report payload and render sample prompts."""
    client = TestClient(app)

    preview_response = client.post(
        "/report/preview",
        data={"chunk_size": "80", "chunk_overlap": "10"},
        files=[
            ("documents", ("sample.txt", b"Travel approvals require manager review.", "text/plain")),
        ],
    )

    assert preview_response.status_code == 200
    cached_payload = client.post("/report/export/json").json()

    monkeypatch.setattr("app.routers.report.load_sample_report", lambda sample_id: cached_payload)
    start_response = client.post("/report/samples/hr-policies/start")

    assert start_response.status_code == 200
    job = start_response.json()

    status_payload = None
    for _ in range(20):
        status_response = client.get(job["status_url"])
        assert status_response.status_code == 200
        status_payload = status_response.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.01)

    assert status_payload is not None
    assert status_payload["status"] == "completed"

    report_response = client.get(job["report_url"])

    assert report_response.status_code == 200
    assert "This is a sample analysis. Ready to check your own documents?" in report_response.text
    assert "This is a pre-computed analysis of sample data. Upload your own documents to run a live analysis." in report_response.text
    assert "See what RAGLint finds in your documents." in report_response.text
