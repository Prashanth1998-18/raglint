"""Precompute cached sample reports for the built-in demo corpora."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Iterable
from uuid import uuid4

from app.main import create_app
from app.routers.report import _prepare_report_result
from app.services.passes.metadata import DEFAULT_METADATA_FIELD_INPUT
from app.services.samples import build_sample_uploads, get_sample_definition, list_sample_corpora


async def _precompute_samples(sample_ids: Iterable[str], *, openai_api_key: str) -> None:
    """Generate cached report payloads for one or more sample corpora."""
    app = create_app()

    for sample_id in sample_ids:
        definition = get_sample_definition(sample_id)
        uploads = build_sample_uploads(sample_id)
        try:
            report_result = await _prepare_report_result(
                app=app,
                session_id=uuid4().hex,
                documents=uploads,
                chunks_export=None,
                chunk_size=500,
                chunk_overlap=50,
                client_modified_map=None,
                openai_api_key=openai_api_key,
                metadata_fields=DEFAULT_METADATA_FIELD_INPUT,
            )
        finally:
            for upload in uploads:
                await upload.close()

        definition.report_path.write_text(
            json.dumps(report_result["report_payload"], indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {definition.report_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sample precompute script."""
    parser = argparse.ArgumentParser(description="Precompute cached reports for the built-in RAGLint sample corpora.")
    parser.add_argument(
        "--openai-api-key",
        required=True,
        help="OpenAI API key used to run embeddings and contradiction detection during precompute.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        choices=[sample.sample_id for sample in list_sample_corpora()],
        help="Specific sample corpus to precompute. Repeat to generate more than one. Defaults to all samples.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the precompute workflow from the command line."""
    args = _parse_args()
    sample_ids = args.sample or [sample.sample_id for sample in list_sample_corpora()]
    asyncio.run(_precompute_samples(sample_ids, openai_api_key=args.openai_api_key))


if __name__ == "__main__":
    main()
