# RAGLint

RAGLint -- Audit your document corpus before it enters your RAG pipeline.

RAGLint is a FastAPI application for reviewing document corpora before they are indexed for retrieval-augmented generation. It helps teams catch duplication, stale content, contradictions, weak metadata, and ROT before those issues become retrieval noise or model confusion in production.

<!-- Screenshot placeholder: add a screenshot of the final report dashboard here. -->

## Quick Start

### Option 1: Run with Docker

```bash
docker compose up --build
```

Open `http://localhost:8000`.

### Option 2: Run locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000`.

## How It Works

### Duplication detection

RAGLint chunks uploaded documents, generates OpenAI embeddings with `text-embedding-3-small`, and compares chunk vectors with cosine similarity. It flags exact duplicates, near-duplicates, and brownfield matches against an existing chunk export so you can spot content that is already in your index.

### Staleness scoring

The staleness pass combines document metadata dates with content heuristics such as explicit date references, version strings, and temporal language. Each chunk receives a score between `0.0` and `1.0`, and brownfield runs can flag older indexed chunks that appear to be superseded by newer uploads.

### Contradiction detection

RAGLint uses embeddings to narrow candidate pairs to related chunks that are similar enough to conflict but not so similar that they are duplicates. It then asks `gpt-4o-mini` to compare those passages and return structured contradiction findings when the claims cannot both be true.

### Metadata audit

The metadata audit checks each uploaded document for title quality, author or owner, created and modified dates, version information, and document type or category. It also looks for corpus-level consistency problems such as mixed date formats or missing dates across part of the corpus.

### ROT classification

The ROT pass aggregates signals from duplication, staleness, contradiction, metadata, and triviality heuristics to classify uploaded documents as healthy, redundant, outdated, trivial, or a combination of those labels. This helps convert a long report into direct cleanup guidance.

## Usage

### Full Corpus Audit

Upload a set of source documents and let RAGLint parse, chunk, embed, and score them. Use this mode when you want a report on the corpus you plan to index or re-index.

### Pre-ingestion Check

Upload new documents plus an existing chunk export in JSON or CSV format. RAGLint will compare the new content against the imported chunk set so you can catch duplicates, stale indexed content, and contradictions before adding more material to your live index.

## Supported File Formats

### Documents

- PDF
- DOCX
- Markdown
- TXT

### Chunk exports

- JSON
- CSV

## API Key

RAGLint uses OpenAI for embeddings and contradiction detection. A full audit requires an OpenAI API key because embeddings drive duplication analysis, brownfield comparisons, and downstream scoring. Contradiction detection adds targeted `gpt-4o-mini` calls on top of that embedding pipeline.

You can provide the key through the web UI for the current browser session only. The repository also includes an `.env.example` file so deployments can standardize secret handling, but the current app flow expects the key to be entered through the UI at analysis time.

Without an API key, the app can still parse and preview uploaded content, but it cannot run the complete audit pipeline.

## Configuration

RAGLint currently uses these defaults:

- Chunk size: `500` characters
- Chunk overlap: `50` characters
- Exact duplicate threshold: `0.98`
- Near-duplicate threshold: `0.85`
- Contradiction candidate similarity range: `0.70` to `0.95`
- Staleness threshold: `12` months
- Brownfield supersession topic similarity threshold: `0.70`

Chunk size and chunk overlap are configurable in the upload form. The similarity and staleness thresholds are defined in the service layer and can be adjusted in code if you need different operating points.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and pull request expectations.

## License

RAGLint is released under the MIT License. See [LICENSE](LICENSE).
