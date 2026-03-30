# Contributing to RAGLint

Thanks for contributing to RAGLint. Keep changes small, reviewable, and aligned with the existing FastAPI structure.

## Development setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the development server:

```bash
uvicorn app.main:app --reload
```

4. Open `http://localhost:8000`.

## Running tests

Run the current test suite before opening a pull request:

```bash
pytest -q tests -p no:cacheprovider
```

If you touch packaging or imports, also run:

```bash
python -m compileall app tests
```

## Code style

- Use type hints for new Python code.
- Add docstrings to modules, classes, and non-trivial functions.
- Keep changes focused and avoid unrelated refactors.
- Preserve the existing project structure and naming conventions.
- Avoid introducing new dependencies unless the change clearly needs them.

## Pull requests

- Open a focused pull request with a clear summary of what changed.
- Include test coverage for logic changes whenever practical.
- Call out any manual verification steps for UI or Docker-related updates.
- Update the README or other docs when behavior, setup, or configuration changes.
