"""Test configuration for local package imports."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if ARCHIVE.exists() and str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

collect_ignore_glob = ["pytest-cache-files-*"]
