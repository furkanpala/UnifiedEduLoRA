"""Shared utilities for the MIT three-conditioning experiment scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple


def load_openai_key(*search_paths: str) -> Tuple[str, Optional[str]]:
    """
    Load the OpenAI API key from (in order):
        1. OPENAI_API_KEY env var
        2. each path in `search_paths` that exists and starts with 'sk-'

    Returns (key, source) where source is the path or '<env var>' or None.
    """
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env.startswith("sk-"):
        return env, "<env var>"
    for path in search_paths:
        if not path:
            continue
        p = Path(path)
        if p.exists():
            content = p.read_text(encoding="utf-8").strip()
            if content.startswith("sk-"):
                return content, str(p)
    return "", None


def default_key_search_paths(repo_dir: str, drive_dir: Optional[str] = None) -> list[str]:
    paths = [f"{repo_dir}/openai_api_key"]
    if drive_dir:
        paths.append(f"{drive_dir}/openai_api_key")
    return paths
