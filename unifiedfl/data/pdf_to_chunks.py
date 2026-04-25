"""
Convert a PDF file into plain-text chunks suitable for generate_qa.py.

BEFORE YOU RUN THIS — things you will likely need to adjust:
  - SKIP_PAGES      : page indices (0-based) to exclude (cover, references, appendix, etc.)
  - SKIP_IF_FEWER   : minimum words a page must contain to be included
  - TARGET_WORDS    : target chunk size; increase if your content is dense, decrease if sparse
  - The header/footer stripper heuristic (_strip_boilerplate) works by dropping the first
    and last line of each page if they are short. Adjust the threshold or remove entirely
    if your PDF does not have running headers/footers.

Requirements:
  pip install pymupdf

Usage:
  python pdf_to_chunks.py lecture.pdf
  python pdf_to_chunks.py lecture.pdf --output my_chunks.json
  python pdf_to_chunks.py lecture.pdf --skip-pages 0 1 45 46 --target-words 300
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List


# ── Adjust these to match your document ──────────────────────────────────────

SKIP_PAGES    = []   # 0-based page indices to skip (e.g. cover, ToC, references)
SKIP_IF_FEWER = 40   # pages with fewer words than this are silently dropped
TARGET_WORDS  = 250  # target words per output chunk (will stay within ~150–400)
MIN_WORDS     = 150  # chunk must have at least this many words to be emitted
MAX_WORDS     = 400  # hard upper limit; chunk is split if it exceeds this


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _strip_boilerplate(text: str) -> str:
    """
    Drop the first and last line of each page block if they look like
    running headers or footers (short lines, page numbers, etc.).
    Adjust or remove this function if your PDF does not have such elements.
    """
    lines = text.splitlines()
    if len(lines) > 4:
        if len(lines[0].split()) <= 6:
            lines = lines[1:]
        if len(lines[-1].split()) <= 6:
            lines = lines[:-1]
    return "\n".join(lines)


def _clean(text: str) -> str:
    text = _strip_boilerplate(text)
    # Re-join hyphenated line-breaks (e.g. "opti-\nmisation" → "optimisation")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are purely numeric (page numbers isolated on a line)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Flatten single newlines within a paragraph to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_into_chunks(text: str, target: int, min_w: int, max_w: int) -> List[str]:
    """Split a long text into chunks of approximately `target` words."""
    words = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(words):
        end = min(i + target, len(words))
        # Try to break at a sentence boundary within a ±30-word window
        candidate = " ".join(words[i:end])
        if end < len(words):
            for boundary in range(end, max(i + min_w, end - 30), -1):
                snippet = " ".join(words[i:boundary])
                if re.search(r"[.!?]\s*$", snippet):
                    candidate = snippet
                    end = boundary
                    break
        chunk = candidate.strip()
        if len(chunk.split()) >= min_w:
            # Hard-split any chunk that exceeds max_w
            chunk_words = chunk.split()
            while len(chunk_words) > max_w:
                chunks.append(" ".join(chunk_words[:max_w]))
                chunk_words = chunk_words[max_w:]
            if len(chunk_words) >= min_w:
                chunks.append(" ".join(chunk_words))
            elif chunks:
                # Append leftover to last chunk rather than emitting a tiny one
                chunks[-1] += " " + " ".join(chunk_words)
        i = end
    return chunks


# ── Main extraction ───────────────────────────────────────────────────────────

def pdf_to_chunks(
    pdf_path: Path,
    skip_pages: List[int] = SKIP_PAGES,
    skip_if_fewer: int = SKIP_IF_FEWER,
    target_words: int = TARGET_WORDS,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    pages_text: List[str] = []

    for page_num, page in enumerate(doc):
        if page_num in skip_pages:
            continue
        raw = page.get_text("text")
        cleaned = _clean(raw)
        if len(cleaned.split()) < skip_if_fewer:
            continue
        pages_text.append(cleaned)

    doc.close()

    full_text = "\n\n".join(pages_text)
    return _split_into_chunks(full_text, target_words, min_words, max_words)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a PDF to plain-text chunks for generate_qa.py"
    )
    p.add_argument("pdf", help="Path to the PDF file")
    p.add_argument("--output", default=None,
                   help="Output JSON file (default: <pdf_stem>_chunks.json)")
    p.add_argument("--skip-pages", type=int, nargs="*", default=SKIP_PAGES,
                   help="0-based page indices to skip")
    p.add_argument("--target-words", type=int, default=TARGET_WORDS,
                   help=f"Target words per chunk (default: {TARGET_WORDS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        print(f"ERROR: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {pdf_path} ...")
    chunks = pdf_to_chunks(
        pdf_path,
        skip_pages=args.skip_pages or [],
        target_words=args.target_words,
    )

    out_path = Path(args.output) if args.output else pdf_path.with_name(pdf_path.stem + "_chunks.json")
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Extracted {len(chunks)} chunks → {out_path}")
    print(f"Word counts: min={min(len(c.split()) for c in chunks)}  "
          f"max={max(len(c.split()) for c in chunks)}  "
          f"avg={sum(len(c.split()) for c in chunks)//len(chunks)}")
    print("\nNext step: feed these chunks into generate_qa.py")
    print("  my_chunks = json.load(open(\"" + str(out_path) + "\"))")


if __name__ == "__main__":
    main()
