"""
Local LLM helpers (Ollama). Used to tidy pasted text into the Markdown the
document renderer understands, and to read figures / scanned pages from PDFs.

Model: $HANDSCRIPT_LLM (default gemma4:12b); server: $OLLAMA_HOST (default
http://localhost:11434). Thinking is switched off: Gemma 4 otherwise spends the
whole token budget on hidden reasoning and returns nothing.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from typing import List, Optional

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("HANDSCRIPT_LLM", "gemma4:12b")

TIDY_PROMPT = """You prepare text for a program that hand-writes documents. Rewrite the TEXT below as clean Markdown, following these rules exactly:
- Keep the author's wording, order and facts. Do not add, remove, summarise or explain anything.
- Headings as #, ## or ###. Bullet points as "- ". Numbered steps as "1. ", "2. ".
- Anything tabular (columns, "key: value" grids, comparisons) as a Markdown pipe table with a header row.
- A process, flow, algorithm steps with branches, or a described diagram: ALSO add it as a ```mermaid block using "flowchart TD", short node labels, and |label| on decision edges.
- Math as LaTeX: inline $...$, display equations on their own as $$...$$.
- Code stays in ``` fences.
- Remove decorative symbols, emojis and HTML. Keep citation marks like [1].
Output ONLY the Markdown, no comments.

TEXT:
"""

FIGURE_PROMPT = """This image is a figure cut from a document. Reply with exactly one of:
- a flowchart / block diagram / process: a ```mermaid block ("flowchart TD" or "flowchart LR") reproducing its boxes, labels and arrows;
- a table: a Markdown pipe table;
- a formula: the LaTeX in $$...$$;
- anything else (photo, plot, circuit, drawing): one line "SKETCH: <short description>".
Output nothing else."""

OCR_PROMPT = """Transcribe this document page to Markdown. Keep the exact wording. Use #/## headings, "- " bullets, "1. " numbered lists, pipe tables for tables, $...$ / $$...$$ LaTeX for math, and a ```mermaid flowchart TD block for any flowchart. Output only the Markdown."""


def _post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def available() -> dict:
    """{'ok': bool, 'model': name, 'error': str}"""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    except OSError as e:
        return {"ok": False, "model": MODEL, "error": f"Ollama not reachable at {OLLAMA}: {e}"}
    if MODEL not in names and f"{MODEL}:latest" not in names:
        return {"ok": False, "model": MODEL, "error": f"model {MODEL} not pulled (have: {', '.join(names)})"}
    return {"ok": True, "model": MODEL, "error": ""}


def generate(prompt: str, images: Optional[List[bytes]] = None, max_tokens: int = 3000,
             timeout: float = 1800) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "think": False,
               "options": {"temperature": 0.15, "num_predict": max_tokens}}
    if images:
        payload["images"] = [base64.b64encode(b).decode() for b in images]
    return _post("/api/generate", payload, timeout).get("response", "").strip()


def _unfence(md: str) -> str:
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", md.strip(), re.S)
    return m.group(1) if m else md


def tidy(text: str, chunk_chars: int = 2500) -> str:
    """Pasted text -> renderer Markdown, in chunks on paragraph boundaries."""
    paras = re.split(r"\n\s*\n", text.strip())
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > chunk_chars:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    out = [_unfence(generate(TIDY_PROMPT + c, max_tokens=max(800, len(c)))) for c in chunks]
    return "\n\n".join(o for o in out if o)


def figure_to_markdown(png: bytes) -> str:
    return _unfence(generate(FIGURE_PROMPT, images=[png], max_tokens=900))


def ocr_page(png: bytes) -> str:
    return _unfence(generate(OCR_PROMPT, images=[png], max_tokens=3000))


OCR_MODEL = os.environ.get("HANDSCRIPT_OCR", MODEL)   # the OCR GGUF answers empty via Ollama


def read_word(png: bytes) -> str:
    """Read one handwritten word/piece (bank quality check). Uses the OCR model."""
    payload = {"model": OCR_MODEL, "prompt": "Read the handwritten text in this image. Output only the text.",
               "images": [base64.b64encode(png).decode()], "stream": False, "think": False,
               "options": {"temperature": 0, "num_predict": 24}}
    lines = _post("/api/generate", payload, 120).get("response", "").strip().splitlines()
    return lines[0].strip() if lines else ""
