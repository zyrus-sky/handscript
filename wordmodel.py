"""
Styled line generation with Emuru (CVPR 2025, MIT, huggingface.co/blowing-up-groundhogs/emuru).

Emuru takes one 64 px-high image of someone's handwriting plus its transcription
and continues writing new text in that style. The style image is assembled from
words labeled in a glyph bank (glyphbank.py): only words whose labels spell the
whole word are used, so image and transcription match exactly.

Needs: torch (CUDA strongly recommended; ~3 GB VRAM), transformers, diffusers, einops.
First use downloads ~3 GB of weights to the Hugging Face cache.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import glyphbank

MODEL_ID = "blowing-up-groundhogs/emuru"
STYLE_HEIGHT = 64
STYLE_WORDS = 6          # words per style line (Emuru was trained on line images)
MIN_STYLE_WORDS = 4
MAX_CHUNK_CHARS = 24     # Emuru degrades on long requests; the engine lays out chunks
WIDTH_OK = (0.78, 1.3)   # generated width / expected width outside this = dropped or junk text

_model = None            # loaded once per process; ~30 s from cache


def _load_model():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModel
        # Remote code reviewed: model definition only (T5 + VAE), no I/O beyond from_pretrained.
        m = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        _model = m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _model


class StyleWriter:
    """Generates line images in the writer's style from a glyph bank's labeled words."""

    def __init__(self, bank_dir: str, seed: Optional[int] = None) -> None:
        self.dir = Path(bank_dir)
        self._rng = random.Random(seed)
        off = set(glyphbank.disabled_sources(self.dir))
        words = [w for w in glyphbank.load_words(self.dir) if w.get("source", "scan") not in off]
        # Whole, clean words: every piece labeled (no "_"), labels concatenate to the word.
        self._words = [w for w in words
                       if w["status"] == "done" and w["labels"] and "_" not in w["labels"]
                       and len("".join(w["labels"])) >= 2]
        if len(self._words) < MIN_STYLE_WORDS:
            raise ValueError(
                f"Need at least {MIN_STYLE_WORDS} fully labeled words in {bank_dir} "
                f"for the handwriting model (have {len(self._words)}). Label more at /label."
            )
        self.slant_deg = float(np.median([w["slant"] for w in self._words]))
        self._refs = glyphbank.page_refs(words)
        # One style for the whole document keeps size and hand consistent between
        # chunks; pick the candidate covering the most distinct letters, keep a backup.
        # Prefer the writer's natural, scanned words for the style: pad words are written
        # slowly and neatly, and the model copies whatever it is shown.
        scan = [w for w in self._words if w.get("source", "scan") != "pad"]
        if len(scan) >= STYLE_WORDS:
            self._words = scan
        cands = [self._style_line() for _ in range(10)]
        cands.sort(key=lambda c: -len(set(c[1].lower()) - {" "}))
        self._styles = cands[:2]

    # ── style image ─────────────────────────────────────────────────────

    def _style_line(self) -> Tuple[Image.Image, str, float, float]:
        """Random labeled words laid out on one baseline, white background, 64 px high.
        Returns (image, text, x-height, baseline y) — Emuru writes at the style's scale
        and on its baseline, so these hold for everything generated from it."""
        picks = self._rng.sample(self._words, min(STYLE_WORDS, len(self._words)))
        # Pages come at different resolutions: bring every word to a common x-height.
        ref = float(np.median([self._refs[glyphbank._page_key(w)] for w in picks]))
        ims, bases = [], []
        for w in picks:
            im = Image.open(self.dir / "words" / f"{w['id']}.png")
            r = ref / self._refs[glyphbank._page_key(w)]
            if abs(r - 1) > 0.02:
                im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.LANCZOS)
            ims.append(im)
            bases.append(w["baseline"] * r)
        above = max(bases)
        below = max(im.height - b for im, b in zip(ims, bases))
        gap = int(0.9 * ref)
        canvas = Image.new("RGBA", (sum(i.width for i in ims) + gap * (len(ims) + 1), int(above + below) + 16))
        x = gap
        for b, im in zip(bases, ims):
            canvas.paste(im, (x, int(round(8 + above - b))), im)
            x += im.width + gap
        bg = Image.new("RGB", canvas.size, "white")
        bg.paste(canvas, (0, 0), canvas)
        k = STYLE_HEIGHT / bg.height
        bg = bg.resize((round(bg.width * k), STYLE_HEIGHT), Image.LANCZOS)
        text = " ".join("".join(w["labels"]) for w in picks)
        xh = ref * k
        return bg, text, xh, (8 + above) * k

    def chars_per_px(self) -> float:
        """Rough text density of the style at 64 px height (for line wrapping)."""
        im, t = self._styles[0][:2]
        return len(t) / im.width

    # ── generation ──────────────────────────────────────────────────────

    def write_line(self, text: str) -> Tuple[Image.Image, float, float, bool]:
        """Generate `text`. Returns (alpha mask "L", baseline y, x-height, ok) at 64 px scale.
        Emuru sometimes stops early and drops words: a result much narrower (or wider)
        than the style's text density predicts is retried with the backup style, and
        ok=False tells the caller to split the text and try again."""
        import torch
        from torchvision.transforms import functional as F

        model = _load_model()
        dev = next(model.parameters()).device
        expect = len(text) / self.chars_per_px()
        best = None
        for attempt in range(len(self._styles)):
            style_img, style_text, xh, base = self._styles[attempt]
            x = F.normalize(F.to_tensor(style_img), [0.5] * 3, [0.5] * 3).to(dev)
            out = model.generate(style_text=style_text, gen_text=text, style_img=x,
                                 max_new_tokens=int(len(text) * 5) + 16)
            gray = np.array(out.convert("L"))
            ratio = gray.shape[1] / max(1.0, expect)
            if best is None or abs(np.log(ratio)) < abs(np.log(best[1])):
                best = (gray, ratio, xh, base)
            if WIDTH_OK[0] <= ratio <= WIDTH_OK[1]:
                break
            torch.cuda.empty_cache()
        gray, ratio, xh, base = best
        gray = gray.astype(np.float32)

        # Model ink is lighter than a scan: stretch this image's own ink→paper range.
        ink, paper = np.percentile(gray, 1), np.percentile(gray, 60)
        a = (paper - 8 - gray) / max(20.0, (paper - 8 - ink) * 0.75)
        alpha = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        cols = np.where(alpha.max(0) > 40)[0]
        if len(cols):
            alpha = alpha[:, cols[0]:cols[-1] + 1]
        return Image.fromarray(alpha, "L"), base, xh, WIDTH_OK[0] <= ratio <= WIDTH_OK[1]


def bank_ready(bank_dir: str) -> bool:
    """True if the bank has enough whole labeled words to style the model."""
    try:
        words = json.loads((Path(bank_dir) / "words.json").read_text(encoding="utf-8"))
    except OSError:
        return False
    off = set(glyphbank.disabled_sources(Path(bank_dir)))
    n = sum(1 for w in words if w["status"] == "done" and w["labels"] and w.get("source", "scan") not in off
            and "_" not in w["labels"] and len("".join(w["labels"])) >= 2)
    return n >= MIN_STYLE_WORDS
