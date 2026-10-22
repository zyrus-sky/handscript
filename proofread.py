"""
Reading check: a small handwriting recogniser (Microsoft TrOCR small, MIT) reads a
just-written word back; if it can't recognise the word, the engine rewrites it with
other pieces (see HandwritingEngine.write_word, EngineConfig.proofread).

Model: microsoft/trocr-small-handwritten (~250 MB in the Hugging Face cache),
loaded from safetensors (torch < 2.6 refuses .bin weights). GPU if available.
"""

from __future__ import annotations

import re
from typing import Optional

from PIL import Image

MODEL_ID = "microsoft/trocr-small-handwritten"
_reader = None


class Reader:
    def __init__(self) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.proc = TrOCRProcessor.from_pretrained(MODEL_ID)
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, use_safetensors=True)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # float32: half precision only pays on GPUs with tensor cores; on a GTX 1650
        # it measured 3.5x SLOWER. Opt in with HANDSCRIPT_OCR_FP16=1.
        import os
        fp16 = self.dev == "cuda" and os.environ.get("HANDSCRIPT_OCR_FP16") == "1"
        self.dtype = torch.float16 if fp16 else torch.float32
        self.model.to(self.dev, self.dtype).eval()

    def read_batch(self, imgs, batch: int = 32) -> list:
        """Texts of many word images at once (one GPU pass per `batch` words)."""
        out = []
        for i in range(0, len(imgs), batch):
            pads = []
            for img in imgs[i:i + batch]:
                img = img.convert("RGB")
                pad = Image.new("RGB", (img.width + 24, img.height + 24), "white")
                pad.paste(img, (12, 12))
                pads.append(pad)
            px = self.proc(images=pads, return_tensors="pt").pixel_values.to(self.dev, self.dtype)
            with self.torch.inference_mode():
                ids = self.model.generate(px, max_new_tokens=24)
            out += [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]
        return out

    def read(self, img: Image.Image) -> str:
        """Text of a single word / short line image (dark ink on light paper)."""
        img = img.convert("RGB")
        # TrOCR expects a bit of margin around the word
        pad = Image.new("RGB", (img.width + 24, img.height + 24), "white")
        pad.paste(img, (12, 12))
        px = self.proc(images=pad, return_tensors="pt").pixel_values.to(self.dev, self.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(px, max_new_tokens=24)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def reader() -> Reader:
    global _reader
    if _reader is None:
        _reader = Reader()
    return _reader


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def mismatch(read: str, want: str) -> float:
    """0 = read exactly as intended, 1 = nothing in common (edit distance / length)."""
    a, b = _norm(read), _norm(want)
    if not b:
        return 0.0
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))
