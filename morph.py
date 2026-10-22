"""
Learned font -> handwriting morphs.

For every single-letter piece of the writer (scans and pad), each candidate font's
version of that letter is registered onto it: a smooth deformation (12 numbers:
shift, linear and quadratic terms in x and y, in x-height units) moves the font's
centre line onto the writer's centre line (two-way chamfer distance, Powell search,
mild regularisation). The residual says how well that font's *form* fits, so the
best font per sample also separates allographs (e.g. one-storey vs two-storey 'a').

The model (banks/<name>/morph.json) keeps, per character and form: the font, how
often it occurs, and the mean / covariance of the deformation. Characters never
seen get the writer's general tendency for their class (small, tall, capital,
descender, digit). vectorfont.render applies a deformation sampled from this.

  python morph.py --bank banks/mine           fit and save
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import vectorfont

XH = 40.0                       # working x-height in px
NP = 12                         # deformation parameters
REG = 4.0                       # regularisation weight
# bending (quadratic / cross) terms are penalised harder than shift / stretch / slant,
# so letters bend gently instead of distorting
REG_W = np.array([0.3, 1, 1, 6, 6, 6, 0.3, 1, 1, 6, 6, 6], np.float64)
TALL = set("bdfhklt")
DESC = set("gjpqy")


def char_class(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if ch.isupper():
        return "cap"
    if ch in DESC:
        return "desc"
    if ch in TALL:
        return "tall"
    return "small"


def field(p: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Deformation (in x-heights) at normalised coords u (right), v (down from baseline)."""
    dx = p[0] + p[1] * u + p[2] * v + p[3] * u * v + p[4] * u * u + p[5] * v * v
    dy = p[6] + p[7] * u + p[8] * v + p[9] * u * v + p[10] * u * u + p[11] * v * v
    return dx, dy


def _thin(m: np.ndarray) -> np.ndarray:
    from engine import _thin as thin
    return thin(m)


def _skeleton_pts(mask: np.ndarray, base_row: float, step: int = 1) -> np.ndarray:
    """Centre-line points (x, y-from-baseline) in px of a binary mask."""
    sk = _thin((mask > 0).astype(np.uint8))
    ys, xs = np.nonzero(sk)
    pts = np.stack([xs.astype(np.float64), ys - base_row], 1)
    return pts[::step] if len(pts) > 400 else pts


def font_skeleton(path: str, ch: str) -> Optional[np.ndarray]:
    res = vectorfont.render(path, ch, XH, variation=0.0, morph=False)
    if res is None:
        return None
    mask, base = res
    a = (np.array(mask) > 110).astype(np.uint8)
    if a.sum() < 10:
        return None
    pts = _skeleton_pts(a, base, step=2)
    pts[:, 0] -= pts[:, 0].min()
    return pts


def writer_skeleton(bank: Path, g: dict) -> Optional[np.ndarray]:
    try:
        a = np.array(Image.open(bank / "glyphs" / g["file"]))
    except OSError:
        return None
    k = XH / (g.get("ref_xh") or 30.0) * g.get("size_fix", 1.0)
    a = cv2.resize(a, (max(2, round(a.shape[1] * k)), max(2, round(a.shape[0] * k))), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0) > 100
    if m.sum() < 10:
        return None
    pts = _skeleton_pts(m.astype(np.uint8), g["baseline"] * k)
    pts[:, 0] -= pts[:, 0].min()
    return pts


def _sub(pts: np.ndarray, n: int = 80) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[np.linspace(0, len(pts) - 1, n).astype(int)]


def fit(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Deformation p moving src points onto dst; returns (p, residual in x-heights).
    Least squares on two-way nearest-point distances (+ small regularisation)."""
    src, dst = _sub(src), _sub(dst)
    tree_dst = cKDTree(dst)
    u, v = src[:, 0] / XH, src[:, 1] / XH
    def moved(p):
        dx, dy = field(p, u, v)
        return np.stack([src[:, 0] + XH * dx, src[:, 1] + XH * dy], 1)
    def residuals(p):
        q = moved(p)
        d1 = tree_dst.query(q)[0]
        d2 = cKDTree(q).query(dst)[0]
        return np.concatenate([d1 / XH, d2 / XH, np.sqrt(REG * 1e-3 * REG_W) * p])
    p0 = np.zeros(NP)
    p0[0] = (dst[:, 0].mean() - src[:, 0].mean()) / XH
    p0[6] = (dst[:, 1].mean() - src[:, 1].mean()) / XH
    res = least_squares(residuals, p0, method="trf", max_nfev=300, x_scale=0.2)
    q = moved(res.x)
    resid = (tree_dst.query(q)[0].mean() + cKDTree(q).query(dst)[0].mean()) / (2 * XH)
    return res.x, float(resid)


def build_model(bank: Path, fonts: List[str], log=print) -> dict:
    meta = json.loads((bank / "bank.json").read_text(encoding="utf-8"))
    samples = [g for g in meta["glyphs"] if len(g["label"]) == 1 and g["label"].isalnum()]
    font_sk: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    fits = []                                    # (char, font, params, resid)
    widths: Dict[str, List[float]] = {}          # writer's centre-line width per char (x-heights)
    for n, g in enumerate(samples, 1):
        ch = g["label"]
        if not (bank / "glyphs" / g["file"]).exists():     # bank rebuilt meanwhile
            continue
        dst = writer_skeleton(bank, g)
        if dst is None or len(dst) < 8:
            continue
        best = None
        # shortlist the 2 fonts whose letter looks most like this sample, fit only those
        from glyphbank import shape_sig, _iou
        try:
            a = np.array(Image.open(bank / "glyphs" / g["file"]))
        except OSError:
            continue
        sig = shape_sig(a)
        scored = []
        for f in fonts:
            r = vectorfont.render(f, ch, XH, variation=0.0, morph=False)
            if r is not None:
                scored.append((_iou(sig, shape_sig(np.array(r[0]))), f))
        shortlist = [f for _, f in sorted(scored, reverse=True)[:2]]
        for f in shortlist:
            if (f, ch) not in font_sk:
                font_sk[(f, ch)] = font_skeleton(f, ch)
            src = font_sk[(f, ch)]
            if src is None or len(src) < 8:
                continue
            p, r = fit(src, dst)
            if best is None or r < best[3]:
                best = (ch, f, p, r)
        if best:
            fits.append(best)
            widths.setdefault(ch, []).append(float(np.ptp(dst[:, 0]) / XH))
        if n % 20 == 0:
            log(f"  fitted {n}/{len(samples)}")
    # per character: allographs = best-matching font; keep forms that fit reasonably
    chars: Dict[str, dict] = {}
    for ch in sorted({f[0] for f in fits}):
        forms = {}
        for _, font, p, r in [f for f in fits if f[0] == ch]:
            forms.setdefault(font, []).append((p, r))
        out = []
        for font, lst in forms.items():
            good = [(p, r) for p, r in lst if r < 0.05] or lst      # poorly fitting samples out
            P = np.array([p for p, _ in good])
            R = np.array([r for _, r in good])
            cov = np.cov(P.T) if len(P) > 1 else np.zeros((NP, NP))
            # median: one mis-cut sample must not drag the form
            out.append({"font": font, "count": len(lst), "mean": np.median(P, 0).tolist(),
                        "cov": np.asarray(cov).tolist(), "resid": float(np.median(R))})
        out.sort(key=lambda f: -f["count"])
        chars[ch] = {"forms": out, "width": float(np.median(widths[ch])) if widths.get(ch) else None}
    # writer's tendency per character class (for characters never seen)
    classes = {}
    for cls in ("small", "tall", "desc", "cap", "digit"):
        P = np.array([p for c, _, p, r in fits if char_class(c) == cls and r < 0.12])
        if len(P) == 0:
            P = np.array([p for _, _, p, r in fits if r < 0.12]) if fits else np.zeros((1, NP))
        classes[cls] = {"mean": np.median(P, 0).tolist(),
                        "cov": (np.cov(P.T) if len(P) > 1 else np.zeros((NP, NP))).tolist(),
                        "count": int(len(P))}
    model = {"xh": XH, "chars": chars, "classes": classes,
             "fit_quality": float(np.median([f[3] for f in fits])) if fits else None}
    (bank / "morph.json").write_text(json.dumps(model), encoding="utf-8")
    return model


class MorphModel:
    """Samples a deformation (and the font form) for a character."""

    def __init__(self, bank_dir: str):
        path = Path(bank_dir) / "morph.json"
        self.m = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def __bool__(self) -> bool:
        return self.m is not None

    def width(self, ch: str) -> Optional[float]:
        e = self.m["chars"].get(ch) if self.m else None
        return e.get("width") if e else None

    def sample(self, ch: str, fonts: List[str], variation: float = 1.0,
               rng: Optional[random.Random] = None) -> Tuple[Optional[str], np.ndarray]:
        """(font path for this form or None = caller's choice, deformation params)."""
        rng = rng or random
        entry = self.m["chars"].get(ch)
        cls = self.m["classes"][char_class(ch)]
        if entry:
            forms = [f for f in entry["forms"] if f["font"] in fonts and f["resid"] < 0.15]
            if forms:
                f = rng.choices(forms, weights=[x["count"] for x in forms])[0]
                n = f["count"]
                # few samples: lean on the class tendency for the spread
                w = n / (n + 3.0)
                mean = np.array(f["mean"])
                cov = w * np.array(f["cov"]) + (1 - w) * np.array(cls["cov"])
                return f["font"], self._draw(mean, cov, variation)
        return None, self._draw(np.array(cls["mean"]), np.array(cls["cov"]), variation)

    @staticmethod
    def _draw(mean: np.ndarray, cov: np.ndarray, variation: float) -> np.ndarray:
        cov = (cov + cov.T) / 2 * (0.6 * variation) ** 2
        try:
            return np.random.multivariate_normal(mean, cov + 1e-6 * np.eye(len(mean)))
        except np.linalg.LinAlgError:
            return mean


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit font->handwriting morphs for a glyph bank.")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--fonts", nargs="*", help="candidate fonts (default: all in fonts/)")
    args = ap.parse_args()
    bank = Path(args.bank)
    fonts = args.fonts or sorted(str(p) for p in Path(__file__).with_name("fonts").glob("*.[to]tf"))
    model = build_model(bank, fonts)
    print(f"{len(model['chars'])} characters modelled, median fit {model['fit_quality']:.3f} x-heights")
    for ch, e in model["chars"].items():
        print(f"  {ch}: " + ", ".join(f"{Path(f['font']).stem} x{f['count']} (fit {f['resid']:.3f})" for f in e["forms"]))


if __name__ == "__main__":
    main()
