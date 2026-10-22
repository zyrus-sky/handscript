"""
Font letters varied at the outline level: every time a letter is written, its
contours (the curves the font defines, like an SVG path) are flattened to points,
deformed by a smooth random field plus a small affine change (slant, width,
height), and filled again. No two instances are identical, edges stay clean, and
the shape stays a coherent letter because neighbouring points move together.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

Contour = List[Tuple[float, float]]


class _FlattenPen(BasePen):
    """Collects contours as point lists, flattening curves into short segments."""

    def __init__(self, glyphset, steps: int = 8):
        super().__init__(glyphset)
        self.contours: List[Contour] = []
        self.cur: Contour = []
        self.steps = steps

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps
            mt = 1 - t
            self.cur.append((mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
                             mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps
            mt = 1 - t
            self.cur.append((mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0],
                             mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]))

    def _closePath(self):
        if len(self.cur) > 2:
            self.contours.append(self.cur)
        self.cur = []

    _endPath = _closePath


@lru_cache(maxsize=16)
def _font(path: str):
    tt = TTFont(path, lazy=True)
    gs = tt.getGlyphSet()
    cmap = tt.getBestCmap() or {}
    upm = tt["head"].unitsPerEm
    def ymax(ch):
        name = cmap.get(ord(ch))
        if not name:
            return None
        pen = _FlattenPen(gs)
        gs[name].draw(pen)
        ys = [p[1] for c in pen.contours for p in c]
        return max(ys) if ys else None
    xh = ymax("x") or 0.5 * upm
    return tt, gs, cmap, upm, xh


@lru_cache(maxsize=512)
def _outline(path: str, ch: str) -> Optional[Tuple[Tuple[Tuple[float, float], ...], ...]]:
    tt, gs, cmap, upm, xh = _font(path)
    name = cmap.get(ord(ch))
    if not name:
        return None
    pen = _FlattenPen(gs)
    gs[name].draw(pen)
    if not pen.contours:
        return None
    return tuple(tuple(c) for c in pen.contours)


def supports(path: str, ch: str) -> bool:
    try:
        return _outline(path, ch) is not None
    except Exception:
        return False


def render(path: str, ch: str, x_height_px: float, body_target: Optional[float] = None,
           variation: float = 1.0, slant_t: float = 0.0, rng: Optional[random.Random] = None,
           morph=None, width_xh: Optional[float] = None) -> Optional[Tuple[Image.Image, float]]:
    """Draw one letter from its outline with a fresh deformation.
    x_height_px: size of the font's x-height in output px.
    body_target: wanted height above the baseline in x-heights (writer's proportion),
    e.g. 1.80 for capitals / tall letters, 1.06 for small ones; None keeps the font's.
    slant_t: extra shear (tan of angle) applied around the baseline.
    morph: learned deformation (morph.py, 12 params in x-heights) moving the font's
    form towards the writer's; it already includes the writer's proportions.
    Returns (L mask, baseline y in the mask) or None if the font lacks the letter."""
    rng = rng or random
    outline = _outline(path, ch)
    if outline is None:
        return None
    tt, gs, cmap, upm, fxh = _font(path)
    k = x_height_px / fxh                                   # font units -> px
    pts = [np.array(c, np.float64) for c in outline]
    allp = np.concatenate(pts)
    ymax = allp[:, 1].max()
    use_morph = morph is not None and morph is not False
    if use_morph:
        import morph as _morph
        xmin = allp[:, 0].min()
        mp = np.asarray(morph, np.float64)
        moved = []
        for c in pts:
            u, vv = (c[:, 0] - xmin) / fxh, -c[:, 1] / fxh
            dx, dy = _morph.field(mp, u, vv)
            moved.append(np.stack([c[:, 0] + dx * fxh, c[:, 1] - dy * fxh], 1))
        pts = moved
        if width_xh:
            # pin the overall width to the writer's measured width for this letter
            allm = np.concatenate(pts)
            w_now = np.ptp(allm[:, 0]) / fxh
            if w_now > 0:
                kx = float(np.clip(width_xh / w_now, 0.6, 1.6))
                x0m = allm[:, 0].min()
                pts = [np.stack([x0m + (c[:, 0] - x0m) * kx, c[:, 1]], 1) for c in pts]
    elif body_target and ymax > 0:
        k_body = body_target * fxh / ymax
        k *= float(np.clip(k_body, 0.8, 1.5))
    v = variation
    # small affine: width / height / slant per instance
    sx = k * rng.uniform(1 - 0.06 * v, 1 + 0.06 * v)
    sy = k * rng.uniform(1 - 0.05 * v, 1 + 0.05 * v)
    shear = slant_t + rng.gauss(0, 0.04 * v)
    # smooth deformation field over the glyph (wavelength ~ x-height): neighbours
    # move together, so curves bend instead of breaking
    amp = (0.02 if use_morph else 0.045) * fxh * v      # learned morph already varies the form
    lam = fxh * rng.uniform(0.8, 1.4)
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(4)]
    ax, ay = rng.uniform(0.5, 1.0) * amp, rng.uniform(0.5, 1.0) * amp
    out = []
    for c in pts:
        x, y = c[:, 0], c[:, 1]
        dx = ax * np.sin(2 * math.pi * y / lam + ph[0]) + 0.5 * ax * np.sin(2 * math.pi * x / lam + ph[1])
        dy = ay * np.sin(2 * math.pi * x / lam + ph[2]) + 0.5 * ay * np.sin(2 * math.pi * y / lam + ph[3])
        X = (x + dx) * sx
        Y = (y + dy) * sy
        X = X + Y * shear                                    # y is up: positive shear leans right
        out.append(np.stack([X, Y], 1))
    allq = np.concatenate(out)
    pad = 4
    x0, x1 = allq[:, 0].min() - pad, allq[:, 0].max() + pad
    y0, y1 = allq[:, 1].min() - pad, allq[:, 1].max() + pad
    S = 2                                                    # supersampling (strokes are redrawn after)
    W, H = int((x1 - x0) * S) + 2, int((y1 - y0) * S) + 2
    # Fill by contour direction (non-zero rule): contours turning like the largest one
    # are ink and merge where strokes overlap; opposite ones are counters (inside of o).
    def area(c):
        x, y = c[:, 0], c[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    areas = [area(c) for c in out]
    outer = np.sign(max(areas, key=abs)) or 1.0
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for c, a in sorted(zip(out, areas), key=lambda t: np.sign(t[1]) != outer):
        xy = np.column_stack([(c[:, 0] - x0) * S, (y1 - c[:, 1]) * S]).ravel().tolist()
        d.polygon(xy, fill=255 if np.sign(a) == outer else 0)
    img = m.resize((max(1, W // S), max(1, H // S)), Image.LANCZOS)
    baseline = y1                                            # font y=0 maps to (y1 - 0) px from the top
    return img, float(baseline)
