"""
Glyph bank: real letter shapes cut out of scanned handwriting.

  python glyphbank.py extract <scan.pdf|images...> --bank banks/mine
      Segments every page into lines and words, removes the writer's slant,
      finds each word's baseline / x-height and suggests cut points.
      Writes banks/mine/words/*.png and banks/mine/words.json.

  Label the words at http://localhost:5000/label (app.py); every save
  rebuilds banks/mine/bank.json + glyphs/*.png, which the engine reads.

  python glyphbank.py build --bank banks/mine    (rebuild bank.json by hand)

A label may cover several letters ("th", "the", "ss"): joined forms are kept
as one piece and preferred by the engine over single letters.
"""

from __future__ import annotations

import argparse
import json
import sys
import random
import re
import threading
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ── segmentation tuning (scan ~250 DPI, dark ink, light show-through) ─────
INK_THRESHOLD = 120        # gray below this = ink
ALPHA_PAPER = 215          # gray at/above this → fully transparent
ALPHA_INK = 60             # gray at/below this → fully opaque
MIN_COMPONENT_AREA = 14    # px; smaller blobs are scan noise
SLANT_RANGE = range(-16, 42, 2)   # degrees tried when estimating slant


# ═══════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════

def _ink_gray(bgr: np.ndarray) -> np.ndarray:
    """Colour photo → grayscale with only the pen ink kept (paper, ruled lines and
    printed headers become white). Blue/coloured ink is separated by hue; black ink
    on a photo falls back to background-normalised darkness."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.medianBlur(gray.astype(np.uint8), 41).astype(np.float32)
    dark = np.clip(bg - gray, 0, 255)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    b = bgr.astype(np.int16)
    blue = ((b[..., 0] - b[..., 2]) > 14) & (hsv[..., 0] >= 100) & (hsv[..., 0] <= 140) & (hsv[..., 1] > 45)
    ink = blue & (dark > 25)
    if ink.mean() < 0.003:                       # not blue ink: keep dark, drop ruled lines
        ink = dark > 45
        rules = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 61), np.uint8))
        ink = ink & (rules == 0)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0
    strength = np.clip(dark / max(1.0, float(np.percentile(dark[ink], 90)) if ink.any() else 1.0), 0, 1)
    return np.where(ink, 255 - 235 * np.maximum(strength, 0.6), 255).astype(np.uint8)


def _is_color_photo(img: np.ndarray) -> bool:
    if img.ndim != 3:
        return False
    b = img.astype(np.int16)
    return float(np.abs(b[..., 0] - b[..., 2]).mean()) > 4.0


def _load_pages(sources: List[str]) -> List[np.ndarray]:
    """Grayscale page images from PDFs (embedded scan at native res) or image files.
    Colour photos (e.g. notebook pages) are reduced to the ink only."""
    pages = []
    for src in sources:
        if src.lower().endswith(".pdf"):
            import fitz  # PyMuPDF
            doc = fitz.open(src)
            for page in doc:
                imgs = page.get_images()
                if len(imgs) == 1:
                    raw = doc.extract_image(imgs[0][0])["image"]
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                else:
                    pix = page.get_pixmap(dpi=250)
                    img = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3],
                                       cv2.COLOR_RGB2BGR)
                pages.append(_to_page_gray(img))
        else:
            pages.append(_to_page_gray(cv2.imread(src, cv2.IMREAD_UNCHANGED)))
    return pages


def _to_page_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if _is_color_photo(img):
        return _ink_gray(img)
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) runs of True values."""
    d = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _shear(img: np.ndarray, t: float, border: int) -> np.ndarray:
    """x' = x + y*t (shifted so nothing goes negative). t = tan(slant) removes a rightward slant."""
    h, w = img.shape
    off = -min(0.0, h * t)
    m = np.float32([[1, t, off], [0, 1, 0]])
    return cv2.warpAffine(img, m, (int(w + abs(h * t)) + 1, h),
                          flags=cv2.INTER_LINEAR, borderValue=border)


def _estimate_slant(mask: np.ndarray) -> float:
    """Angle (deg, + = leaning right) whose removal makes strokes most vertical."""
    small = cv2.resize(mask, (max(1, mask.shape[1] // 2), max(1, mask.shape[0] // 2)),
                       interpolation=cv2.INTER_AREA)
    best, best_score = 0.0, -1.0
    for deg in SLANT_RANGE:
        sheared = _shear(small, float(np.tan(np.radians(deg))), 0)
        score = float(np.square(sheared.sum(0).astype(np.float64)).sum())
        if score > best_score:
            best, best_score = float(deg), score
    return best


def _x_band(mask: np.ndarray) -> Tuple[int, int]:
    """(top, bottom) of the dense middle band = x-height zone; bottom is the baseline."""
    prof = np.convolve(mask.sum(1).astype(float), np.ones(3) / 3, "same")
    if prof.max() <= 0:
        return 0, mask.shape[0]
    peak = int(prof.argmax())
    thr = 0.4 * prof[peak]
    top, bot = peak, peak
    while top > 0 and prof[top - 1] >= thr:
        top -= 1
    while bot < len(prof) - 1 and prof[bot + 1] >= thr:
        bot += 1
    return top, bot + 1


def _auto_cuts(mask: np.ndarray) -> List[int]:
    """Cut suggestions at blank columns between separate strokes."""
    cols = mask.sum(0) == 0
    return [int((a + b) // 2) for a, b in _runs(cols) if a > 0 and b < len(cols)]


def _line_groups(gray: np.ndarray):
    """Ink components grouped into text lines.
    Returns (labels, stats, centroids, {line_index: [component ids]}) — lines top to bottom."""
    ink = (gray < INK_THRESHOLD).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA]
    if not keep:
        return labels, stats, cents, {}

    # Text lines: chain stroke components left→right, each joining the line whose
    # running centre-y is closest (follows sloping / drifting lines, which a
    # whole-page projection merges together).
    heights = stats[keep, cv2.CC_STAT_HEIGHT]
    main = [c for c in keep if stats[c, cv2.CC_STAT_HEIGHT] >= max(10, 0.4 * np.median(heights))]
    med_h = float(np.median(stats[main, cv2.CC_STAT_HEIGHT])) if main else 20.0
    tol = 0.6 * med_h
    lines: List[dict] = []
    for c in sorted(main, key=lambda c: stats[c, cv2.CC_STAT_LEFT]):
        left = stats[c, cv2.CC_STAT_LEFT]
        top, h = stats[c, cv2.CC_STAT_TOP], stats[c, cv2.CC_STAT_HEIGHT]
        cy = cents[c][1]
        # Tall strokes: their x-height core sits at the top (descender) or bottom (ascender).
        probes = (cy, top + med_h / 2, top + h - med_h / 2)
        best, best_d = None, tol
        for ln in lines:
            d = min(abs(ln["cy"] - p) for p in probes)
            if d < best_d and left - ln["right"] < 10 * med_h:
                best, best_d = ln, d
        if best is None:
            best = {"cy": cy, "right": 0, "comps": []}
            lines.append(best)
        best["comps"].append(c)
        if h < 1.5 * med_h:
            best["cy"] = 0.7 * best["cy"] + 0.3 * cy
        best["right"] = max(best["right"], left + stats[c, cv2.CC_STAT_WIDTH])
    if not lines:
        return labels, stats, cents, {}

    # Dots, bars and other small marks join the line with the nearest component.
    centres = np.array([cents[c] for c in main])
    owner = {c: li for li, ln in enumerate(lines) for c in ln["comps"]}
    line_comps: Dict[int, List[int]] = {i: list(ln["comps"]) for i, ln in enumerate(lines)}
    for c in keep:
        if c in owner:
            continue
        d = np.hypot(*(centres - cents[c]).T)
        j = int(d.argmin())
        if d[j] < 1.5 * med_h:
            line_comps[owner[main[j]]].append(c)
    order = sorted(line_comps, key=lambda i: min(cents[c][1] for c in line_comps[i]))
    line_comps = {k: line_comps[i] for k, i in enumerate(order)}
    return labels, stats, cents, line_comps


def _segment_page(gray: np.ndarray) -> List[dict]:
    """Return word records (deslanted gray + alpha crops plus metrics) for one page."""
    labels, stats, cents, line_comps = _line_groups(gray)
    words = []
    for li, comps in line_comps.items():
        if not comps:
            continue
        x0 = min(stats[c, cv2.CC_STAT_LEFT] for c in comps)
        y0 = min(stats[c, cv2.CC_STAT_TOP] for c in comps)
        x1 = max(stats[c, cv2.CC_STAT_LEFT] + stats[c, cv2.CC_STAT_WIDTH] for c in comps)
        y1 = max(stats[c, cv2.CC_STAT_TOP] + stats[c, cv2.CC_STAT_HEIGHT] for c in comps)
        own = np.isin(labels[y0:y1, x0:x1], comps).astype(np.uint8)
        own_soft = cv2.dilate(own, np.ones((3, 3), np.uint8))
        g = np.where(own_soft > 0, gray[y0:y1, x0:x1], 255).astype(np.uint8)

        slant = _estimate_slant(own)
        t = float(np.tan(np.radians(slant)))
        g = _shear(g, t, 255)
        m = (g < INK_THRESHOLD).astype(np.uint8)
        _, line_base = _x_band(m)
        xh_line = max(8, line_base - _x_band(m)[0])

        # Words: column gaps wider than ~0.55 x-height.
        gap = max(10, int(0.55 * xh_line))
        colink = m.sum(0) > 0
        spans: List[List[int]] = []
        for a, b in _runs(colink):
            if spans and a - spans[-1][1] < gap:
                spans[-1][1] = b
            else:
                spans.append([a, b])

        for a, b in spans:
            wm = m[:, a:b]
            rows = np.where(wm.sum(1) > 0)[0]
            if len(rows) == 0 or max(b - a, rows[-1] - rows[0]) < 0.6 * xh_line or wm.sum() < 60:
                continue  # specks and stray punctuation
            top, bot = int(rows[0]), int(rows[-1]) + 1
            wg = g[top:bot, a:b].copy()
            wm = wm[top:bot].copy()
            xt, xb = _x_band(wm)

            # Tiny marks below the baseline are i-dots / specks from the next line.
            xh = max(4, xb - xt)
            nc, cl, cs, _ = cv2.connectedComponentsWithStats(wm, connectivity=8)
            for c in range(1, nc):
                if cs[c, cv2.CC_STAT_AREA] < 0.15 * xh * xh and cs[c, cv2.CC_STAT_TOP] > xb + 0.15 * xh:
                    wm[cl == c] = 0
                    wg[cv2.dilate((cl == c).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 255
            rows = np.where(wm.sum(1) > 0)[0]
            if len(rows) == 0:
                continue
            wg, wm = wg[:rows[-1] + 1], wm[:rows[-1] + 1]
            words.append({
                "line": li,
                "slant": slant,
                "gray": wg,
                "baseline": xb,
                "xh": max(4, xb - xt),
                "cuts": _auto_cuts(wm),
            })

    # A single word's x-band is noisy (open 'a', tall loops): pull outliers to the
    # line median (page median for short lines). The engine scales by this.
    if words:
        page_xh = float(np.median([w["xh"] for w in words]))
        for li in {w["line"] for w in words}:
            lw = [w for w in words if w["line"] == li]
            ref = float(np.median([w["xh"] for w in lw])) if len(lw) >= 3 else page_xh
            if not 0.6 * page_xh <= ref <= 1.6 * page_xh:
                ref = page_xh
            for w in lw:
                if not 0.75 * ref <= w["xh"] <= 1.3 * ref:
                    w["xh"] = int(round(ref))
    return words


def _page_layout(gray: np.ndarray) -> Optional[dict]:
    """Writer's page geometry, all in px of this page (caller normalises by width)."""
    labels, stats, cents, line_comps = _line_groups(gray)
    if len(line_comps) < 3:
        return None
    H, W = gray.shape
    lines = []
    for comps in line_comps.values():
        hs = np.array([stats[c, cv2.CC_STAT_HEIGHT] for c in comps])
        small = [c for c, h in zip(comps, hs) if h >= 6 and h <= 1.5 * np.median(hs)]
        if len(small) < 3:
            continue
        xs = np.array([cents[c][0] for c in small])
        bottoms = np.array([stats[c, cv2.CC_STAT_TOP] + stats[c, cv2.CC_STAT_HEIGHT] for c in small], float)
        slope, icpt = np.polyfit(xs, bottoms, 1) if np.ptp(xs) > 50 else (0.0, float(np.median(bottoms)))
        x0 = min(stats[c, cv2.CC_STAT_LEFT] for c in comps)
        x1 = max(stats[c, cv2.CC_STAT_LEFT] + stats[c, cv2.CC_STAT_WIDTH] for c in comps)
        top = min(stats[c, cv2.CC_STAT_TOP] for c in comps)
        # Horizontal gaps between successive ink components (slant overlap ignored).
        gaps, reach = [], None
        for c in sorted(comps, key=lambda c: stats[c, cv2.CC_STAT_LEFT]):
            l, w = stats[c, cv2.CC_STAT_LEFT], stats[c, cv2.CC_STAT_WIDTH]
            if stats[c, cv2.CC_STAT_AREA] < 40:
                continue
            if reach is not None and l > reach:
                gaps.append(float(l - reach))
            reach = l + w if reach is None else max(reach, l + w)
        lines.append({"x0": x0, "x1": x1, "top": top, "slope": float(slope),
                      "base_mid": float(icpt + slope * W / 2), "gaps": gaps})
    if len(lines) < 3:
        return None
    lines.sort(key=lambda l: l["base_mid"])
    pitches = np.diff([l["base_mid"] for l in lines])
    pitches = pitches[pitches > 15]

    # Stroke width ≈ 2·area / perimeter for thin strokes.
    ink = (gray < INK_THRESHOLD).astype(np.uint8)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    stroke = 2.0 * float(ink.sum()) / max(1.0, perim)

    return {
        "width": W, "height": H,
        "pitches": pitches.tolist(),
        "slopes": [l["slope"] for l in lines],
        "left": float(np.percentile([l["x0"] for l in lines], 15)),
        "right": float(W - np.percentile([l["x1"] for l in lines], 85)),
        "top": float(lines[0]["top"]),
        "stroke": stroke,
        "gaps": [g for l in lines for g in l["gaps"]],
    }


def measure_layout(sources: List[str], bank: Path) -> dict:
    """Measure the writer's layout over all pages; ratios are relative to page width
    (sizes) or x-height (gaps, stroke). Written to bank/layout.json for the engine."""
    words = load_words(bank)
    pages = [_page_layout(g) for g in _load_pages(sources)]
    ok = [(i + 1, p) for i, p in enumerate(pages) if p]
    if not ok:
        raise ValueError("Could not measure the page layout.")

    def per_page(i, p, key):
        return [v / p["width"] for v in p[key]]

    xh_ratio = [w["xh"] / next(p["width"] for j, p in ok if j == w["page"])
                for w in words if any(j == w["page"] for j, _ in ok)]
    xh = float(np.median(xh_ratio))
    pitch = [v for i, p in ok for v in per_page(i, p, "pitches")]
    line_pitch = float(np.median(pitch))
    para = [v for v in pitch if v > 1.45 * line_pitch]
    regular = [v for v in pitch if v <= 1.45 * line_pitch]
    slopes = [v for _, p in ok for v in p["slopes"] if abs(v) < 0.08]

    # Gaps between ink runs on a line, in x-heights; word gaps are the wide ones.
    gaps = [v / (xh * p["width"]) for _, p in ok for v in p["gaps"]]
    word_gaps = [g for g in gaps if 0.45 <= g <= 4.0]
    layout = {
        "x_height": xh,                                        # / page width
        "line_pitch": line_pitch,                              # / page width
        "line_pitch_jitter": float(np.std(regular)) if regular else 0.0,
        "paragraph_extra": float(np.median(para) - line_pitch) if para else 0.5 * line_pitch,
        "margin_left": float(np.median([p["left"] / p["width"] for _, p in ok])),
        "margin_right": float(np.median([p["right"] / p["width"] for _, p in ok])),
        "margin_top": float(np.median([p["top"] / p["width"] for _, p in ok])),
        "slope_mean": float(np.mean(slopes)) if slopes else 0.0,
        "slope_std": float(np.std(slopes)) if slopes else 0.0,
        "stroke": float(np.median([p["stroke"] / p["width"] for _, p in ok])) / xh,   # / x-height
        "word_gap": [float(np.percentile(word_gaps, 25)), float(np.percentile(word_gaps, 75))]
        if word_gaps else [0.8, 1.4],                          # / x-height
    }
    lg = letter_gaps(bank)
    if lg:
        layout.update(lg)
    (bank / "layout.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
    return layout


def letter_gaps(bank: Path, source: str = "scan") -> Optional[dict]:
    """The writer's spacing between neighbouring letters inside words, measured on
    the ink itself: for consecutive ink shapes in a word (pen lifts), the horizontal
    distance from the right edge of one to the left edge of the next, row by row
    (closest approach), in x-heights. Joined letters (no pen lift) count as touching."""
    words = [w for w in load_words(bank) if w.get("source", "scan") == source]
    refs = page_refs(load_words(bank))
    gaps, touching, pairs = [], 0, 0
    for w in words:
        a = np.array(Image.open(bank / "words" / f"{w['id']}.png"))[:, :, 3]
        m = (a > 100).astype(np.uint8)
        ref = refs[_page_key(w)]
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.08 * ref * ref]
        comps.sort(key=lambda i: st[i, cv2.CC_STAT_LEFT])
        for i, j in zip(comps, comps[1:]):
            pairs += 1
            mi, mj = lab == i, lab == j
            rows = np.where(mi.any(1) & mj.any(1))[0]
            if len(rows) == 0:
                continue
            d = min(int(np.where(mj[r])[0].min()) - int(np.where(mi[r])[0].max()) for r in rows)
            if d <= 0:
                touching += 1
                continue
            if d < 1.5 * ref:                      # wider is a word gap missed by the segmenter
                gaps.append(d / ref)
        # joined letters inside one shape are 'touching' too: estimate letters per shape
        letters = len("".join(w["labels"])) if w.get("labels") else None
        if letters and len(comps) and letters > len(comps):
            touching += letters - len(comps)
            pairs += letters - len(comps)
    if len(gaps) < 20:
        return None
    return {"letter_gap": [float(np.percentile(gaps, q)) for q in (20, 50, 80)],
            "touch_fraction": touching / max(1, pairs)}


def load_layout(bank: Path) -> Optional[dict]:
    path = Path(bank) / "layout.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _alpha(gray: np.ndarray) -> np.ndarray:
    a = (ALPHA_PAPER - gray.astype(np.float32)) / (ALPHA_PAPER - ALPHA_INK)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def extract(sources: List[str], bank: Path, append: bool = False) -> int:
    """Segment pages into word images. append=True adds the pages after the bank's
    existing ones (labels kept); new pages are numbered on from the last one."""
    (bank / "words").mkdir(parents=True, exist_ok=True)
    old = load_words(bank) if append and (bank / "words.json").exists() else []
    first = 1 + max((w["page"] for w in old), default=0)
    source = Path(sources[0]).stem if append else "scan"
    records = []
    for pi, page in enumerate(_load_pages(sources), start=first - 1):
        for wi, w in enumerate(_segment_page(page)):
            wid = f"p{pi + 1:02d}_l{w['line']:02d}_w{wi:03d}"
            a = _alpha(w["gray"])
            rgba = np.dstack([np.zeros_like(a)] * 3 + [a])
            Image.fromarray(rgba, "RGBA").save(bank / "words" / f"{wid}.png")
            records.append({
                "id": wid, "page": pi + 1, "line": w["line"],
                "w": int(a.shape[1]), "h": int(a.shape[0]),
                "baseline": int(w["baseline"]), "xh": int(w["xh"]),
                "slant": w["slant"], "cuts": w["cuts"],
                "labels": None, "status": "todo", "source": source,
            })
    (bank / "words.json").write_text(json.dumps(old + records, indent=1), encoding="utf-8")
    return len(records)


def disabled_sources(bank: Path) -> List[str]:
    """Sources (scan documents / 'pad') kept in the bank but left out of bank.json."""
    p = Path(bank) / "sources.json"
    return json.loads(p.read_text(encoding="utf-8")).get("disabled", []) if p.exists() else []


def prefer_pad(bank: Path) -> bool:
    p = Path(bank) / "sources.json"
    return bool(json.loads(p.read_text(encoding="utf-8")).get("prefer_pad")) if p.exists() else False


def set_prefer_pad(bank: Path, on: bool) -> None:
    p = Path(bank) / "sources.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"disabled": []}
    d["prefer_pad"] = bool(on)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    build(Path(bank))


def set_source(bank: Path, source: str, enabled: bool) -> List[str]:
    off = [s for s in disabled_sources(bank) if s != source] + ([] if enabled else [source])
    p = Path(bank) / "sources.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d["disabled"] = off
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    build(Path(bank))
    return off


def page_refs(words: List[dict]) -> Dict[str, float]:
    """Median x-height per source page. Photos come at different resolutions, so a
    piece is scaled by its own page's median (stable) rather than its word's (noisy)."""
    groups: Dict[str, List[int]] = {}
    for w in words:
        groups.setdefault(_page_key(w), []).append(w["xh"])
    return {k: float(np.median(v)) for k, v in groups.items()}


def _page_key(w: dict) -> str:
    return "pad" if w.get("source") == "pad" else f"p{w['page']}"


# ═══════════════════════════════════════════════════════════════════════
# Building the bank from labeled words
# ═══════════════════════════════════════════════════════════════════════

def load_words(bank: Path) -> List[dict]:
    return json.loads((bank / "words.json").read_text(encoding="utf-8"))


def save_words(bank: Path, words: List[dict]) -> None:
    (bank / "words.json").write_text(json.dumps(words, indent=1), encoding="utf-8")


_BUILD_LOCK = threading.RLock()       # one writer at a time: saves and rebuilds don't collide
_build_timers: Dict[str, threading.Timer] = {}


def schedule_build(bank: Path, delay: float = 1.5) -> None:
    """Rebuild in the background once saving pauses (the pad answers instantly)."""
    key = str(bank)
    old = _build_timers.get(key)
    if old:
        old.cancel()
    t = threading.Timer(delay, build, args=(bank,))
    t.daemon = True
    _build_timers[key] = t
    t.start()


def build(bank: Path) -> dict:
    with _BUILD_LOCK:
        return _build(bank)


def _build(bank: Path) -> dict:
    """Cut labeled words into glyph PNGs (alpha masks) and write bank.json."""
    gdir = bank / "glyphs"
    gdir.mkdir(exist_ok=True)
    # overwrite in place and drop leftovers at the end: a generation running while the
    # bank is rebuilt (e.g. during labeling) never finds its piece images missing
    before = {p.name for p in gdir.glob("*.png")}

    glyphs, slants = [], []
    all_words = load_words(bank)
    refs = page_refs(all_words)
    off = set(disabled_sources(bank))
    for w in all_words:
        if w["status"] != "done" or not w["labels"] or w.get("source", "scan") in off:
            continue
        alpha = np.array(Image.open(bank / "words" / f"{w['id']}.png"))[:, :, 3]
        edges = [0] + sorted(w["cuts"]) + [w["w"]]
        slants.append(w["slant"])
        for k, label in enumerate(w["labels"]):
            if label == "_" or k + 1 >= len(edges):
                continue
            piece = alpha[:, edges[k]:edges[k + 1]]
            rows = np.where(piece.max(1) > 40)[0]
            cols = np.where(piece.max(0) > 40)[0]
            if len(rows) == 0:
                continue
            top, bot = int(rows[0]), int(rows[-1]) + 1
            piece = piece[top:bot]
            ref = refs[_page_key(w)]
            if w.get("source") != "pad":                # pad strokes have no cut stubs
                piece = _trim_tail(piece, label, ref)
            fname = f"{w['id']}_{k}.png"
            Image.fromarray(piece, "L").save(gdir / fname)
            entry = {
                "label": label, "file": fname,
                "baseline": w["baseline"] - top, "xh": w["xh"], "ref_xh": ref,
                # ink extent within the piece (cuts can leave blank margins)
                "ink_left": int(cols[0]), "ink_right": int(cols[-1]) + 1,
                "slant": w["slant"], "cut": len(w["labels"]) > 1,
                "source": w.get("source", "scan"),
            }
            # pad strokes are clean, so their measured size is trustworthy: allow more
            # correction (people write smaller than the pad's guide lines)
            clamp = (0.45, 2.4) if w.get("source") == "pad" else (0.55, 1.5)
            entry["base0"] = entry["baseline"]          # where the pad's guide line was
            entry.update(_piece_metrics(piece, entry["baseline"], ref, label, clamp))
            if w.get("source") == "pad" and len(w["labels"]) == 1:
                entry["strokes"] = f"{w['id']}.json"    # redrawn with variation at render time
            glyphs.append(entry)

    for stale in before - {g["file"] for g in glyphs}:
        try:
            (gdir / stale).unlink()
        except OSError:
            pass
    # quality check: pieces that would look wrong are left out (reversible)
    # new pieces get read too when the handwriting reader is already loaded (the server
    # warms it at start); readings are cached, so only new pieces cost anything
    pr = sys.modules.get("proofread")
    report = audit(bank, glyphs, use_ocr=bool(pr is not None and getattr(pr, "_reader", None) is not None))
    excluded = _excluded(glyphs, report)
    for g in glyphs:                         # how typical of its label (planner prefers typical)
        g["typ"] = report.get("typ", {}).get(g["file"], 1.0)
    if prefer_pad(bank):
        # scanned copies step aside wherever the writer has 2+ pad copies of that label
        pad_n: Dict[str, int] = {}
        for g in glyphs:
            if g.get("source") == "pad" and g["file"] not in excluded:
                pad_n[g["label"]] = pad_n.get(g["label"], 0) + 1
        keep = set(report.get("keep", []))
        excluded |= {g["file"] for g in glyphs if g.get("source") != "pad"
                     and pad_n.get(g["label"], 0) >= 2 and g["file"] not in keep}
    meta = {
        "slant_deg": statistics.median(slants) if slants else 0.0,
        "pad_stretch": pad_stretch(all_words, glyphs, bank / "strokes"),
        "glyphs": [g for g in glyphs if g["file"] not in excluded],
        "excluded": len(excluded),
        "excluded_glyphs": [g for g in glyphs if g["file"] in excluded],
    }
    (bank / "bank.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# ═══════════════════════════════════════════════════════════════════════
# Quality check ("doctor")
# ═══════════════════════════════════════════════════════════════════════

def shape_sig(a: np.ndarray, size: int = 40) -> np.ndarray:
    """Normalised ink shape (aspect kept, fitted in size x size, slightly thickened)
    for comparing letter shapes regardless of scale and pen."""
    ys, xs = np.where(a > 100)
    out = np.zeros((size, size), np.float32)
    if len(ys) == 0:
        return out
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    k = (size - 4) / max(h, w)
    a = cv2.resize(a, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    oy, ox = (size - a.shape[0]) // 2, (size - a.shape[1]) // 2
    out[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a > 90
    return cv2.dilate(out, np.ones((3, 3), np.uint8))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / max(1.0, ((a + b) > 0).sum()))


def audit(bank: Path, glyphs: Optional[List[dict]] = None, use_ocr: bool = False) -> dict:
    """Flag pieces that would look wrong when written. Returns and saves
    audit.json: {"flags": {file: [reasons]}, "keep": [files the user restored],
    "ocr": {file: text}}. Checks:
      - tail-less letters whose ink hangs well under the baseline (cut leftovers)
      - size still off after the height correction hit its limit
      - implausible width for the number of letters
      - broken into many fragments (stubs from neighbouring letters)
      - unlike every other copy of the same label (mislabeled / bad cut)
      - optional: the local OCR model cannot read the label in it."""
    path = bank / "audit.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    keep = prev.get("keep", [])
    ocr_cache: Dict[str, str] = prev.get("ocr", {})
    if glyphs is None:
        glyphs = json.loads((bank / "bank.json").read_text(encoding="utf-8"))["glyphs"]
    flags: Dict[str, List[str]] = {}
    sigs: Dict[str, np.ndarray] = {}
    for g in glyphs:
        why = []
        a = np.array(Image.open(bank / "glyphs" / g["file"]))
        sigs[g["file"]] = shape_sig(a)
        letters = [c for c in g["label"] if c.isalnum()]
        ref = g.get("ref_xh") or 30.0
        if letters and not g.get("desc", True) and g.get("below", 0) > 0.35:
            why.append(f"hangs {g['below']:.1f} x-heights under the line")
        sf = g.get("size_fix", 1.0)
        lo, hi = (0.46, 2.39) if g.get("source") == "pad" else (0.56, 1.49)
        if sf <= lo or sf >= hi:
            why.append("size still wrong after correction")
        if letters:
            per = (g["ink_right"] - g["ink_left"]) / ref / max(1, len(g["label"]))
            if per < 0.2 or per > 2.4:
                why.append(f"width {per:.2f} x-heights per letter")
        n, _, st, _ = cv2.connectedComponentsWithStats((a > 100).astype(np.uint8), connectivity=8)
        blobs = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.02 * ref * ref)
        if blobs > len(g["label"]) + 2:
            why.append(f"{blobs} separate fragments")
        flags[g["file"]] = why
    # copies of the same label should resemble each other
    by_label: Dict[str, List[dict]] = {}
    for g in glyphs:
        by_label.setdefault(g["label"], []).append(g)
    typ: Dict[str, float] = {}
    for label, gs in by_label.items():
        if len(gs) < 3:
            continue
        sims = []
        for g in gs:
            others = [_iou(sigs[g["file"]], sigs[o["file"]]) for o in gs if o is not g]
            sims.append(float(np.median(others)))
        med = float(np.median(sims))
        mad = float(np.median([abs(x - med) for x in sims])) or 0.02
        for g, sim in zip(gs, sims):
            typ[g["file"]] = round(sim / med, 3) if med > 0 else 1.0
            if sim < med - 3.0 * mad and sim < 0.8 * med:
                flags[g["file"]].append(f"unlike the other {len(gs) - 1} copies of '{label}'")
    if use_ocr or ocr_cache:                 # cached readings keep applying on every rebuild
        todo = [g for g in glyphs if len(g["label"]) >= 2 and not flags[g["file"]]
                and g["file"] not in ocr_cache]
        if use_ocr and todo:
            # the handwriting reader (TrOCR, GPU, batched); the local LLM if it's missing
            imgs = []
            for g in todo:
                a = np.array(Image.open(bank / "glyphs" / g["file"]))
                img = Image.fromarray(255 - a).convert("RGB")
                k = 36.0 / max(8.0, float(g.get("ref_xh") or 30.0) / max(0.3, g.get("size_fix", 1.0)))
                img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
                imgs.append(img)
            try:
                import proofread
                reads = proofread.reader().read_batch(imgs)
            except Exception:
                import io
                import llm
                reads = []
                for img in imgs:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    reads.append(llm.read_word(buf.getvalue()))
            for g, r in zip(todo, reads):
                ocr_cache[g["file"]] = r
        for g in glyphs:
            if len(g["label"]) < 2 or flags[g["file"]] or g["file"] not in ocr_cache:
                continue
            # the reader mixes up look-alike letters and digits (zoo -> 200): not the writer's fault
            same = str.maketrans("0215", "ozis")
            got = re.sub(r"[^a-z0-9]", "", ocr_cache[g["file"]].lower()).translate(same).replace("l", "i")
            want = re.sub(r"[^a-z0-9]", "", g["label"].lower()).translate(same).replace("l", "i")
            if want and _edit_ratio(got, want) > 0.5:
                flags[g["file"]].append(f"OCR reads '{ocr_cache[g['file']]}'")
    report = {"flags": {f: w for f, w in flags.items() if w}, "keep": keep, "ocr": ocr_cache, "typ": typ,
              "removed": prev.get("removed", [])}
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def _severe(reason: str) -> bool:
    if reason.startswith("hangs"):
        return float(reason.split()[1]) > 0.6
    return reason.startswith(("unlike", "OCR")) or "fragments" in reason


def _excluded(glyphs: List[dict], report: dict) -> set:
    """Severe problems are always left out. Mild ones (size at the correction limit,
    a small tail, odd width) only while the label keeps 2+ unflagged copies; a slightly
    off piece of the writer's own still beats a font letter."""
    flags, keep = report["flags"], set(report.get("keep", []))
    out = set(report.get("removed", []))
    clean: Dict[str, int] = {}
    for g in glyphs:
        if not flags.get(g["file"]):
            clean[g["label"]] = clean.get(g["label"], 0) + 1
    for g in glyphs:
        why = flags.get(g["file"])
        if not why or g["file"] in keep:
            continue
        if any(_severe(r) for r in why) or clean.get(g["label"], 0) >= 2:
            out.add(g["file"])
    return out


def _edit_ratio(a: str, b: str) -> float:
    """Levenshtein distance / length of the longer string."""
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def set_removed(bank: Path, file: str, removed: bool) -> None:
    """The writer's own verdict on a piece: removed pieces are never used (reversible)."""
    path = bank / "audit.json"
    rep = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"flags": {}, "keep": []}
    rm = [f for f in rep.get("removed", []) if f != file] + ([file] if removed else [])
    rep["removed"] = rm
    if not removed:                               # restoring also overrides an automatic flag
        rep["keep"] = [k for k in rep.get("keep", []) if k != file] + [file]
    path.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    build(bank)


def pieces_by_label(bank: Path) -> List[dict]:
    """For the bank manager: every piece grouped by label, with its status."""
    meta = json.loads((bank / "bank.json").read_text(encoding="utf-8"))
    path = bank / "audit.json"
    rep = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    removed, flags = set(rep.get("removed", [])), rep.get("flags", {})
    groups: Dict[str, List[dict]] = {}
    for g, used in [(g, True) for g in meta["glyphs"]] + [(g, False) for g in meta.get("excluded_glyphs", [])]:
        groups.setdefault(g["label"], []).append({
            "file": g["file"], "source": g.get("source", "scan"), "used": used,
            "status": "in use" if used else ("removed by you" if g["file"] in removed else
                                             "left out (pad copies preferred or failed the check)"),
            "why": flags.get(g["file"], []),
        })
    return [{"label": k, "pieces": v} for k, v in sorted(groups.items(), key=lambda kv: (len(kv[0]) > 1, kv[0].lower()))]


def sentences_to_write(bank: Path, text: str, limit: int = 12, max_words: int = 8) -> List[str]:
    """Sentences from the text worth writing on the pad: people write words and letters
    more naturally inside sentences than one at a time. Greedily picks the parts that
    cover letters the bank lacks and words that repeat a lot but have few copies.
    Punctuation is left out of the prompts (word splitting stays reliable)."""
    plain = re.sub(r"[#*`>|_\[\]()]", " ", text)
    raw = [s for s in re.split(r"(?<=[.!?:;])\s+|\n+", plain) if s.strip()]
    parts: List[List[str]] = []
    for s in raw:
        ws = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", s)
        for i in range(0, len(ws), max_words):
            chunk = ws[i:i + max_words]
            if len(chunk) >= 2:
                parts.append(chunk)
    if not parts:
        return []
    cov = coverage(bank)
    have = {ch for label in cov for ch in label}
    freq: Dict[str, int] = {}
    for p in parts:
        for w in p:
            freq[w] = freq.get(w, 0) + 1
    need = {w: max(0, min(4, n) - cov.get(w, 0)) for w, n in freq.items() if n >= 2 and len(w) >= 2}
    missing = {ch for p in parts for w in p for ch in w if ch.isalnum() and ch not in have}
    chosen: List[str] = []
    written: set = set()
    pool = list(parts)
    while pool and len(chosen) < limit:
        def score(p):
            letters = {ch for w in p for ch in w} & missing
            # words never written whole get assembled from pieces (less natural)
            unwritten = sum(1 for w in p if len(w) >= 3 and cov.get(w, 0) == 0 and w not in written)
            return 3 * len(letters) + sum(1 for w in p if need.get(w, 0) > 0) + 0.5 * unwritten
        best = max(pool, key=score)
        if score(best) <= 0:
            break
        pool.remove(best)
        chosen.append(" ".join(best))
        written.update(best)
        missing -= {ch for w in best for ch in w}
        for w in best:
            if need.get(w, 0) > 0:
                need[w] -= 1
    return chosen


def split_sentence_strokes(strokes: list, n_words: int, x_height: float) -> Optional[List[list]]:
    """Group pad strokes into words by the horizontal gaps between them (dots and
    crossbars overlap their word, so they join it). None if it can't find n_words."""
    boxes = sorted(((min(p[0] for p in st), max(p[0] for p in st), st) for st in strokes if st),
                   key=lambda b: b[0])
    clusters: List[list] = []
    for x0, x1, st in boxes:
        if clusters and x0 <= clusters[-1][1] + 0.35 * x_height:
            clusters[-1][1] = max(clusters[-1][1], x1)
            clusters[-1][2].append(st)
        else:
            clusters.append([x0, x1, [st]])
    while len(clusters) > n_words:                     # merge across the smallest gaps
        gaps = [clusters[i + 1][0] - clusters[i][1] for i in range(len(clusters) - 1)]
        i = int(np.argmin(gaps))
        a, b = clusters[i], clusters.pop(i + 1)
        a[1] = max(a[1], b[1])
        a[2] += b[2]
    if len(clusters) != n_words:
        return None
    return [c[2] for c in clusters]


def add_pad_sentence(bank: Path, text: str, strokes: list, baseline: float, x_height: float) -> List[dict]:
    """Save a sentence written on the pad as one sample per word (labels from the text);
    returns the new word records. The caller rebuilds (schedule_build)."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    groups = split_sentence_strokes(strokes, len(words), x_height)
    if groups is None:
        raise ValueError(f"Couldn't separate {len(words)} words — leave a clear gap between words and try again")
    saved = []
    with _BUILD_LOCK:
        for w, sts in zip(words, groups):
            x0 = min(p[0] for st in sts for p in st)
            moved = [[[p[0] - x0 + 10, p[1], p[2] if len(p) > 2 else 0.5] for p in st] for st in sts]
            saved.append(_add_pad_sample(bank, w, moved, baseline, x_height, rebuild=False))
    return saved


STOP_WORDS = set("""a an the and or but of to in on at by for with from as is are was were be been
it its this that these those there their they we you he she i not no so if then than can will
would should could may might do does did has have had which who what when where how all any""".split())


def words_to_write(bank: Path, text: str, limit: int = 60) -> List[str]:
    """What the writer should write on the pad for this text, one prompt per entry:
    1. a short word for each letter the bank lacks;
    2. words never written whole (they get assembled from pieces), most frequent first;
    3. more copies of the frequent words - stop words included ("the", "and", "of"),
       even when the bank has some: every extra copy makes repeats look less identical
       (target: as many copies as the word appears, up to 6);
    4. any other word of the text not written whole yet."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    by_freq = sorted(freq, key=lambda w: (-freq[w], len(w)))
    cov = coverage(bank)
    have = {ch for label in cov for ch in label}
    out: List[str] = []
    missing = sorted({ch for w in words for ch in w if ch.isalnum() and ch not in have})
    for ch in missing:                            # shortest frequent word with that letter
        cands = sorted((w for w in freq if ch in w), key=lambda w: (len(w), -freq[w]))
        if cands and cands[0] not in out:
            out.append(cands[0])
    for w in by_freq:                             # never written whole, used 2+ times
        if freq[w] >= 2 and len(w) >= 2 and cov.get(w, 0) == 0 and w not in out:
            out.append(w)
    for w in by_freq:                             # more copies of the frequent / stop words
        if (freq[w] < 2 and w.lower() not in STOP_WORDS) or len(out) >= limit:
            continue
        need = min(6, freq[w]) - cov.get(w, 0)
        if cov.get(w, 0) < 12:
            need = max(need, 1)                   # at least one fresh copy, even if some exist
        need -= out.count(w)
        out += [w] * max(0, min(need, limit - len(out)))
    for w in by_freq:                             # the rest, once
        if len(out) >= limit:
            break
        if len(w) >= 2 and cov.get(w, 0) == 0 and w not in out:
            out.append(w)
    return out[:limit]


def set_keep(bank: Path, file: str, keep: bool) -> None:
    """Restore (keep=True) or re-exclude a flagged piece, then rebuild."""
    path = bank / "audit.json"
    rep = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"flags": {}, "keep": []}
    ks = [k for k in rep.get("keep", []) if k != file] + ([file] if keep else [])
    rep["keep"] = ks
    path.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    build(bank)


TALL = set("bdfhklt0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?!()[]{}/\\|")
DESC = set("gjpqy,;")
# brackets span the text: from tall-letter height (~1.72 x-heights) to a little under
# the line (~0.33), whatever size they were written at
BRACKETS = set("()[]{}")
BRACKET_TOP, BRACKET_BOTTOM = 1.72, 0.33


def _ink_bottom(m: np.ndarray) -> float:
    """Where the ink as a whole ends (row below which only ~3 % of the ink lies).
    A vote over columns fails for letters like T, 7, F, Y whose stem is the only part
    reaching the baseline; tiny stray marks below are still ignored."""
    ys = np.nonzero(m)[0]
    return float(np.percentile(ys, 97)) + 1.0


def _trim_tail(piece: np.ndarray, label: str, ref: float) -> np.ndarray:
    """Repair instead of discard: a letter without descenders whose ink hangs under
    its baseline is carrying a stub of the next/previous letter (from the cut).
    Erase what lies clearly below the baseline (the ink bottom of most columns)."""
    if set(label) & DESC or not any(c.isalnum() for c in label):
        return piece
    m = piece > 100
    if not m.any():
        return piece
    base = _ink_bottom(m)
    limit = int(round(base + 0.15 * ref))
    if limit >= piece.shape[0]:
        return piece
    out = piece.copy()
    out[limit:] = 0
    rows = np.where((out > 40).any(1))[0]
    return out[:rows[-1] + 1] if len(rows) else piece


def _piece_metrics(piece: np.ndarray, base: float, ref: float, label: str,
                   clamp: Tuple[float, float] = (0.55, 1.5)) -> dict:
    """Measurements the layout pass uses to make pieces fit together:
    - baseline: refined from the piece's own bottom edge (the word-level estimate it
      inherits is rough), unless the piece has descenders;
    - size_fix: scale that brings the ink height above the baseline to what its letters
      should have (tall letters / capitals 1.80 x-heights, others 1.06, as measured), clamped;
    - stroke: pen width / x-height; entry / exit: height (x-heights above baseline)
      where the ink enters on the left and leaves on the right, for smooth joins."""
    m = piece > 100
    out = {"size_fix": 1.0, "stroke": 0.0, "entry": 0.5, "exit": 0.5}
    if m.sum() < 8:
        return out
    # ? and ! measured like (tall) letters: bottom on the baseline, capital height
    letters = [c for c in label if c.isalnum() or c in "?!"]
    if len(label) == 1 and label in "gjpqy":
        # a single tailed letter: the top of its main shape (biggest ink component, so a
        # j's dot doesn't count) is the x-height line, the baseline one x-height lower
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n > 1:
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            top = st[big, cv2.CC_STAT_TOP]
            base = float(top + ref)
            out["baseline"] = int(round(base))
    elif letters and not (set(label) & DESC):
        # no descenders: the bottom of the ink IS the baseline, however far the
        # word-level estimate was off (a piece sitting below it renders like a subscript)
        base = _ink_bottom(m)
        out["baseline"] = int(round(base))
    rows = np.where(m.any(1))[0]
    if len(label) == 1 and label in BRACKETS:
        # scaled to span the text (see BRACKET_TOP / BRACKET_BOTTOM); baseline as stored
        out["size_fix"] = float(np.clip((BRACKET_TOP + BRACKET_BOTTOM) * ref / max(1, rows[-1] + 1 - rows[0]),
                                        *clamp))
        out["desc"] = True
        out["below"] = float((rows[-1] + 1 - base) / ref)
        return out
    # height above the baseline from the letters themselves: small separate marks above
    # them (i/j dots, accents) don't count, or every word with an i came out too small
    n_cc, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    big = [i for i in range(1, n_cc) if st[i, cv2.CC_STAT_AREA] >= 0.04 * m.sum()]
    top = min(int(st[i, cv2.CC_STAT_TOP]) for i in big) if big else int(rows[0])
    body = base - top
    out["below"] = float((rows[-1] + 1 - base) / ref)       # how far ink hangs under the baseline
    out["desc"] = bool(set(label) & DESC)
    if letters and body > 0.3 * ref:
        # targets measured on the writer's own pieces: short-only letters stand
        # 1.06 page x-heights above the baseline, pieces with tall letters 1.80
        expect = (1.80 if any(c in TALL for c in letters) else 1.06) * ref
        out["size_fix"] = float(np.clip(expect / body, *clamp))
        if len(letters) >= 2:
            # words / letter groups: size by their letter (x-) height, the 40th percentile
            # of the column tops - ascenders, i-dots and accents cover few columns and
            # can't pull it up. Sizing by the highest point made words with tall
            # ascenders or i's come out with small letters (0.78-1.18 x-heights apart).
            # 0.967: the bank's median before, so the overall size stays the same.
            cols = np.where(m[:int(base)].any(0))[0] if base > 0 else []
            if len(cols) >= 3:
                h = np.array([base - int(np.argmax(m[:, c])) for c in cols], np.float32)
                xh_here = float(np.percentile(h, 40))
                if xh_here > 0.25 * ref:
                    out["size_fix"] = float(np.clip(0.967 * ref / xh_here, *clamp))
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    out["stroke"] = float(2.0 * m.sum() / max(1.0, perim) / ref)
    cols = np.where(m.any(0))[0]
    span = max(1, int(0.15 * (cols[-1] - cols[0] + 1)))
    for key, sl in (("entry", slice(cols[0], cols[0] + span)), ("exit", slice(cols[-1] + 1 - span, cols[-1] + 1))):
        ys = np.where(m[:, sl])[0]
        if len(ys):
            out[key] = float((base - np.median(ys)) / ref)
    return out


def coverage(bank: Path) -> Dict[str, int]:
    path = bank / "bank.json"
    if not path.exists():
        return {}
    counts: Dict[str, int] = {}
    for g in json.loads(path.read_text(encoding="utf-8"))["glyphs"]:
        counts[g["label"]] = counts.get(g["label"], 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════════════
# Runtime access for the engine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Glyph:
    label: str
    mask: Image.Image   # "L" alpha mask, deslanted
    baseline: int       # px from mask top
    xh: int             # writer's x-height in the source word
    ink_left: int
    ink_right: int
    file: str = ""
    ref_xh: float = 0.0  # median x-height of the source page: the piece's scale reference
    size_fix: float = 1.0  # height correction from the layout pass (glyphbank._piece_metrics)
    # Pen strokes (writing-pad samples): [[[x, y, pressure], ...], ...] in x-heights,
    # baseline at y=0, deslanted. When present the engine redraws them every time.
    strokes: Optional[list] = None


def pad_stretch(words: List[dict], glyphs: List[dict], strokes_dir: Optional[Path] = None) -> float:
    """How much wider the writer's paper hand is than their pen-tablet hand. Writing on
    a tablet without a screen (eyes on the monitor) comes out tall, narrow and cramped -
    letters touching or running into each other - while on paper the same writer's
    letters are rounder with gaps between them. Measured as the median width per letter
    (in each word's own x-heights) of the paper words (scans, photos) over the pad words.
    The engine widens pad pieces by this much. 1.0 when there's nothing to compare."""
    fix = {g["file"].rsplit("_", 1)[0]: g.get("size_fix", 1.0) for g in glyphs if g.get("source") == "pad"}
    paper, pad = [], []
    for w in words:
        if w.get("status") == "todo" or not w.get("xh") or not w.get("labels"):
            continue
        txt = "".join(l for l in w["labels"] if l and l != "_")
        n = sum(c.isalpha() for c in txt)
        if n < 3 or n != len(txt):
            continue
        if w.get("source") == "pad":
            if w["id"] not in fix or strokes_dir is None:
                continue
            path = strokes_dir / f"{w['id']}.json"
            if not path.exists():
                continue
            xs = [q[0] for st in json.loads(path.read_text(encoding="utf-8"))["strokes"] for q in st]
            # ink extent in pad x-heights, times the size correction = the letter's own x-heights
            if xs:
                pad.append((max(xs) - min(xs)) * fix[w["id"]] / n)
        else:
            paper.append(w["w"] / w["xh"] / n)
    if len(paper) < 20 or len(pad) < 20:
        return 1.0
    return float(np.clip(np.median(paper) / np.median(pad), 1.0, 1.6))


class GlyphBank:
    """Variants per label; longest-label tokenizing (sometimes shorter, for variety);
    least-recently-used variant choice so repeats are spread out."""

    def __init__(self, bank_dir: str) -> None:
        self.dir = Path(bank_dir)
        meta = json.loads((self.dir / "bank.json").read_text(encoding="utf-8"))
        self.slant_deg: float = meta.get("slant_deg", 0.0)
        self.pad_stretch: float = meta.get("pad_stretch", 1.0)
        self._meta: Dict[str, List[dict]] = {}
        for g in meta["glyphs"]:
            self._meta.setdefault(g["label"], []).append(g)
        self._cache: Dict[str, Glyph] = {}
        self._used: Dict[str, int] = {}      # file -> tick of last use
        from collections import deque
        self._recent = deque(maxlen=40)         # labels used for the last ~40 words
        self._page_used: Dict[str, int] = {}    # piece file -> uses on the current page
        self.page_reuse: List[Dict[str, int]] = [{}]   # per page: multi-letter label -> repeated uses
        self._tick = 0
        self.max_len = max((len(k) for k in self._meta), default=1)
        # All pieces come from the same scans at the same scale, so they share one
        # scale reference: per-word x-height estimates are too noisy to scale by.
        self.ref_xh: float = float(np.median([g["xh"] for g in meta["glyphs"]])) if meta["glyphs"] else 1.0
        strokes = [g["stroke"] for g in meta["glyphs"] if g.get("stroke")]
        self.stroke_ref: float = float(np.median(strokes)) if strokes else 0.0

    def new_page(self) -> None:
        """The engine started a page: page-level repetition counts start over."""
        self._page_used = {}
        self.page_reuse.append({})

    def __contains__(self, label: str) -> bool:
        return label in self._meta

    def tokenize(self, word: str) -> List[Tuple[str, bool]]:
        """Split a word into (token, from_bank) — longest bank label first, else 1 char."""
        out, i = [], 0
        while i < len(word):
            matches = [n for n in range(min(self.max_len, len(word) - i), 0, -1)
                       if word[i:i + n] in self._meta]
            if not matches:
                out.append((word[i], False))
                i += 1
                continue
            n = matches[0]
            # A joined piece with few variants is recognisable when reused: now and
            # then build it from shorter pieces instead.
            k = len(self._meta[word[i:i + n]])
            # Only into pieces of 2+ letters: single letters cut out of joined
            # writing are the least readable pieces in the bank.
            if n > 2 and matches[1] >= 2 and k < 4 and random.random() < 0.5 / k:
                n = matches[1]
            out.append((word[i:i + n], True))
            i += n
        return out

    def plan(self, word: str, stroke_ref: Optional[float] = None,
             font_cost=None) -> List[Tuple[str, Optional[dict]]]:
        """Choose how to write a whole word: which pieces (whole word, joined groups,
        single letters, or font fallback) and which variant of each, by dynamic
        programming over the word. Costs prefer few, long pieces; penalise single
        letters cut out of joined writing, recently used variants, pieces that need a
        big size correction or whose pen weight / slant stand out; and between pieces,
        a mismatch in the height where one stroke leaves and the next enters.
        Returns [(text, glyph-meta or None for font fallback)]."""
        n = len(word)
        stroke_ref = stroke_ref or self.stroke_ref
        # best[j] = (cost, path) for paths covering word[:j], keyed by last node
        best: Dict[int, Dict[object, Tuple[float, list]]] = {0: {None: (0.0, [])}}
        for i in range(n):
            if i not in best:
                continue
            nodes = []
            for L in range(min(self.max_len, n - i), 0, -1):
                label = word[i:i + L]
                for g in self._meta.get(label, ()):
                    nodes.append((i + L, label, g, self._unary(g, stroke_ref)))
            # font fallback (learned morph letters can be competitive: font_cost)
            nodes.append((i + 1, word[i], None, (font_cost(word[i]) if font_cost else 2.2) + random.gauss(0, 0.12)))
            for j, label, g, u in nodes:
                for prev_key, (c0, path) in best[i].items():
                    prev = path[-1][1] if path else None
                    c = c0 + u
                    if prev is not None and g is not None:
                        c += 0.6 * abs(prev.get("exit", 0.5) - g.get("entry", 0.5))
                    elif path and (prev is None) != (g is None):
                        c += 0.4                                   # font next to own writing
                    key = (label, g["file"] if g else None)
                    slot = best.setdefault(j, {})
                    if key not in slot or c < slot[key][0]:
                        slot[key] = (c, path + [(label, g)])
            # keep the search small
            for j in list(best):
                if j > i and len(best[j]) > 40:
                    best[j] = dict(sorted(best[j].items(), key=lambda kv: kv[1][0])[:40])
        path = min(best[n].values(), key=lambda v: v[0])[1]
        for _, g in path:
            if g is not None:
                self._tick += 1
                self._used[g["file"]] = self._tick
                self._recent.append(g["label"])
                n = self._page_used.get(g["file"], 0)
                if n and len(g["label"]) > 1:      # the exact same word image again on this page
                    self.page_reuse[-1][g["label"]] = self.page_reuse[-1].get(g["label"], 0) + 1
                self._page_used[g["file"]] = n + 1
        return path

    def _unary(self, g: dict, stroke_ref: float) -> float:
        c = 0.35                                              # per piece: fewer, longer pieces
        if len(g["label"]) > 1:
            # the same word written over and over in one document: make other ways of
            # writing it (smaller pieces, learned letters) progressively more attractive
            # counted over the last ~40 words only and capped, so later pages don't
            # drift into ever more broken-up words
            # (kept light: breaking a word into letters to dodge a repeat looked worse than
            # the repeat; the cure is more written copies - see the "Repeats" warning)
            c += min(0.2, 0.05 * sum(1 for l in self._recent if l == g["label"]))
        if g.get("source") == "scan":
            c += 0.35            # first scan: fast, messy hand; prefer notebook / pad pieces
        if g.get("cut") and len(g["label"]) == 1:
            c += 0.45                                         # lone letter cut from joined writing
        # the same piece already used on this page: another copy / composition looks
        # more natural (only repeats when nothing else is available)
        # Only whole words / joined groups are recognisable when repeated; a single
        # letter repeats all the time in real writing (and is re-penned each time), so
        # it gets a light penalty - a heavy one pushed half the letters onto the font.
        multi = len(g["label"]) > 1
        c += (0.45 if multi else 0.12) * min(3, self._page_used.get(g["file"], 0))
        last = self._used.get(g["file"])
        if last is not None:
            c += 0.3 * max(0.0, 1.0 - (self._tick - last) / 12.0)   # used recently: rotate variants
        sf = g.get("size_fix", 1.0)
        if g.get("source") == "pad":
            # clean pen strokes: a big size correction is reliable (people write small
            # on the pad), so only an extreme one counts against the piece
            if sf <= 0.46 or sf >= 2.39:
                c += 0.8
        else:
            c += 1.2 * abs(np.log(sf))
            if sf <= 0.67 or sf >= 1.49:
                c += 0.8                                      # still mis-sized after correction
        if not g.get("desc", True):
            c += 2.0 * max(0.0, g.get("below", 0.0) - 0.2)    # leftover tail under a tail-less letter
        if g.get("stroke") and stroke_ref:
            c += 0.8 * min(1.0, abs(g["stroke"] - stroke_ref) / stroke_ref)
        c += 0.015 * abs(g.get("slant", self.slant_deg) - self.slant_deg)
        # unlike the other copies of its label (a messy or odd sample): use it only when
        # nothing better is around
        c += 1.5 * max(0.0, 0.95 - g.get("typ", 1.0))
        return c + random.gauss(0, 0.12)                       # a little natural choice

    def load(self, g: dict) -> "Glyph":
        return self._load(g)

    def pick(self, label: str) -> Glyph:
        options = sorted(self._meta[label], key=lambda g: self._used.get(g["file"], -1))
        # random among the least recently used half (never a strict cycle)
        g = random.choice(options[:max(1, (len(options) + 1) // 2)])
        self._tick += 1
        self._used[g["file"]] = self._tick
        return self._load(g)

    def sample(self, label: str) -> Glyph:
        """Any variant, without touching the no-repeat state (for measuring)."""
        return self._load(self._meta[label][0])

    def _load(self, g: dict) -> Glyph:
        if g["file"] not in self._cache:
            mask = Image.open(self.dir / "glyphs" / g["file"]).convert("L")
            strokes = None
            if g.get("strokes"):
                path = self.dir / "strokes" / g["strokes"]
                if path.exists():
                    strokes = json.loads(path.read_text(encoding="utf-8"))["strokes"]
            self._cache[g["file"]] = Glyph(g["label"], mask, g["baseline"], g["xh"],
                                           g.get("ink_left", 0), g.get("ink_right", mask.width),
                                           g["file"], g.get("ref_xh", self.ref_xh), g.get("size_fix", 1.0),
                                           strokes)
            # pad strokes are drawn around the guide line; the measured baseline may sit
            # above / below it (words written high or low): the shift, in stroke units
            ref = float(g.get("ref_xh") or self.ref_xh or 30.0)
            self._cache[g["file"]].base_shift = (g["baseline"] - g.get("base0", g["baseline"])) / ref
        return self._cache[g["file"]]


# ═══════════════════════════════════════════════════════════════════════
# Writing-pad samples (pen strokes)
# ═══════════════════════════════════════════════════════════════════════

def _rdp(p: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: indices of the points that matter (deviation > eps)."""
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i, :2], p[j, :2]
        ab = b - a
        L = np.hypot(*ab)
        seg = p[i + 1:j, :2] - a
        d = (np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L) if L > 1e-9 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.nonzero(keep)[0]


def _catmull_rom(P: np.ndarray, step: float) -> np.ndarray:
    """Centripetal Catmull-Rom curve through the points P (n x 2), sampled every ~step."""
    if len(P) < 3:
        n = max(2, int(np.ceil(np.hypot(*(P[-1] - P[0])) / step)) + 1)
        t = np.linspace(0, 1, n)[:, None]
        return P[0] * (1 - t) + P[-1] * t
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = [P[:1]]
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.hypot(*(p1 - p0)), 1e-6) ** 0.5
        t2 = t1 + max(np.hypot(*(p2 - p1)), 1e-6) ** 0.5
        t3 = t2 + max(np.hypot(*(p3 - p2)), 1e-6) ** 0.5
        n = max(2, int(np.ceil(np.hypot(*(p2 - p1)) / step)) + 1)
        t = np.linspace(t1, t2, n)[1:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.vstack(out)


def _smooth_stroke(p: np.ndarray, eps: float = 0.035, step: float = 0.006) -> np.ndarray:
    """Pen points arrive on whole screen pixels (1/46-1/62 of an x-height on the pad):
    printed large, that shows as a staircase - also along nearly straight lines, where
    the 1-pixel jogs are far apart. Keep only the points that matter (RDP, tolerance
    eps x-heights ~ 2 pad pixels: jogs and tremor go, real corners and curves stay) and draw a
    smooth curve through them (centripetal Catmull-Rom). Pressure is smoothed along the
    stroke and carried over by position along it. Columns: x, y, pressure."""
    if len(p) < 3:
        return p
    keep = np.concatenate([[True], np.any(np.diff(p[:, :2], axis=0) != 0, axis=1)])
    if not keep.all() and p.shape[1] > 2:
        # the pen resting on one spot: its attributes (e.g. ink amount) are the max there
        starts = np.nonzero(keep)[0]
        p = np.column_stack([p[starts, :2], np.maximum.reduceat(p[:, 2:], starts, axis=0)])
    else:
        p = p[keep]
    if len(p) < 3:
        return p
    seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
    t = np.concatenate([[0], np.cumsum(seg)])
    if t[-1] < 4 * step:
        return p
    # attributes (pressure, ink amount, ...): Gaussian along the stroke
    u = np.linspace(0, t[-1], max(8, int(t[-1] / step)))
    n = max(1, int(round(0.04 / step)))
    k = np.exp(-0.5 * (np.arange(-3 * n, 3 * n + 1) / n) ** 2)
    k /= k.sum()
    attrs = []
    for c in range(2, p.shape[1]):
        a = np.interp(u, t, p[:, c])
        attrs.append(np.convolve(np.concatenate([np.full(3 * n, a[0]), a, np.full(3 * n, a[-1])]), k, mode="valid"))
    # shape: significant points -> smooth curve
    idx = _rdp(p, eps)
    curve = _catmull_rom(p[idx, :2], step)
    cl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
    frac = cl / max(cl[-1], 1e-9)
    return np.column_stack([curve] + [np.interp(frac, u / t[-1], a) for a in attrs])


def _vary_strokes(strokes: List[np.ndarray], amp: float, unit: float, rng) -> List[np.ndarray]:
    """One use of a pad sample, written again: the same word by the same hand is never
    the same picture twice. Only slow, whole-word changes (a fast wobble looks fake):
    letters drift a little closer or further apart along the word, the baseline curves
    gently, and separate small strokes (i-dots, t-crosses, accents) land a bit
    differently. amp / unit: one x-height in stroke units."""
    two_pi = 2 * np.pi
    L1, a1, p1 = rng.uniform(1.6, 3.0) * unit, rng.uniform(0.04, 0.08) * amp, rng.uniform(0, two_pi)
    L2, a2, p2 = rng.uniform(2.0, 4.0) * unit, rng.uniform(0.02, 0.045) * amp, rng.uniform(0, two_pi)
    out = []
    for p in strokes:
        q = p.copy()
        length = float(np.hypot(*np.diff(p[:, :2], axis=0).T).sum()) if len(p) > 1 else 0.0
        small = length < 0.45 * unit
        q[:, 0] = p[:, 0] + a1 * np.sin(two_pi * p[:, 0] / L1 + p1) + rng.gauss(0, (0.06 if small else 0.015) * amp)
        q[:, 1] = p[:, 1] + a2 * np.sin(two_pi * p[:, 0] / L2 + p2) + rng.gauss(0, (0.05 if small else 0.012) * amp)
        out.append(q)
    return out


def _trim_exit(strokes: List[np.ndarray], max_tail: float) -> List[np.ndarray]:
    """Shorten a long flat exit stroke: on the tablet the pen keeps travelling right
    after the last letter (a third of the word's width for some samples), which next to
    the following word reads as a smear. On paper the writer's words end cleanly. Only
    the stroke that ends at the sample's right edge, only its final near-flat rightward
    run, and only the part beyond max_tail (stroke units) is cut."""
    if not strokes or max_tail <= 0:
        return strokes
    right = max(float(p[:, 0].max()) for p in strokes)
    out = list(strokes)
    for i, p in enumerate(strokes):
        if len(p) < 4 or p[-1, 0] < right - 0.05 * max_tail:
            continue
        j = len(p) - 1
        while j > 0:                          # walk back along the flat rightward run
            dx, dy = p[j, 0] - p[j - 1, 0], p[j, 1] - p[j - 1, 1]
            if dx <= 0 or abs(dy) > 0.6 * dx:
                break
            j -= 1
        run = p[-1, 0] - p[j, 0]
        # an exit tail trails low, after the letter; a t's cross or a dash is a stroke of
        # its own (all flat) at letter height - those stay as written
        low = float(p[j:, 1].mean()) > -1.5 * max_tail
        if j > 0 and low and run > max_tail:
            keep = np.searchsorted(p[j:, 0], p[j, 0] + max_tail) + j
            out[i] = p[:max(keep, j + 2)]
    return out


def _cusps(x: np.ndarray, y: np.ndarray, reach: float) -> List[int]:
    """Indices where the stroke turns back on itself (the bottom of an n, the top of
    an m's arch): the pen stops there and the way back covers the way down."""
    n = len(x)
    if n < 7:
        return []
    seg = np.hypot(np.diff(x), np.diff(y))
    k = max(2, int(round(reach / max(float(np.median(seg)), 1e-6))))
    if n <= 2 * k + 1:
        return []
    i = np.arange(k, n - k)
    v1 = np.stack([x[i] - x[i - k], y[i] - y[i - k]], 1)
    v2 = np.stack([x[i + k] - x[i], y[i + k] - y[i]], 1)
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    out = []
    for j in np.nonzero(cos < -0.55)[0]:                 # turned by more than ~125 degrees
        c = int(i[j])
        if (not out or c - out[-1] > 2 * k) and cos[j] <= cos[max(0, j - k):j + k + 1].min():
            out.append(c)
    return out


_SMOOTH_CACHE: Dict[int, tuple] = {}   # id(strokes) -> (strokes, smoothed strokes)


def render_strokes(strokes: list, xh_px: float, width_px: float, jitter: float = 0.0,
                   sx: float = 1.0, sy: float = 1.0, rng: Optional[random.Random] = None,
                   dens: Optional[list] = None, max_below: Optional[float] = None,
                   max_tail: Optional[float] = None, vary: float = 0.0, unit: float = 1.0,
                   y_shift: float = 0.0) -> Tuple[Image.Image, float, float, float]:
    """Draw pen strokes (x-height units, baseline y=0) as an "L" mask.
    jitter > 0 wobbles every stroke smoothly (fraction of x-height) and varies
    pressure, so each rendering of the same sample differs like real writing.
    Returns (mask, baseline y, ink left x, ink right x).

    dens: pass a list to also get the ink amount per pixel (an "L" image appended to
    it, same size as the mask; decode with ink_amount()). Like a ballpoint: slow parts
    and pauses lay down more ink (measured from the spacing of the pen's events), the
    pen lands with a small dark blob and lifts with a thinner, lighter tail, and where
    the line crosses or retraces itself (turn-backs) there are two layers of ink."""
    rng = rng or random.Random()
    S = 3                                             # supersampling for smooth edges
    raw = []
    for st in strokes:
        p = np.array(st, np.float64)
        if len(p) == 0:
            continue
        if p.shape[1] < 3:
            p = np.column_stack([p[:, :2], np.full(len(p), 0.5)])
        raw.append(p[:, :3])
    if not raw:
        return Image.new("L", (1, 1), 0), 0.0, 0.0, 1.0
    # speed: the pen reports at a steady rate, so the gap between events is its speed
    gaps = np.concatenate([np.hypot(*np.diff(p[:, :2], axis=0).T) for p in raw if len(p) > 1] or [np.ones(1)])
    gaps = gaps[gaps > 0]
    # the pad drops repeated positions, so slow writing shows as the smallest steps:
    # measure speed against the slow end, not the median
    vmed = float(np.percentile(gaps, 20)) if len(gaps) else 1.0
    hit = _SMOOTH_CACHE.get(id(strokes))
    if hit is not None and hit[0] is strokes:
        smoothed = hit[1]                             # same sample again: shape is fixed
    else:
        smoothed = []
        for p in raw:
            if len(p) > 1:
                g = np.hypot(*np.diff(p[:, :2], axis=0).T)
                g = np.concatenate([[g[0]], g])
            else:
                g = np.full(1, vmed)
            ink = np.clip(1.2 - 0.45 * (g / vmed - 1), 0.55, 1.2)  # slow: more ink, fast: less
            smoothed.append(_smooth_stroke(np.column_stack([p, ink])))
        if len(_SMOOTH_CACHE) > 4000:
            _SMOOTH_CACHE.clear()
        _SMOOTH_CACHE[id(strokes)] = (strokes, smoothed)
    if y_shift:
        # move the writer's real baseline (measured from the ink) onto y = 0
        smoothed = [np.column_stack([p[:, 0], p[:, 1] - y_shift, p[:, 2:]]) for p in smoothed]
    if max_tail:
        smoothed = _trim_exit(smoothed, max_tail)
    if vary > 0:
        smoothed = _vary_strokes(smoothed, vary * unit, unit, rng)
    # max_below (stroke units): tails longer than that are compressed smoothly from half
    # way down (shape kept), so a tablet-long loop doesn't run into the next line
    lowest = max(float(p[:, 1].max()) for p in smoothed)
    squash = None
    if max_below and lowest > max_below:
        k0 = 0.5 * max_below
        squash = (k0, (max_below - k0) / (lowest - k0))
    pts_all = []
    for p in smoothed:
        yy = p[:, 1]
        if squash is not None:
            yy = np.where(yy > squash[0], squash[0] + (yy - squash[0]) * squash[1], yy)
        x, y, pr = p[:, 0] * xh_px * sx, yy * xh_px * sy, p[:, 2].copy()
        ink = p[:, 3].copy() if p.shape[1] > 3 else np.ones(len(p))
        if jitter > 0 and len(p) > 1:
            t = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]) / max(xh_px, 1)
            for arr in (x, y):
                for f in (rng.uniform(0.6, 1.4), rng.uniform(1.8, 3.2)):
                    arr += rng.gauss(0, jitter * xh_px) * np.sin(np.pi * f * t + rng.uniform(0, 6.3))
            pr = np.clip(pr * rng.uniform(0.85, 1.15) + 0.08 * np.sin(t * rng.uniform(1, 3)), 0.05, 1.0)
        pts_all.append((x, y, pr, ink))
    # pressure relative to this sample's own median: a pen pressed hard (or a mouse)
    # must not make the whole sample heavier than the other pieces of a word - only
    # the swelling and thinning within the sample is kept
    med = float(np.median(np.concatenate([a[2] for a in pts_all]))) or 0.5
    pad = width_px * 2 + 2
    minx = min(a[0].min() for a in pts_all) - pad
    miny = min(a[1].min() for a in pts_all) - pad
    maxx = max(a[0].max() for a in pts_all) + pad
    maxy = max(a[1].max() for a in pts_all) + pad
    W, H = int(np.ceil(maxx - minx)) + 1, int(np.ceil(maxy - miny)) + 1
    cover = np.zeros((H * S, W * S), np.uint8)        # how many ink layers
    amount = np.ones((H * S, W * S), np.float32)      # ink amount (1 = one normal layer)
    for x, y, pr, ink in pts_all:
        X, Y = (x - minx) * S, (y - miny) * S
        wd = width_px * np.clip(1.0 + 0.45 * (pr / med - 1.0), 0.7, 1.35) * S
        if len(X) > 1:
            # a round pen tip stamped densely along the (smoothed) path: one smooth tube
            seg = np.hypot(np.diff(X), np.diff(Y))
            t = np.concatenate([[0], np.cumsum(seg)])
            u = np.arange(0, t[-1] + 1e-9, max(0.6, 0.12 * float(np.min(wd))))
            X, Y, wd, ink = (np.interp(u, t, X), np.interp(u, t, Y), np.interp(u, t, wd), np.interp(u, t, ink))
            # landing blob / lifting tail (distances in x-heights from each end)
            xs = xh_px * S
            d0, d1 = u / xs, (u[-1] - u) / xs
            land, lift = np.exp(-(d0 / 0.045) ** 2), np.exp(-(d1 / 0.07) ** 2)
            wd = wd * (1 + 0.1 * land) * (1 - 0.28 * lift)
            ink = ink * (1 + 0.3 * land) * (1 - 0.22 * lift)
            # turn-backs only on real strokes (a dot or tick isn't retraced)
            cuts = [0] + (_cusps(X, Y, 0.05 * xs) if u[-1] > 0.35 * xs else []) + [len(X) - 1]
        else:
            cuts = [0, 0]
        # each part between turn-backs is its own layer: where they overlap, two layers
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            sl = slice(c0, c1 + 1)
            r = wd[sl] / 2
            bx0 = max(0, int(np.floor((X[sl] - r).min())) - 1)
            by0 = max(0, int(np.floor((Y[sl] - r).min())) - 1)
            bx1 = min(W * S, int(np.ceil((X[sl] + r).max())) + 2)
            by1 = min(H * S, int(np.ceil((Y[sl] + r).max())) + 2)
            if bx1 <= bx0 or by1 <= by0:
                continue
            lay = Image.new("L", (bx1 - bx0, by1 - by0), 0)
            amt = Image.new("L", lay.size, 0)
            dl, da = ImageDraw.Draw(lay), ImageDraw.Draw(amt)
            xs_, ys_ = (X[sl] - bx0).tolist(), (Y[sl] - by0).tolist()
            rs_ = r.tolist()
            ks_ = np.clip(ink[sl] * 100, 1, 255).astype(int).tolist()
            for j, (xi, yi, ri) in enumerate(zip(xs_, ys_, rs_)):
                dl.ellipse((xi - ri, yi - ri, xi + ri, yi + ri), fill=255)
                if j % 3 == 0:                   # the ink amount varies slowly: fewer stamps
                    rr = ri * 1.1
                    da.ellipse((xi - rr, yi - rr, xi + rr, yi + rr), fill=ks_[j])
            on = np.array(lay) > 127
            cover[by0:by1, bx0:bx1] += on
            reg = amount[by0:by1, bx0:bx1]
            a = np.array(amt, np.float32) / 100
            reg[on] = np.maximum(np.where(reg[on] == 1, 0, reg[on]), a[on])
    mask = Image.fromarray(np.where(cover > 0, 255, 0).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    cols = np.where(np.array(mask).max(0) > 40)[0]
    il, ir = (float(cols[0]), float(cols[-1]) + 1) if len(cols) else (0.0, float(W))
    if dens is not None:
        # two layers of ink: the light passes through both (about one more layer's worth)
        amount = np.minimum(np.where(cover > 0, amount, 1.0) + 0.8 * (cover >= 2), 2.0)
        small = cv2.resize(amount, (W, H), interpolation=cv2.INTER_AREA)
        dens.append(Image.fromarray(np.clip((small - 0.5) / 2.0 * 255, 0, 255).astype(np.uint8), "L"))
    return mask, float(-miny), il, ir


def ink_amount(dens: Image.Image) -> np.ndarray:
    """Decode render_strokes' ink-amount image (1.0 = one normal layer)."""
    return 0.5 + np.asarray(dens, np.float32) / 255 * 2.0


def _scan_ref_xh(words: List[dict]) -> float:
    scan = [w["xh"] for w in words if w.get("source") != "pad"]
    return float(np.median(scan)) if scan else 30.0


def add_pad_sample(bank: Path, text: str, strokes: list, baseline: float, x_height: float,
                   rebuild: bool = True) -> dict:
    """Store a word written on a pen tablet. strokes: [[[x, y, pressure], ...], ...] in
    canvas px; baseline / x_height: the canvas guide lines. Label = text (no cutting)."""
    with _BUILD_LOCK:
        return _add_pad_sample(bank, text, strokes, baseline, x_height, rebuild)


def _add_pad_sample(bank: Path, text: str, strokes: list, baseline: float, x_height: float,
                    rebuild: bool) -> dict:
    text = "".join(text.split())
    strokes = [st for st in strokes if len(st) >= 1]
    if not text or not strokes:
        raise ValueError("Empty sample")
    words = load_words(bank)
    ref = _scan_ref_xh(words)
    layout = load_layout(bank)
    width = (layout["stroke"] if layout else 0.11) * ref

    # canvas px -> x-height units, baseline at 0
    norm = [[[p[0] / x_height, (p[1] - baseline) / x_height, float(p[2]) if len(p) > 2 else 0.5]
             for p in st] for st in strokes]
    # remove the writer's slant in vector space (same estimator as the scans)
    probe, _, _, _ = render_strokes(norm, ref, width)
    slant = _estimate_slant((np.array(probe) > 100).astype(np.uint8))
    t = float(np.tan(np.radians(slant)))
    norm = [[[x + y * t, y, p] for x, y, p in st] for st in norm]
    x0 = min(p[0] for st in norm for p in st)
    norm = [[[round(x - x0, 4), round(y, 4), round(p, 3)] for x, y, p in st] for st in norm]

    mask, base, _, _ = render_strokes(norm, ref, width)
    n = 1 + sum(1 for w in words if w.get("source") == "pad")
    wid = f"pad_{n:04d}"
    while any(w["id"] == wid for w in words):
        n += 1
        wid = f"pad_{n:04d}"
    a = np.array(mask)
    Image.fromarray(np.dstack([np.zeros_like(a)] * 3 + [a]), "RGBA").save(bank / "words" / f"{wid}.png")
    (bank / "strokes").mkdir(exist_ok=True)
    (bank / "strokes" / f"{wid}.json").write_text(json.dumps({"text": text, "strokes": norm}), encoding="utf-8")
    rec = {"id": wid, "page": 0, "line": 0, "w": int(a.shape[1]), "h": int(a.shape[0]),
           "baseline": int(round(base)), "xh": int(round(ref)), "slant": slant, "cuts": [],
           "labels": [text], "status": "done", "source": "pad"}
    words.append(rec)
    save_words(bank, words)
    if rebuild:
        build(bank)
    return rec


def pad_entry(bank: Path, rec: dict) -> Optional[dict]:
    """The bank.json entry `build` would make for one pad sample, made on its own (and
    its glyph image written) so the editor can use a new sample at once while the full
    rebuild runs in the background. Skips the audit: the writer just wrote it for this."""
    words = load_words(bank)
    ref = page_refs(words)[_page_key(rec)]
    alpha = np.array(Image.open(bank / "words" / f"{rec['id']}.png"))[:, :, 3]
    rows = np.where(alpha.max(1) > 40)[0]
    cols = np.where(alpha.max(0) > 40)[0]
    if len(rows) == 0:
        return None
    top, bot = int(rows[0]), int(rows[-1]) + 1
    piece = alpha[top:bot]
    fname = f"{rec['id']}_0.png"
    (bank / "glyphs").mkdir(exist_ok=True)
    Image.fromarray(piece, "L").save(bank / "glyphs" / fname)
    label = rec["labels"][0]
    entry = {"label": label, "file": fname, "baseline": rec["baseline"] - top, "xh": rec["xh"],
             "ref_xh": ref, "ink_left": int(cols[0]), "ink_right": int(cols[-1]) + 1,
             "slant": rec["slant"], "cut": False, "source": "pad", "strokes": f"{rec['id']}.json"}
    entry["base0"] = entry["baseline"]
    entry.update(_piece_metrics(piece, entry["baseline"], ref, label, (0.45, 2.4)))
    return entry


def _rerender_pad_word(bank: Path, w: dict, strokes: list, ref: float, width: float) -> None:
    mask, base, _, _ = render_strokes(strokes, ref, width)
    a = np.array(mask)
    Image.fromarray(np.dstack([np.zeros_like(a)] * 3 + [a]), "RGBA").save(bank / "words" / f"{w['id']}.png")
    w.update(w=int(a.shape[1]), h=int(a.shape[0]), baseline=int(round(base)))


def fix_pad_baselines(bank: Path, rebuild: bool = True) -> int:
    """Single characters written on the wrong guide line (e.g. on the dashed x-height
    line instead of the red baseline) sit too high or low. Letters without descenders
    are snapped so their ink bottom is the baseline; descender letters (g j p q y ,;)
    move by the offset their neighbours from the same session had. Returns how many
    samples were moved."""
    words = load_words(bank)
    pads = [w for w in words if w.get("source") == "pad" and w.get("labels") and len(w["labels"][0]) == 1]
    data = {}
    for w in pads:
        p = bank / "strokes" / f"{w['id']}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            ys = np.array([q[1] for st in d["strokes"] for q in st])
            if len(ys):
                data[w["id"]] = (d, float(np.percentile(ys, 92)))
    order = [w["id"] for w in pads if w["id"] in data]
    shifts = {}
    for i, wid in enumerate(order):
        ch = next(w for w in pads if w["id"] == wid)["labels"][0]
        if ch in DESC:
            # tailed letters: the top of the main shape (arms of y, bowl of g/p/q, the
            # j's stem — its dot ignored) is the x-height line, one x-height above the
            # baseline. Measured on the letter itself, not borrowed from neighbours.
            d = data[wid][0]
            main = [st for st in d["strokes"] if len(st) > 1 and np.ptp([q[1] for q in st]) > 0.25]
            ys = np.array([q[1] for st in (main or d["strokes"]) for q in st])
            top = float(np.percentile(ys, 3))
            if abs(top + 1.0) > 0.2:
                shifts[wid] = -1.0 - top
        elif ch in BRACKETS:
            # centre it where a full-height bracket's centre goes (it gets scaled to
            # full height around the baseline by size_fix)
            ys = np.array([q[1] for st in data[wid][0]["strokes"] for q in st])
            h = float(ys.max() - ys.min())
            if h > 0:
                k = (BRACKET_TOP + BRACKET_BOTTOM) / h
                dy = (BRACKET_BOTTOM - BRACKET_TOP) / 2 / k - (ys.max() + ys.min()) / 2
                if abs(dy) > 0.03:
                    shifts[wid] = dy
        elif ch in "?!":
            # the dot is the bottom and sits on the baseline (a dot hanging under it
            # made the "?" spill down below the line)
            bottom = float(max(q[1] for st in data[wid][0]["strokes"] for q in st))
            if abs(bottom) > 0.06:
                shifts[wid] = -bottom
        else:
            bottom = data[wid][1]
            if abs(bottom) > 0.35:
                shifts[wid] = -bottom
    if not shifts:
        return 0
    ref = _scan_ref_xh(words)
    layout = load_layout(bank)
    width = (layout["stroke"] if layout else 0.11) * ref
    for w in pads:
        if w["id"] not in shifts:
            continue
        d, _ = data[w["id"]]
        dy = shifts[w["id"]]
        d["strokes"] = [[[x, round(y + dy, 4), pr] for x, y, pr in st] for st in d["strokes"]]
        (bank / "strokes" / f"{w['id']}.json").write_text(json.dumps(d), encoding="utf-8")
        _rerender_pad_word(bank, w, d["strokes"], ref, width)
    save_words(bank, words)
    if rebuild:
        build(bank)
    return len(shifts)


def list_pad_samples(bank: Path) -> List[dict]:
    """Pad samples, newest first: [{id, text}]."""
    out = [{"id": w["id"], "text": (w.get("labels") or [""])[0]}
           for w in load_words(bank) if w.get("source") == "pad"]
    return out[::-1]


def delete_pad_sample(bank: Path, wid: str) -> bool:
    """Remove a pad sample (record, image, strokes) and rebuild."""
    words = load_words(bank)
    keep = [w for w in words if not (w["id"] == wid and w.get("source") == "pad")]
    if len(keep) == len(words):
        return False
    save_words(bank, keep)
    for p in (bank / "words" / f"{wid}.png", bank / "strokes" / f"{wid}.json"):
        try:
            p.unlink()
        except OSError:
            pass
    build(bank)
    return True


COMMON_WORDS = ("the and of to a in is that for it as with was on be by this are from at or an not "
                "have which their can its has more also will they each data water use into than "
                "between through after before during these those such other").split()
RARE_WORDS = ("quick jump box quiz zoo yes you your year key kind next six zero lazy jazz jog quite "
              "query sky type explain extra project subject object analysis quality equal major "
              "July Kerala JSON XML CSV API").split()
CAP_WORDS = ("The A In It This They India National Data Water Kerala Department January March "
             "Monday Zone Quality Valley Bangalore Chennai Delhi Pune Goa Odisha Report Figure "
             "Table Section Government Ministry Survey University Hourly Live Open").split()


def prompts(bank: Path, n: int = 40) -> List[str]:
    """What to ask for on the writing pad: words covering missing letters/capitals
    first, then common words that have few variants (so repeats stop standing out)."""
    cov = coverage(bank)
    have = {ch for label in cov for ch in label}
    out: List[str] = []
    missing = [ch for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if ch not in have]
    pool = RARE_WORDS + CAP_WORDS + COMMON_WORDS
    for ch in missing:
        w = next((w for w in pool if ch in w and w not in out), None)
        out.append(w if w else ch)
    for w in COMMON_WORDS:
        out += [w] * max(0, 4 - cov.get(w, 0))
    out += [w for w in RARE_WORDS + CAP_WORDS if w not in out]
    seen: Dict[str, int] = {}
    res = []
    for w in out:                     # keep order, at most 4 asks per word
        seen[w] = seen.get(w, 0) + 1
        if seen[w] <= 4:
            res.append(w)
    return res[:n]


def add_sketch(bank: Path, name: str, strokes: list, x_height: Optional[float] = None) -> str:
    """Save a free drawing from the pad (canvas px strokes) as sketches/<name>.json.
    Use it in a document as ![caption](pad:<name>)."""
    name = re.sub(r"[^A-Za-z0-9_\-]+", "-", name.strip()).strip("-")[:40] or "sketch"
    strokes = [[[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]) if len(p) > 2 else 0.5, 3)]
                for p in st] for st in strokes if st]
    if not strokes:
        raise ValueError("Empty sketch")
    (bank / "sketches").mkdir(exist_ok=True)
    (bank / "sketches" / f"{name}.json").write_text(
        json.dumps({"name": name, "strokes": strokes, "x_height": x_height}), encoding="utf-8")
    return name


def list_sketches(bank: Path) -> List[str]:
    return sorted(p.stem for p in (bank / "sketches").glob("*.json")) if (bank / "sketches").exists() else []


def find_banks(root: Path) -> List[str]:
    return sorted(p.parent.name for p in root.glob("*/words.json"))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Build a glyph bank from scanned handwriting.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="segment scans into word images")
    ex.add_argument("sources", nargs="+", help="PDF or image files")
    ex.add_argument("--bank", required=True, help="bank folder, e.g. banks/mine")
    ex.add_argument("--append", action="store_true", help="add these pages to an existing bank (keeps labels)")
    ms = sub.add_parser("measure", help="measure the writer's page layout -> layout.json")
    ms.add_argument("sources", nargs="+", help="the same PDF or image files used for extract")
    ms.add_argument("--bank", required=True)
    so = sub.add_parser("sources", help="list sources, or turn one on/off for generation")
    so.add_argument("--bank", required=True)
    so.add_argument("--disable", help="source name to leave out (labels are kept)")
    so.add_argument("--enable", help="source name to use again")
    au = sub.add_parser("audit", help="check pieces; flagged ones are left out of generation")
    au.add_argument("--bank", required=True)
    au.add_argument("--ocr", action="store_true", help="also let the local OCR model read multi-letter pieces")
    bd = sub.add_parser("build", help="rebuild bank.json from labeled words")
    bd.add_argument("--bank", required=True)
    args = ap.parse_args()

    bank = Path(args.bank)
    if args.cmd == "audit":
        if args.ocr:
            audit(bank, use_ocr=True)            # fills the OCR cache, then build applies it
        meta = build(bank)
        rep = json.loads((bank / "audit.json").read_text(encoding="utf-8"))
        for f, why in sorted(rep["flags"].items()):
            print(f"{'kept ' if f in rep['keep'] else 'OUT  '} {f}: {'; '.join(why)}")
        print(f"{len(meta['glyphs'])} pieces in use, {meta['excluded']} left out "
              f"({len(rep['flags'])} flagged; mild ones are kept where a letter has no better copy)")
        return
    if args.cmd == "sources":
        if args.disable or args.enable:
            set_source(bank, args.disable or args.enable, enabled=bool(args.enable))
        off = set(disabled_sources(bank))
        counts: Dict[str, List[int]] = {}
        for w in load_words(bank):
            c = counts.setdefault(w.get("source", "scan"), [0, 0])
            c[0] += 1
            c[1] += w["status"] == "done"
        for src, (n, done) in counts.items():
            print(f"{'off' if src in off else 'on ':3}  {src}: {done} labeled / {n} words")
        return
    if args.cmd == "measure":
        print(json.dumps(measure_layout(args.sources, bank), indent=1))
        return
    if args.cmd == "extract":
        if (bank / "words.json").exists() and not args.append:
            raise SystemExit(f"{bank}/words.json exists; use --append to add pages "
                             f"(or delete it to re-extract; labels would be lost).")
        print(f"{extract(args.sources, bank, args.append)} words -> {bank / 'words'}")
        if args.append:
            build(bank)       # layout.json is kept: it describes the original pages
        else:
            measure_layout(args.sources, bank)
            print(f"layout -> {bank / 'layout.json'}")
    else:
        meta = build(bank)
        print(f"{len(meta['glyphs'])} glyphs, slant {meta['slant_deg']:.1f}°")


if __name__ == "__main__":
    main()
