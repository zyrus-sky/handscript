"""
Structured documents: Markdown → handwritten pages.

Blocks: headings (larger, underlined), paragraphs with inline $math$, bullet and
numbered lists (nested, hanging indent), tables (hand-ruled grid, wrapped cells),
```mermaid flowcharts (flowchart.py), $$display math$$ / \\[..\\] / ```math (mathlayout.py),
``` code (lines and indentation kept), > quotes (indented), --- rules (hand line),
![caption](pad:name) sketches drawn on the writing pad. Plain text without Markdown
is just paragraphs, so old inputs render as before.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from markdown_it import MarkdownIt

import flowchart
from mathlayout import MathWriter

INLINE_MATH = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")


def _prepare(text: str) -> str:
    """Normalise newlines; turn $$..$$ and \\[..\\] into ```math blocks for the parser."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: f"\n```math\n{m.group(1).strip()}\n```\n", text, flags=re.S)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"\n```math\n{m.group(1).strip()}\n```\n", text, flags=re.S)
    return text


def _inline_text(tok) -> Tuple[str, List[Tuple[str, str]]]:
    """Plain text of an inline token (emphasis/links stripped) + any images in it."""
    out, images = [], []
    for c in tok.children or []:
        if c.type in ("text", "code_inline"):
            out.append(c.content)
        elif c.type in ("softbreak", "hardbreak"):
            out.append(" ")
        elif c.type == "image":
            images.append((c.attrs.get("src", ""), c.content))
    return "".join(out), images


class DocWriter:
    def __init__(self, eng):
        self.e = eng
        self.math = MathWriter(eng)
        self.xh = eng._x_height
        self.base_left = None

    # ── helpers ──────────────────────────────────────────────────────────

    def lh(self, scale: float = 1.0) -> float:
        return self.e.cfg.line_height * scale * (1 + 0.5 * (self.e._page_scale - 1))

    def gap(self, k: float = 1.0) -> None:
        self.e._y += self.e.cfg.paragraph_spacing * k

    def start_line(self, scale: float = 1.0) -> None:
        """Begin writing on a fresh line (no blank line if the cursor is already at
        the start of an empty line)."""
        e = self.e
        if e._x > e._left + 2 * self.xh:
            e.newline(scale)
        else:
            e._x = e._left + random.uniform(*e.cfg.margin_left_jitter) * 0.5
            e._new_line_params()
        if e._needs_page_break():
            e._page_break()

    def words_with_math(self, text: str, scale: float = 1.0) -> None:
        pos = 0
        for m in INLINE_MATH.finditer(text):
            self.e.write_text(text[pos:m.start()], scale)
            self.inline_math(m.group(1), scale)
            pos = m.end()
        self.e.write_text(text[pos:], scale)

    def inline_math(self, latex: str, scale: float) -> None:
        e = self.e
        box = self.math.layout(latex, display=False, scale=scale)
        if e._x + box.w > e._right and e._x > e._line_start_x + 1:
            e.newline(scale)
        box.draw(e._x, e.baseline_y())
        e._x += box.w + random.uniform(*e.cfg.word_spacing) * scale

    def measure_lines(self, text: str, max_w: float, scale: float):
        """Greedy wrap for boxes/cells: (lines, width, height)."""
        e = self.e
        space = sum(e.cfg.word_spacing) / 2 * scale
        lines, cur, cur_w, widest = [], [], 0.0, 0.0
        for w in text.split():
            ww = e.word_width(w, scale) * 1.18
            if cur and cur_w + space + ww > max_w:
                lines.append(" ".join(cur))
                widest = max(widest, cur_w)
                cur, cur_w = [], 0.0
            cur_w += (space if cur else 0) + ww
            cur.append(w)
        if cur:
            lines.append(" ".join(cur))
            widest = max(widest, cur_w)
        n = max(1, len(lines))
        # the variant picked at render time can be wider than the measured one
        return lines or [""], widest * 1.18, n * self.lh(scale)

    def write_lines_at(self, lines: List[str], cx: float, cy: float, scale: float, centre: bool = True) -> None:
        """Write lines centred on (cx, cy) (or starting at cx), without wrapping."""
        e = self.e
        saved = (e._x, e._y, e._left, e._right, e._line_start_x)
        lh = self.lh(scale)
        top = cy - lh * len(lines) / 2
        for k, line in enumerate(lines):
            w = sum(e.word_width(t, scale) for t in line.split()) + \
                sum(e.cfg.word_spacing) / 2 * scale * max(0, len(line.split()) - 1)
            e._x = cx - w / 2 if centre else cx
            # baseline at ~70 % down the line slot
            e._y = top + lh * k + 0.72 * lh - e._ascent * scale
            e._left, e._right = e._x, e._x + w + 4 * self.xh
            e._new_line_params()
            e._line_slope *= 0.3
            for word in line.split():
                e.write_word(word, scale, wrap=False)
        e._x, e._y, e._left, e._right, e._line_start_x = saved

    # ── blocks ───────────────────────────────────────────────────────────

    def heading(self, text: str, level: int) -> None:
        e = self.e
        scale = {1: 1.3, 2: 1.15}.get(level, 1.05)
        self.gap(0.8)
        self.start_line(scale)
        e.ensure_room(self.lh(scale) * 2)
        x0 = e._x
        self.words_with_math(text, scale)
        if level <= 2:
            y = e.baseline_y() + 0.45 * self.xh * scale
            e.hand_line(x0 - 0.2 * self.xh, y, e._x - 0.4 * self.xh, y + random.gauss(0, 0.1) * self.xh, wobble=0.6)
        self.gap(0.3)

    def paragraph(self, text: str) -> None:
        self.start_line()
        self.words_with_math(text)

    def list_item(self, marker: str, text: str, level: int) -> None:
        e = self.e
        indent = (level - 1) * 2.2 * self.xh
        e._left = self.base_left + indent
        self.start_line()
        e._x = e._left
        e.write_word(marker, 1.0, wrap=False)
        hang = max(e._x, e._left + 1.6 * self.xh)
        e._x = hang
        e._left = hang                       # continuation lines hang under the text
        self.words_with_math(text)
        e._left = self.base_left

    def rule(self) -> None:
        e = self.e
        self.start_line()
        y = e.baseline_y() - 0.4 * self.xh
        e.hand_line(e._left, y, e._right, y)

    def code(self, text: str) -> None:
        e = self.e
        for line in text.rstrip("\n").split("\n"):
            self.start_line(0.95)
            lead = len(line) - len(line.lstrip(" "))
            e._x += lead * 0.55 * self.xh
            for w in line.split():
                e.write_word(w, 0.95)

    def display_math(self, latex: str) -> None:
        e = self.e
        for part in [p for p in re.split(r"\\\\", latex) if p.strip()]:
            box = self.math.layout(part, display=True)
            avail = e._right - e._left
            s = 1.0
            if box.w > avail:
                s = avail / box.w
                box = self.math.layout(part, display=True, scale=s)
            need = box.asc + box.desc + 0.6 * self.xh
            self.start_line()
            e.ensure_room(need)
            by = e._y + max(e._ascent, box.asc + 0.2 * self.xh)
            box.draw(e._left + (avail - box.w) / 2, by)
            e._y = by + box.desc - e._ascent
            e._x = e._right            # next block starts on a new line

    def table(self, rows: List[List[str]], header: bool) -> None:
        e = self.e
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        s = 0.92
        pad = 0.55 * self.xh
        avail = e._right - e._left
        nat = [max(self.measure_lines(c, 1e9, s)[1] for c in col) + 2 * pad for col in zip(*rows)]
        minw = 3.5 * self.xh
        if sum(nat) > avail:
            free = avail - minw * ncol
            extra = [max(0.0, n - minw) for n in nat]
            k = free / max(1.0, sum(extra))
            widths = [minw + x * k for x in extra]
        else:
            widths = nat
        laid = [[self.measure_lines(c, w - 2 * pad, s) for c, w in zip(r, widths)] for r in rows]
        heights = [max(h for _, _, h in r) + 2 * pad * 0.6 for r in laid]
        self.start_line()
        e._y -= 0.2 * self.lh()
        x0 = e._left
        W = sum(widths)
        def hline(y):
            e.hand_line(x0 - 0.15 * self.xh, y, x0 + W + 0.15 * self.xh, y, wobble=0.5)
        top = e._y + 0.3 * self.lh()
        e.ensure_room(heights[0] + (heights[1] if len(heights) > 1 else 0))
        top = max(top, e._y)
        seg_top = top
        hline(top)
        y = top
        for ri, (row, h) in enumerate(zip(laid, heights)):
            if y + h > e.cfg.page_height - e.cfg.margin_bottom:
                self._vlines(x0, widths, seg_top, y)
                e._page_break()
                y = seg_top = e._y + 0.3 * self.lh()
                hline(y)
            cx = x0
            for (lines, _, _), w in zip(row, widths):
                saved = (e._left, e._right)
                e._left, e._right = cx + pad, cx + w - pad
                for k, line in enumerate(lines):
                    e._x = e._left
                    e._y = y + pad * 0.6 + k * self.lh(s) + 0.72 * self.lh(s) - e._ascent * s
                    e._new_line_params()
                    e._line_slope *= 0.3
                    for word in line.split():
                        e.write_word(word, s, wrap=False)
                e._left, e._right = saved
                cx += w
            y += h
            hline(y)
            if ri == 0 and header:
                hline(y + 0.18 * self.xh)        # double rule under the header
        self._vlines(x0, widths, seg_top, y)
        e._y = y + 0.3 * self.lh() - self.lh() + 0.1 * self.lh()
        e._x = e._right

    def _vlines(self, x0: float, widths: List[float], top: float, bottom: float) -> None:
        x = x0
        for w in [0.0] + widths:
            x += w
            self.e.hand_line(x, top - 0.15 * self.xh, x + random.gauss(0, 0.1) * self.xh,
                             bottom + 0.15 * self.xh, wobble=0.5)

    def diagram(self, src: str) -> None:
        e = self.e
        fc = flowchart.parse(src)
        if not fc.nodes:
            self.code(src)
            return
        avail = e._right - e._left
        usable_h = e.cfg.page_height - e.cfg.margin_top - e.cfg.margin_bottom
        s = 0.85
        for _ in range(4):
            W, H = flowchart.layout(fc, lambda t, mw: self.measure_lines(t, mw, s),
                                    9.0 * self.xh * s, 2.6 * self.xh * s, 1.8 * self.xh * s)
            if W <= avail and H <= usable_h * 0.9:
                break
            s *= min(avail / W, usable_h * 0.9 / H, 0.95)
        self.start_line()
        e.ensure_room(H + self.lh())
        ox = e._left + (avail - W) / 2
        oy = e._y + 0.6 * self.lh()
        flowchart.draw(fc, e, ox, oy, self.write_lines_at, s)
        e._y = oy + H + 0.3 * self.lh() - e._ascent
        e._x = e._right

    def sketch(self, name: str, caption: str) -> None:
        e = self.e
        path = Path(e.cfg.glyph_bank or "") / "sketches" / f"{name}.json"
        if not e.cfg.glyph_bank or not path.exists():
            self.paragraph(f"[sketch {name}: draw it on the pad]")
            return
        sk = json.loads(path.read_text(encoding="utf-8"))
        strokes = sk["strokes"]
        xs = [p[0] for st in strokes for p in st]
        ys = [p[1] for st in strokes for p in st]
        if not xs:
            return
        w0, h0 = max(xs) - min(xs), max(ys) - min(ys)
        avail = e._right - e._left
        # sketches have no letter guides to size by: fit ~55 % of the line width,
        # at most 40 % of the page height
        usable_h = e.cfg.page_height - e.cfg.margin_top - e.cfg.margin_bottom
        k = min(0.55 * avail / max(1.0, w0), 0.4 * usable_h / max(1.0, h0))
        self.start_line()
        e.ensure_room(h0 * k + self.lh())
        ox = e._left + (avail - w0 * k) / 2 - min(xs) * k
        oy = e._y + 0.4 * self.lh() - min(ys) * k
        for st in strokes:
            if len(st) > 1:
                e.hand_path([(ox + p[0] * k, oy + p[1] * k) for p in st], wobble=0.25)
        e._y = oy + max(ys) * k + 0.2 * self.lh() - e._ascent
        e._x = e._right
        if caption:
            lines, w, _ = self.measure_lines(caption, avail, 0.9)
            self.start_line(0.9)
            e._x = e._left + max(0.0, (avail - w) / 2)
            e.write_text(caption, 0.9)
        self.gap(0.8)


def render_document(eng, text: str):
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(_prepare(text))
    eng.begin()
    d = DocWriter(eng)
    d.base_left = eng._left
    list_stack: List[List] = []              # [ordered, counter]
    quote = 0
    rows: List[List[str]] = []
    row: List[str] = []
    header = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        typ = t.type
        if typ == "heading_open":
            txt, _ = _inline_text(tokens[i + 1])
            d.heading(txt, int(t.tag[1]))
            i += 3
            continue
        if typ in ("bullet_list_open", "ordered_list_open"):
            list_stack.append([typ.startswith("ordered"), int(t.attrs.get("start", 1) or 1) - 1])
        elif typ in ("bullet_list_close", "ordered_list_close"):
            list_stack.pop()
            if not list_stack:
                d.gap(0.4)
        elif typ == "list_item_open":
            lst = list_stack[-1]
            lst[1] += 1
            marker = f"{lst[1]}." if lst[0] else "-"
            # first paragraph of the item
            j = i + 1
            while j < len(tokens) and tokens[j].type != "inline":
                j += 1
            txt, imgs = _inline_text(tokens[j]) if j < len(tokens) else ("", [])
            d.list_item(marker, txt, len(list_stack))
            i = j + 1
            continue
        elif typ == "blockquote_open":
            quote += 1
            eng._left = d.base_left = d.base_left + 2.0 * eng._x_height
        elif typ == "blockquote_close":
            quote -= 1
            eng._left = d.base_left = d.base_left - 2.0 * eng._x_height
        elif typ == "paragraph_open" and not list_stack:
            txt, imgs = _inline_text(tokens[i + 1])
            for src, cap in imgs:
                if src.startswith("pad:"):
                    d.sketch(src[4:], cap)
                else:
                    d.paragraph(f"[figure: {cap or src}]")
            if txt.strip():
                d.paragraph(txt)
                d.gap(0.5)
            i += 3
            continue
        elif typ in ("fence", "code_block"):
            lang = (t.info or "").strip().lower()
            if lang == "mermaid":
                d.diagram(t.content)
            elif lang in ("math", "latex", "tex"):
                d.display_math(t.content)
            else:
                d.code(t.content)
            d.gap(0.5)
        elif typ == "hr":
            d.rule()
        elif typ == "table_open":
            rows, header = [], False
        elif typ == "thead_open":
            header = True
        elif typ == "tr_open":
            row = []
        elif typ in ("th_open", "td_open"):
            txt, _ = _inline_text(tokens[i + 1])
            row.append(txt)
            i += 3
            continue
        elif typ == "tr_close":
            rows.append(row)
        elif typ == "table_close":
            d.table(rows, header)
            d.gap(0.6)
        i += 1
    return eng.finish()
