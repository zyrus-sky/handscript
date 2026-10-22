"""
PDF → Markdown for the document renderer, keeping the arrangement: headings (by
font size / bold), paragraphs (lines re-joined, hyphenation undone), bullet and
numbered lists, tables (PyMuPDF table finder), and figures. Figures (embedded
images or clusters of vector drawings) are either read by the local LLM
(→ mermaid / table / LaTeX, see llm.py) or left as ![..](pad:name) placeholders to
draw on the writing pad. Pages without a text layer (scans) are OCR'd by the LLM.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

import fitz  # PyMuPDF

BULLET = re.compile(r"^\s*([•●▪◦■□➢►\-–—*�])\s*")
NUMBERED = re.compile(r"^\s*(\(?\d{1,3}[.)]|[a-zA-Z][.)])\s+")


def _rects_overlap(a: fitz.Rect, b: fitz.Rect, pad: float = 0) -> bool:
    return not (a.x1 + pad < b.x0 or b.x1 + pad < a.x0 or a.y1 + pad < b.y0 or b.y1 + pad < a.y0)


def _drawing_clusters(page: fitz.Page, exclude: List[fitz.Rect]) -> List[fitz.Rect]:
    """Regions with several vector shapes (boxes, arrows): likely diagrams."""
    rects = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < 2 and r.height < 2:
            continue
        if any(_rects_overlap(r, t) for t in exclude):
            continue
        rects.append(fitz.Rect(r))
    clusters: List[Tuple[fitz.Rect, int]] = []
    for r in rects:
        for k, (c, n) in enumerate(clusters):
            if _rects_overlap(c, r, pad=12):
                clusters[k] = (c | r, n + 1)
                break
        else:
            clusters.append((r, 1))
    merged = True
    while merged:                                   # merge clusters that now touch
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _rects_overlap(clusters[i][0], clusters[j][0], pad=12):
                    clusters[i] = (clusters[i][0] | clusters[j][0], clusters[i][1] + clusters[j][1])
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break
    page_area = page.rect.width * page.rect.height
    return [c for c, n in clusters if n >= 4 and c.width * c.height > 0.01 * page_area
            and c.width < 0.98 * page.rect.width]


def _table_md(tab) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in tab.extract()]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def pdf_to_markdown(data: bytes, use_llm: bool = False, max_pages: int = 40) -> Tuple[str, List[str]]:
    """Returns (markdown, notes). notes tell the user what needs attention."""
    import llm
    doc = fitz.open(stream=data, filetype="pdf")
    notes: List[str] = []
    llm_ok = use_llm and llm.available()["ok"]
    if use_llm and not llm_ok:
        notes.append("Local AI not available: " + llm.available()["error"])

    # body text size = most common span size (by characters)
    sizes = Counter()
    for page in list(doc)[:max_pages]:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line["spans"]:
                    sizes[round(sp["size"], 1)] += len(sp["text"].strip())
    body = sizes.most_common(1)[0][0] if sizes else 11.0
    heading_sizes = sorted({s for s in sizes if s >= body * 1.18}, reverse=True)

    out: List[str] = []
    fig_n = 0
    for pno, page in enumerate(list(doc)[:max_pages], start=1):
        text_dict = page.get_text("dict")
        has_text = any(b.get("lines") for b in text_dict["blocks"])
        if not has_text:
            png = page.get_pixmap(dpi=150).tobytes("png")
            if llm_ok:
                out.append(llm.ocr_page(png))
            else:
                notes.append(f"Page {pno} is a scan with no text layer; enable the local AI to read it.")
            continue
        items: List[Tuple[float, str]] = []           # (y, markdown)
        try:
            tables = list(page.find_tables().tables)
        except Exception:
            tables = []
        # boxes of a diagram can look like a table: mostly-empty "tables" are figures
        real, boxy = [], []
        for t in tables:
            cells = [c for r in t.extract() for c in r]
            empty = sum(1 for c in cells if not (c or "").strip()) / max(1, len(cells))
            (boxy if empty > 0.35 else real).append(t)
        tables = real
        t_rects = [fitz.Rect(t.bbox) for t in tables]
        for t, r in zip(tables, t_rects):
            md = _table_md(t)
            if md:
                items.append((r.y0, md))
        figures = [fitz.Rect(b["bbox"]) for b in text_dict["blocks"] if b.get("type") == 1]
        figures += [fitz.Rect(t.bbox) for t in boxy]
        figures += _drawing_clusters(page, t_rects)
        merged: List[fitz.Rect] = []               # one region per figure
        for r in sorted(figures, key=lambda r: -r.width * r.height):
            for k, m in enumerate(merged):
                if _rects_overlap(m, r, pad=6):
                    merged[k] = m | r
                    break
            else:
                merged.append(fitz.Rect(r))
        figures = merged
        for r in figures:
            if r.width < 40 or r.height < 30:
                continue
            fig_n += 1
            name = f"fig{fig_n}"
            if llm_ok:
                png = page.get_pixmap(dpi=170, clip=r).tobytes("png")
                md = llm.figure_to_markdown(png)
                if md.startswith("SKETCH:"):
                    md = f"![{md[7:].strip()}](pad:{name})"
                    notes.append(f"Figure {name} (page {pno}): draw it on the pad as sketch '{name}'.")
            else:
                md = f"![figure {fig_n}, page {pno}](pad:{name})"
                notes.append(f"Figure {name} (page {pno}): draw it on the pad as sketch '{name}', "
                             f"or enable the local AI to convert diagrams.")
            items.append((r.y0, md))

        for b in text_dict["blocks"]:
            if b.get("type") != 0:
                continue
            br = fitz.Rect(b["bbox"])
            if any(_rects_overlap(br, t) for t in t_rects) or any(br in f for f in figures):
                continue
            lines = []
            for line in b["lines"]:
                txt = "".join(sp["text"] for sp in line["spans"]).strip()
                if not txt:
                    continue
                size = max(sp["size"] for sp in line["spans"])
                bold = all(sp["flags"] & 16 for sp in line["spans"] if sp["text"].strip())
                lines.append((txt, size, bold))
            if not lines:
                continue
            first, size, bold = lines[0]
            if round(size, 1) in heading_sizes and len(lines) <= 2:
                level = min(3, heading_sizes.index(round(size, 1)) + 1)
                items.append((br.y0, "#" * level + " " + " ".join(l[0] for l in lines)))
                continue
            if bold and len(lines) == 1 and len(first) < 70 and not first.endswith("."):
                items.append((br.y0, "### " + first))
                continue
            md_lines, para = [], ""
            for txt, _, _ in lines:
                if BULLET.match(txt) or NUMBERED.match(txt):
                    if para:
                        md_lines.append(para)
                    para = BULLET.sub("- ", txt) if BULLET.match(txt) else re.sub(r"^\s*\(?(\d+)[.)]\s+", r"\1. ", txt)
                elif para.endswith("-") and not para.endswith(" -"):
                    para = para[:-1] + txt                       # undo hyphenation
                else:
                    para = f"{para} {txt}" if para else txt
            if para:
                md_lines.append(para)
            items.append((br.y0, "\n".join(md_lines)))
        items.sort(key=lambda it: it[0])
        out.append("\n\n".join(md for _, md in items))
    if len(doc) > max_pages:
        notes.append(f"Only the first {max_pages} of {len(doc)} pages were imported.")
    return "\n\n".join(o for o in out if o.strip()), notes
