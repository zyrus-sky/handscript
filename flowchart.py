"""
Hand-drawn flowcharts from Mermaid `flowchart` / `graph` text.

Supported: direction TD/TB/BT/LR/RL; nodes A, A[text], A(text), A([text]), A((text)),
A{text}, A[[text]], A[(text)], A>text]; edges --> --- -.-> ==> with optional labels
(-->|label| or -- label -->); chains (A --> B --> C); `A & B --> C`; `%%` comments;
subgraph/end/style/class lines are ignored. Layout: ranks by longest path (cycles
broken), order within a rank by barycenter sweeps, then boxes and arrows are drawn
with the engine's hand strokes and labels written in the writer's hand.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SHAPES = [  # (open, close, shape) — longest first
    ("([", "])", "stadium"), ("((", "))", "circle"), ("[[", "]]", "rect"), ("[(", ")]", "rect"),
    ("{{", "}}", "hex"), ("[/", "/]", "rect"), ("[\\", "\\]", "rect"),
    ("[", "]", "rect"), ("(", ")", "round"), ("{", "}", "diamond"), (">", "]", "rect"),
]
EDGE_RE = re.compile(r"\s*(<?(?:-->|---|-\.->|-\.-|==>|===|--o|--x)|<?--\s*[^-<>|]+?\s*-->|<?==\s*[^=<>|]+?\s*==>)\s*(?:\|([^|]*)\|)?\s*")


@dataclass
class FNode:
    id: str
    text: str
    shape: str = "rect"
    rank: int = 0
    order: float = 0.0
    lines: List[str] = field(default_factory=list)
    w: float = 0.0
    h: float = 0.0
    x: float = 0.0   # centre
    y: float = 0.0


@dataclass
class FEdge:
    a: str
    b: str
    label: str = ""
    arrow: bool = True
    dashed: bool = False


@dataclass
class Flowchart:
    direction: str = "TD"
    nodes: Dict[str, FNode] = field(default_factory=dict)
    edges: List[FEdge] = field(default_factory=list)


def _clean(text: str) -> str:
    text = text.strip().strip('"').replace("<br>", " ").replace("<br/>", " ").replace("\\n", " ")
    return re.sub(r"\s+", " ", text)


def _node_ref(src: str, fc: Flowchart) -> Tuple[Optional[str], str]:
    """Parse one node reference at the start of src; returns (id, rest)."""
    m = re.match(r"\s*([A-Za-z0-9_\-\.]+)", src)
    if not m:
        return None, src
    nid, rest = m.group(1), src[m.end():]
    for o, c, shape in SHAPES:
        if rest.startswith(o):
            end = rest.find(c, len(o))
            if end < 0:
                continue
            text = _clean(rest[len(o):end])
            fc.nodes[nid] = FNode(nid, text or nid, shape)
            return nid, rest[end + len(c):]
    if nid not in fc.nodes:
        fc.nodes[nid] = FNode(nid, nid, "rect")
    return nid, rest


def parse(src: str) -> Flowchart:
    fc = Flowchart()
    for raw in src.splitlines():
        line = raw.split("%%")[0].strip().rstrip(";")
        if not line:
            continue
        head = re.match(r"(flowchart|graph)\s+(TD|TB|BT|LR|RL)\b", line, re.I)
        if head:
            fc.direction = head.group(2).upper().replace("TB", "TD")
            continue
        if re.match(r"(subgraph|end|style|classDef|class|linkStyle|click|direction)\b", line):
            continue
        # A & B --> C & D --> E
        groups, rest = [], line
        while rest:
            ids = []
            while True:
                nid, rest = _node_ref(rest, fc)
                if nid is None:
                    break
                ids.append(nid)
                amp = re.match(r"\s*&\s*", rest)
                if not amp:
                    break
                rest = rest[amp.end():]
            if not ids:
                break
            groups.append((ids, None))
            m = EDGE_RE.match(rest)
            if not m:
                break
            op, label = m.group(1), m.group(2) or ""
            inline = re.match(r"<?(?:--|==)\s*(.+?)\s*(?:-->|==>)", op)
            if inline and not label:
                label = inline.group(1)
            groups[-1] = (ids, (op, _clean(label)))
            rest = rest[m.end():]
        for (src_ids, edge), (dst_ids, _) in zip(groups, groups[1:]):
            if edge is None:
                continue
            op, label = edge
            for a in src_ids:
                for b in dst_ids:
                    fc.edges.append(FEdge(a, b, label, arrow=">" in op or "o" in op[-2:] or "x" in op[-2:],
                                          dashed="." in op))
    return fc


def layout(fc: Flowchart, measure, max_text_w: float, gap_main: float, gap_cross: float) -> Tuple[float, float]:
    """Assign sizes (via measure(text) -> (lines, w, h)) and positions. Returns (W, H)
    of the drawing in the engine's px, origin top-left."""
    ids = list(fc.nodes)
    succ = {i: [] for i in ids}
    for e in fc.edges:
        succ[e.a].append(e.b)
    # break cycles with a DFS: back edges are ignored for ranking
    state, back = {}, set()
    def dfs(u):
        state[u] = 1
        for v in succ[u]:
            if state.get(v) == 1:
                back.add((u, v))
            elif v not in state:
                dfs(v)
        state[u] = 2
    for i in ids:
        if i not in state:
            dfs(i)
    preds = {i: [] for i in ids}
    for e in fc.edges:
        if (e.a, e.b) not in back:
            preds[e.b].append(e.a)
    rank: Dict[str, int] = {}
    def rk(u, seen=()):
        if u in rank:
            return rank[u]
        rank[u] = 0 if not preds[u] else 1 + max(rk(p) for p in preds[u])
        return rank[u]
    for i in ids:
        rk(i)
    layers: Dict[int, List[str]] = {}
    for i in ids:
        fc.nodes[i].rank = rank[i]
        layers.setdefault(rank[i], []).append(i)
    for r, lst in layers.items():
        for k, i in enumerate(lst):
            fc.nodes[i].order = k
    for _ in range(4):                                   # barycenter sweeps
        for r in sorted(layers)[1:]:
            for i in layers[r]:
                ps = [fc.nodes[p].order for p in preds[i]]
                if ps:
                    fc.nodes[i].order = sum(ps) / len(ps)
            layers[r].sort(key=lambda i: fc.nodes[i].order)
            for k, i in enumerate(layers[r]):
                fc.nodes[i].order = k

    for n in fc.nodes.values():
        n.lines, n.w, n.h = measure(n.text, max_text_w)
        pad_w, pad_h = (1.8, 1.4) if n.shape == "diamond" else (1.15, 1.0)
        if n.shape == "circle":
            d = max(n.w, n.h) * 1.25
            n.w = n.h = d
        else:
            n.w *= pad_w
            n.h *= pad_h
    horizontal = fc.direction in ("LR", "RL")
    # main axis = rank direction
    main_sizes = {r: max((fc.nodes[i].w if horizontal else fc.nodes[i].h) for i in lst) for r, lst in layers.items()}
    pos_main, cur = {}, 0.0
    for r in sorted(layers):
        pos_main[r] = cur + main_sizes[r] / 2
        cur += main_sizes[r] + gap_main
    total_main = cur - gap_main
    cross_tot = {}
    for r, lst in layers.items():
        cross_tot[r] = sum((fc.nodes[i].h if horizontal else fc.nodes[i].w) for i in lst) + gap_cross * (len(lst) - 1)
    total_cross = max(cross_tot.values())
    for r, lst in layers.items():
        c = (total_cross - cross_tot[r]) / 2
        for i in lst:
            n = fc.nodes[i]
            size = n.h if horizontal else n.w
            cross = c + size / 2
            c += size + gap_cross
            m = pos_main[r]
            if fc.direction in ("BT", "RL"):
                m = total_main - m
            n.x, n.y = (m, cross) if horizontal else (cross, m)
    return (total_main, total_cross) if horizontal else (total_cross, total_main)


def _border_point(n: FNode, tx: float, ty: float) -> Tuple[float, float]:
    """Where the line from n's centre towards (tx, ty) leaves n's outline."""
    dx, dy = tx - n.x, ty - n.y
    if dx == 0 and dy == 0:
        return n.x, n.y
    hw, hh = n.w / 2, n.h / 2
    if n.shape == "circle":
        L = math.hypot(dx, dy)
        return n.x + dx / L * hw, n.y + dy / L * hh
    if n.shape == "diamond":
        t = 1.0 / (abs(dx) / hw + abs(dy) / hh)
        return n.x + dx * t, n.y + dy * t
    t = min(hw / abs(dx) if dx else 1e9, hh / abs(dy) if dy else 1e9)
    return n.x + dx * t, n.y + dy * t


def draw(fc: Flowchart, eng, ox: float, oy: float, write_lines, scale: float) -> None:
    """Draw shapes, edges and labels with the engine, offset by (ox, oy)."""
    for n in fc.nodes.values():
        x, y, hw, hh = ox + n.x, oy + n.y, n.w / 2, n.h / 2
        if n.shape == "diamond":
            eng.hand_path([(x, y - hh), (x + hw, y), (x, y + hh), (x - hw, y)], closed=True)
        elif n.shape == "circle":
            pts = [(x + hw * math.cos(t / 24 * 2 * math.pi), y + hh * math.sin(t / 24 * 2 * math.pi)) for t in range(24)]
            eng.hand_path(pts, closed=True)
        elif n.shape in ("round", "stadium"):
            r = min(hh, hw) * (0.9 if n.shape == "stadium" else 0.45)
            pts = []
            for cx, cy, a0 in ((x + hw - r, y - hh + r, -90), (x + hw - r, y + hh - r, 0),
                               (x - hw + r, y + hh - r, 90), (x - hw + r, y - hh + r, 180)):
                pts += [(cx + r * math.cos(math.radians(a0 + k * 15)), cy + r * math.sin(math.radians(a0 + k * 15)))
                        for k in range(7)]
            eng.hand_path(pts, closed=True)
        elif n.shape == "hex":
            k = min(hw * 0.3, hh)
            eng.hand_path([(x - hw + k, y - hh), (x + hw - k, y - hh), (x + hw, y), (x + hw - k, y + hh),
                           (x - hw + k, y + hh), (x - hw, y)], closed=True)
        else:
            eng.hand_path([(x - hw, y - hh), (x + hw, y - hh), (x + hw, y + hh), (x - hw, y + hh)], closed=True)
        write_lines(n.lines, x, y, scale)
    for e in fc.edges:
        a, b = fc.nodes[e.a], fc.nodes[e.b]
        ax, ay = _border_point(a, b.x, b.y)
        bx, by = _border_point(b, a.x, a.y)
        ax, ay, bx, by = ax + ox, ay + oy, bx + ox, by + oy
        if e.dashed:
            L = math.hypot(bx - ax, by - ay)
            steps = max(2, int(L / (0.9 * eng._x_height)))
            for k in range(0, steps, 2):
                t0, t1 = k / steps, min(1.0, (k + 1) / steps)
                eng.hand_line(ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1)
            if e.arrow:
                eng.hand_arrow(bx - (bx - ax) * 0.02, by - (by - ay) * 0.02, bx, by)
        elif e.arrow:
            eng.hand_arrow(ax, ay, bx, by)
        else:
            eng.hand_line(ax, ay, bx, by)
        if e.label:
            mx, my = (ax + bx) / 2, (ay + by) / 2
            write_lines([e.label], mx + 0.6 * eng._x_height, my, scale * 0.8, centre=False)
