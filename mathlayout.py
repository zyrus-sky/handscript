"""
Handwritten math: a small LaTeX subset laid out as boxes and written with the
engine's glyphs (the writer's own pieces where the bank has them, a math font
for the rest) plus hand-drawn fraction bars and root signs.

Supported: letters, digits, operators, ^ _ (any order), {groups}, \\frac, \\dfrac,
\\sqrt[n]{..}, \\sum \\prod \\int \\oint \\lim (limits above/below in display mode),
\\left( \\right) (delimiters written as-is), \\text{} \\mathrm{} \\mathbf{} \\operatorname{},
Greek letters, common relations/arrows/operators, \\sin \\cos \\log ... as upright text,
\\, \\; \\quad spacing. Unknown commands are written as their name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from PIL import ImageFont

SYMBOLS = {
    # Greek
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    # operators / relations / arrows
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "ast": "∗",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "approx": "≈",
    "equiv": "≡", "sim": "∼", "propto": "∝", "ll": "≪", "gg": "≫",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "leftrightarrow": "↔", "Leftrightarrow": "⇔", "mapsto": "↦", "implies": "⇒", "iff": "⇔",
    "infty": "∞", "partial": "∂", "nabla": "∇", "forall": "∀", "exists": "∃", "neg": "¬",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "cup": "∪",
    "cap": "∩", "emptyset": "∅", "varnothing": "∅", "land": "∧", "lor": "∨", "angle": "∠",
    "degree": "°", "circ": "∘", "prime": "′", "ldots": "…", "cdots": "⋯", "dots": "…",
    "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "{": "{", "}": "}", "%": "%", "$": "$", "#": "#", "&": "&", "_": "_", "|": "‖",
}
BIG_OPS = {"sum": "∑", "prod": "∏", "int": "∫", "oint": "∮", "iint": "∬", "bigcup": "⋃", "bigcap": "⋂"}
FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "max", "min",
         "det", "arg", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "mod", "gcd", "sup", "inf"}
SPACES = {",": 0.17, ";": 0.28, ":": 0.22, "quad": 1.0, "qquad": 2.0, "!": -0.1, " ": 0.3}
BINARY = set("+-=<>×·÷±∓≤≥≠≈≡∼∝→←⇒⇐↔⇔∈∉⊂⊆⊃∪∩∧∨") | {"−"}
TALL = set("bdfhklt0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ()[]{}|/\\∑∏∫∮√∂∇ΓΔΘΛΞΠΣΦΨΩβδζθλξ!?'\"")
DESC = set("gjpqyβζημξρφχψ()[]{}|,;∫∮")

_FONT_CANDIDATES = [r"C:\Windows\Fonts\cambria.ttc", r"C:\Windows\Fonts\seguisym.ttf",
                    r"C:\Windows\Fonts\segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


# ── parsing ───────────────────────────────────────────────────────────────

def _tokens(src: str) -> List[str]:
    out, i = [], 0
    while i < len(src):
        c = src[i]
        if c == "\\":
            j = i + 1
            if j < len(src) and src[j].isalpha():
                while j < len(src) and src[j].isalpha():
                    j += 1
            else:
                j += 1
            out.append(src[i:j])
            i = j
        elif c.isspace():
            i += 1
        else:
            out.append(c)
            i += 1
    return out


@dataclass
class Node:
    kind: str                      # char | row | frac | sqrt | script | bigop | text | space
    text: str = ""
    kids: List["Node"] = field(default_factory=list)
    sup: Optional["Node"] = None
    sub: Optional["Node"] = None
    upright: bool = False


class _Parser:
    def __init__(self, src: str):
        self.t = _tokens(src)
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self) -> Optional[str]:
        tok = self.peek()
        self.i += 1
        return tok

    def row(self, stop: Tuple[str, ...] = ()) -> Node:
        kids = []
        while self.peek() is not None and self.peek() not in stop:
            a = self.atom()
            if a is None:
                continue
            while self.peek() in ("^", "_"):
                op = self.next()
                arg = self.arg()
                if a.kind not in ("script", "bigop"):
                    a = Node("script", kids=[a])
                if op == "^":
                    a.sup = arg
                else:
                    a.sub = arg
            kids.append(a)
        return Node("row", kids=kids)

    def arg(self) -> Node:
        if self.peek() == "{":
            self.next()
            r = self.row(("}",))
            self.next()
            return r
        a = self.atom()
        return a if a is not None else Node("row")

    def raw_group(self) -> str:
        if self.peek() != "{":
            return self.next() or ""
        self.next()
        depth, parts = 1, []
        while self.peek() is not None:
            tok = self.next()
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth -= 1
                if depth == 0:
                    break
            parts.append(tok)
        out = ""
        for p in parts:                              # \text keeps its spaces only between words
            out += (" " if out and p[0].isalnum() and out[-1].isalnum() and len(p) > 1 else "") + p
        return out

    def atom(self) -> Optional[Node]:
        tok = self.next()
        if tok is None:
            return None
        if tok == "{":
            r = self.row(("}",))
            self.next()
            return r
        if tok in ("}", "&"):
            return None
        if tok == "\\\\":
            return Node("space", text="1.0")
        if not tok.startswith("\\"):
            return Node("char", text=tok)
        name = tok[1:]
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            return Node("frac", kids=[self.arg(), self.arg()])
        if name == "sqrt":
            idx = None
            if self.peek() == "[":
                self.next()
                idx = self.row(("]",))
                self.next()
            n = Node("sqrt", kids=[self.arg()])
            n.sup = idx
            return n
        if name in ("text", "mathrm", "textrm", "operatorname", "mathbf", "textbf", "mathit", "mbox"):
            return Node("text", text=self.raw_group(), upright=True)
        if name in ("left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr"):
            nxt = self.next() or ""
            if nxt == ".":
                return None
            return Node("char", text=SYMBOLS.get(nxt[1:], nxt[1:]) if nxt.startswith("\\") else nxt)
        if name in BIG_OPS:
            return Node("bigop", text=BIG_OPS[name])
        if name in FUNCS:
            return Node("text", text=name, upright=True)
        if name in SPACES:
            return Node("space", text=str(SPACES[name]))
        if name in ("mathbb", "mathcal", "mathfrak", "vec", "hat", "bar", "overline", "tilde", "dot"):
            inner = self.arg()
            return inner
        if name in SYMBOLS:
            return Node("char", text=SYMBOLS[name])
        return Node("text", text=name, upright=True)


def parse(src: str) -> Node:
    return _Parser(src.strip()).row()


# ── layout + drawing ──────────────────────────────────────────────────────

@dataclass
class Box:
    w: float
    asc: float
    desc: float
    draw: Callable[[float, float], None]      # draw(x, base_y)


class MathWriter:
    """Lays out a parsed formula against an engine and draws it."""

    def __init__(self, eng):
        self.e = eng
        self.xh = eng._x_height * eng._page_scale
        self._font = None

    def font(self) -> ImageFont.FreeTypeFont:
        if self._font is None:
            size = int(self.e.cfg.font_size)
            for path in _FONT_CANDIDATES:
                if os.path.exists(path):
                    try:
                        self._font = ImageFont.truetype(path, size)
                        break
                    except OSError:
                        continue
            if self._font is None:
                self._font = self.e._random_font()
        return self._font

    # one character
    def char(self, ch: str, s: float, upright: bool = False) -> Box:
        e, xh = self.e, self.xh
        use_bank = e.bank is not None and ch in e.bank
        font = None if (use_bank or (ch.isascii() and ch.isalnum())) else self.font()
        if use_bank:
            g = e.bank._meta[ch][0]
            w = (g["ink_right"] - g["ink_left"]) * e._x_height / (g.get("ref_xh") or e.bank.ref_xh) \
                * g.get("size_fix", 1.0) * s * e._page_scale
        else:
            f = font or e._random_font()
            bb = f.getbbox(ch)
            w = max(1.0, bb[2] - bb[0]) * s * e._page_scale
        asc = xh * s * (1.8 if ch in TALL else 1.05)
        desc = xh * s * (0.55 if ch in DESC else 0.05)
        pad = 0.08 * xh * s
        def draw(x, by, ch=ch, s=s, font=font, pad=pad):
            e.glyph(ch, x + pad, by, s, font=font)
        return Box(w + 2 * pad, asc, desc, draw)

    def row(self, boxes: List[Box]) -> Box:
        if not boxes:
            return Box(0.0, 0.0, 0.0, lambda x, y: None)
        w = sum(b.w for b in boxes)
        asc = max(b.asc for b in boxes)
        desc = max(b.desc for b in boxes)
        def draw(x, by):
            for b in boxes:
                b.draw(x, by)
                x += b.w
        return Box(w, asc, desc, draw)

    def space(self, em: float, s: float) -> Box:
        return Box(em * 1.6 * self.xh * s, 0.0, 0.0, lambda x, y: None)

    def build(self, n: Node, s: float, display: bool) -> Box:
        xh = self.xh
        if n.kind == "char":
            b = self.char(n.text, s)
            if n.text in BINARY:
                sp = self.space(0.16, s)
                return self.row([sp, b, sp])
            return b
        if n.kind == "text":
            parts = [self.char(c, s, upright=True) if c != " " else self.space(0.25, s) for c in n.text]
            return self.row(parts + ([self.space(0.1, s)] if n.upright else []))
        if n.kind == "space":
            return self.space(float(n.text), s)
        if n.kind == "row":
            return self.row([self.build(k, s, display) for k in n.kids])
        if n.kind == "frac":
            ks = s if display and s >= 1.0 else s * 0.8
            num, den = self.build(n.kids[0], ks, False), self.build(n.kids[1], ks, False)
            gap, axis = 0.25 * xh * s, 0.55 * xh * s
            w = max(num.w, den.w) + 0.5 * xh * s
            def draw(x, by, num=num, den=den, w=w, gap=gap, axis=axis):
                ay = by - axis
                self.e.hand_line(x + 0.1 * xh, ay, x + w - 0.1 * xh, ay, wobble=0.5)
                num.draw(x + (w - num.w) / 2, ay - gap - num.desc)
                den.draw(x + (w - den.w) / 2, ay + gap + den.asc)
            return Box(w, axis + gap + num.asc + num.desc, den.asc + den.desc + gap - axis, draw)
        if n.kind == "sqrt":
            body = self.build(n.kids[0], s, display)
            idx = self.build(n.sup, s * 0.55, False) if n.sup is not None else None
            hook = 0.9 * xh * s
            top = body.asc + 0.25 * xh * s
            def draw(x, by, body=body, idx=idx, hook=hook, top=top):
                x0 = x + (idx.w * 0.6 if idx else 0)
                self.e.hand_path([(x0, by - 0.45 * xh * s), (x0 + 0.25 * hook, by - 0.6 * xh * s),
                                  (x0 + 0.55 * hook, by + body.desc), (x0 + hook, by - top),
                                  (x0 + hook + body.w + 0.15 * xh * s, by - top)], wobble=0.4)
                if idx:
                    idx.draw(x, by - 0.75 * xh * s)
                body.draw(x0 + hook + 0.05 * xh * s, by)
            return Box(body.w + hook + 0.25 * xh * s + (idx.w * 0.6 if idx else 0), top + 0.1 * xh * s, body.desc, draw)
        if n.kind == "bigop":
            op = self.char(n.text, s * (1.6 if display else 1.15))
            sup = self.build(n.sup, s * 0.65, False) if n.sup is not None else None
            sub = self.build(n.sub, s * 0.65, False) if n.sub is not None else None
            limits = display and n.text != "∫" and n.text != "∮"
            if limits:
                w = max(op.w, sup.w if sup else 0, sub.w if sub else 0) + 0.2 * xh * s
                def draw(x, by, op=op, sup=sup, sub=sub, w=w):
                    op.draw(x + (w - op.w) / 2, by + 0.25 * xh * s)
                    if sup:
                        sup.draw(x + (w - sup.w) / 2, by - op.asc - sup.desc - 0.05 * xh * s)
                    if sub:
                        sub.draw(x + (w - sub.w) / 2, by + op.desc + 0.25 * xh * s + sub.asc)
                return Box(w, op.asc + (sup.asc + sup.desc if sup else 0),
                           op.desc + 0.25 * xh * s + (sub.asc + sub.desc if sub else 0), draw)
            n2 = Node("script", kids=[Node("row")], sup=n.sup, sub=n.sub)
            return self.row([op, self.build(n2, s, False), self.space(0.1, s)])
        if n.kind == "script":
            base = self.build(n.kids[0], s, display)
            sup = self.build(n.sup, s * 0.65, False) if n.sup is not None else None
            sub = self.build(n.sub, s * 0.65, False) if n.sub is not None else None
            up = max(0.85 * xh * s, base.asc - 0.35 * xh * s)
            down = 0.35 * xh * s + (base.desc * 0.5)
            w = base.w + max(sup.w if sup else 0, sub.w if sub else 0) + 0.05 * xh * s
            def draw(x, by, base=base, sup=sup, sub=sub, up=up, down=down):
                base.draw(x, by)
                if sup:
                    sup.draw(x + base.w, by - up)
                if sub:
                    sub.draw(x + base.w, by + down)
            asc = max(base.asc, (up + sup.asc) if sup else 0)
            desc = max(base.desc, (down + sub.desc) if sub else 0)
            return Box(w, asc, desc, draw)
        return Box(0, 0, 0, lambda x, y: None)

    def layout(self, latex: str, display: bool, scale: float = 1.0) -> Box:
        return self.build(parse(latex), scale, display)
