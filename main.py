"""
CLI entry point for the Handwritten Text Rendering Engine.

Same settings and output as the web generator (genconfig.py). Input is Markdown or
plain text (headings, lists, tables, ```mermaid flowcharts, $math$, ![](pad:name)
sketches - see docrender.py), from --text, --file, --pdf or stdin (--file -).

  python main.py --file notes.md -o notes.pdf --ink blue --preview previews/
"""

import argparse
import json
import sys
from pathlib import Path

import genconfig
from engine import HandwritingEngine


def _ink(value: str):
    if value.lower() in genconfig.INKS:
        return genconfig.INKS[value.lower()]
    v = value.lstrip("#")
    if len(v) == 6:
        return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))
    parts = [int(x) for x in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ink: blue, black, #RRGGBB or R,G,B")
    return tuple(parts)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Generate handwritten PDFs from Markdown or plain text.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Text / Markdown to render (\\n for new lines).")
    src.add_argument("--file", type=str, help="A .md/.txt file to render ('-' reads stdin).")
    src.add_argument("--pdf", type=str, help="A PDF to copy out by hand (text, tables; figures become placeholders).")

    parser.add_argument("-o", "--output", type=str, default="output.pdf", help="Output PDF path.")
    parser.add_argument("--bank", type=str, default="mine",
                        help="Handwriting bank in banks/ (default: mine); 'none' for fonts only.")
    parser.add_argument("--fonts", nargs="+", help="Fonts for letters the bank lacks (default: bundled).")
    parser.add_argument("--ink", type=_ink, default=genconfig.INKS["black"],
                        help="blue (ballpoint), black, #RRGGBB or R,G,B (default: black).")
    parser.add_argument("--texture", type=float, default=0.4, help="Ballpoint ink texture 0-1 (default 0.4).")
    parser.add_argument("--show-through", type=float, nargs="?", const=0.5, default=None,
                        help="Next page faintly visible through the sheet (strength, default 0.5).")
    parser.add_argument("--dpi", type=int, default=300, choices=(300, 450, 600))
    parser.add_argument("--line-spacing", type=float, default=1.0)
    parser.add_argument("--letter-width", type=float, default=1.0)
    parser.add_argument("--proofread", action="store_true",
                        help="Reading check: rewrite words the handwriting reader misreads (slower).")
    parser.add_argument("--scan-look", action="store_true", help="Look scanned/photographed (not for printing).")
    parser.add_argument("--model", action="store_true", help="Generate with the Emuru handwriting model (GPU).")
    parser.add_argument("--llm", action="store_true", help="With --pdf: let the local AI describe figures.")
    parser.add_argument("--preview", type=str, help="Also write small PNG previews of each page to this folder.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args()

    notes = []
    if args.text:
        text = args.text.replace("\\n", "\n")
    elif args.pdf:
        import pdfimport
        path = Path(args.pdf)
        if not path.exists():
            sys.exit(f"Error: file not found: {args.pdf}")
        text, notes = pdfimport.pdf_to_markdown(path.read_bytes(), use_llm=args.llm)
    elif args.file == "-":
        text = sys.stdin.buffer.read().decode("utf-8")
    else:
        path = Path(args.file)
        if not path.exists():
            sys.exit(f"Error: file not found: {args.file}")
        text = path.read_text(encoding="utf-8")
    if not text.strip():
        sys.exit("Error: nothing to write")

    params = {
        "glyph_bank": None if args.bank.lower() == "none" else args.bank,
        "ink_color": list(args.ink),
        "ink_texture": args.texture,
        "show_through": args.show_through is not None,
        "show_through_strength": args.show_through or 0.5,
        "dpi": args.dpi,
        "line_spacing": args.line_spacing,
        "letter_width": args.letter_width,
        "proofread": args.proofread,
        "scan_look": args.scan_look,
        "handwriting_model": args.model,
    }
    cfg = genconfig.config_from_params(params)
    if params["glyph_bank"] and cfg.glyph_bank is None:
        sys.exit(f"Error: no built handwriting bank banks/{args.bank}")
    engine = HandwritingEngine(genconfig.font_paths(cfg, args.fonts), cfg)
    engine.render(text)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    engine.save_pdf(str(out))

    previews = []
    if args.preview:
        from PIL import Image
        d = Path(args.preview)
        d.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(engine.pages):
            rgb = Image.new("RGB", page.size, "white")
            rgb.paste(page, mask=page.split()[3] if page.mode == "RGBA" else None)
            rgb.thumbnail((1100, 1556), Image.LANCZOS)
            p = d / f"{out.stem}_p{i + 1}.png"
            rgb.save(p)
            previews.append(str(p))

    summary = {
        "output": str(out.resolve()),
        "pages": len(engine.pages),
        # characters written with a font because the bank has no piece for them
        "font_letters": [c for c, _ in engine.fallback_used.most_common(30) if c.strip()],
        # the same word image used several times on a page (write more copies on the pad)
        "repeats": [f"{r['word']} x{r['repeats']} (page {r['page']})" for r in engine.repeat_report()
                    if len(r['word']) >= 3][:10],
        "hard_to_read": [w for w, _, _ in getattr(engine, "proof_log", [])][:20],
        "notes": notes,
        "previews": previews,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    else:
        print(f"✓ Generated {summary['pages']} page(s) → {summary['output']}")
        for k in ("font_letters", "repeats", "hard_to_read", "notes", "previews"):
            if summary[k]:
                print(f"  {k.replace('_', ' ')}: {', '.join(map(str, summary[k]))}")


if __name__ == "__main__":
    main()
