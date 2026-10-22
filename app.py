"""
Flask backend for HandScript — serves the Stitch-generated UI
and provides API endpoints for the handwriting rendering engine.
"""

import io
import threading
import json
import os
import re
import base64
import uuid
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image

from engine import HandwritingEngine, EngineConfig, pages_to_pdf
import genconfig
import glyphbank
import wordmodel

app = Flask(__name__, static_folder="ui")
CORS(app)

# ── Globals ──────────────────────────────────────────────────────────────

UPLOAD_DIR = Path(tempfile.gettempdir()) / "handscript_fonts"
UPLOAD_DIR.mkdir(exist_ok=True)

# Default bundled fonts
BUNDLED_FONTS_DIR = Path(__file__).parent / "fonts"

# Glyph banks (letters cut from the user's scans): banks/<name>/
BANKS_DIR = Path(__file__).parent / "banks"

# Store generated pages per session (simple in-memory)
sessions: dict = {}


# ── Routes: Static UI ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


@app.route("/ui/<path:filename>")
def serve_ui(filename):
    return send_from_directory("ui", filename)


# ── Routes: Font management ──────────────────────────────────────────────

@app.route("/api/fonts/upload", methods=["POST"])
def upload_font():
    """Upload a .ttf/.otf font file."""
    if "font" not in request.files:
        return jsonify({"error": "No font file provided"}), 400

    f = request.files["font"]
    if not f.filename.lower().endswith((".ttf", ".otf")):
        return jsonify({"error": "Only .ttf and .otf files are supported"}), 400

    safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
    save_path = UPLOAD_DIR / safe_name
    f.save(str(save_path))

    return jsonify({
        "id": safe_name,
        "name": f.filename,
        "path": str(save_path),
    })


@app.route("/api/fonts/bundled", methods=["GET"])
def list_bundled_fonts():
    """List pre-bundled font files."""
    fonts = []
    if BUNDLED_FONTS_DIR.exists():
        for fp in BUNDLED_FONTS_DIR.glob("*"):
            if fp.suffix.lower() in (".ttf", ".otf"):
                fonts.append({"name": fp.name, "path": str(fp)})
    return jsonify(fonts)


# ── Routes: Generation ───────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate handwritten pages from text + config."""
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is required"}), 400

    # Font paths
    font_paths = data.get("font_paths", [])
    if not font_paths:
        # Fallback to bundled fonts
        font_paths = [str(p) for p in BUNDLED_FONTS_DIR.glob("*.ttf")]
        font_paths += [str(p) for p in BUNDLED_FONTS_DIR.glob("*.otf")]
    if not font_paths:
        return jsonify({"error": "No fonts available"}), 400

    # Build config from params (genconfig: shared with the command line, main.py)
    params = data.get("config", {})
    cfg = genconfig.config_from_params(params)
    font_paths = genconfig.font_paths(cfg, font_paths)

    # Generate
    engine = HandwritingEngine(font_paths, cfg)
    try:
        engine.render(text)
    except ValueError as e:  # e.g. too few labeled words for the handwriting model
        return jsonify({"error": str(e)}), 400

    needs = [c for c, _ in engine.fallback_used.most_common(60) if c.strip()]

    session_id = uuid.uuid4().hex
    # keep the engine: its word layers make single words editable afterwards
    sessions[session_id] = engine
    while len(sessions) > 3:                       # bounded memory (~100 MB/page each): drop the oldest
        sessions.pop(next(iter(sessions)))

    return jsonify({
        "session_id": session_id,
        "page_count": len(engine.pages),
        "pages": [_preview(p) for p in engine.pages],
        "words": [_word_boxes(engine, i) for i in range(len(engine.pages))],
        "needs": needs,
        "repeats": engine.repeat_report(),
    })


PREVIEW_W = 620


def _rgb(page: Image.Image) -> Image.Image:
    rgb = Image.new("RGB", page.size, (255, 255, 255))
    rgb.paste(page, mask=page.split()[3] if page.mode == "RGBA" else None)
    return rgb


def _preview(page: Image.Image) -> str:
    preview = _rgb(page)
    preview.thumbnail((PREVIEW_W, int(PREVIEW_W * 1.415)), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _word_boxes(engine, page: int) -> list:
    """Clickable word boxes for the preview of one page (preview pixels)."""
    k = PREVIEW_W / engine.cfg.page_width
    out = []
    for i, r in enumerate(engine.word_recs):
        if r["page"] != page:
            continue
        bb = engine.ink_bbox(i)
        if bb:
            out.append({"k": i, "text": r["text"], "box": [round(v * k, 1) for v in bb]})
    return out


def _session(session_id):
    engine = sessions.get(session_id)
    if engine is not None and not hasattr(engine, "_edit_lock"):
        engine._edit_lock = threading.Lock()
    return engine


def _locked(f):
    """One edit at a time per generated document ("making versions" and "save" both
    move the engine's pen state; run together they corrupted each other)."""
    import functools

    @functools.wraps(f)
    def run(session_id, k):
        engine = _session(session_id)
        if engine is None:
            return f(session_id, k)
        with engine._edit_lock:
            return f(session_id, k)
    return run


@app.route("/api/edit/<session_id>/<int:k>/candidates", methods=["POST"])
@_locked
def edit_candidates(session_id, k):
    """A few alternative renderings of one word (thumbnails)."""
    engine = _session(session_id)
    if engine is None:
        return jsonify({"error": "The server was restarted since you generated — generate again."}), 404
    if not 0 <= k < len(engine.word_recs):
        return jsonify({"error": "No such word"}), 404
    out = []
    for layer in engine.word_candidates(k, 4):
        bb = layer.getbbox()
        crop = layer.crop(bb) if bb else layer
        img = Image.new("RGB", (crop.width + 16, crop.height + 16), "white")
        img.paste(crop, (8, 8), crop)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return jsonify({"text": engine.word_recs[k]["text"], "candidates": out})


def _edited(engine, k):
    page = engine.word_recs[k]["page"]
    out = {"page": page, "preview": _preview(engine.pages[page]), "words": _word_boxes(engine, page)}
    if page > 0 and engine.cfg.show_through:            # the sheet before shows this page through
        out["prev"] = _preview(engine.pages[page - 1])
    return jsonify(out)


@app.route("/api/edit/<session_id>/<int:k>/apply", methods=["POST"])
@_locked
def edit_apply(session_id, k):
    engine = _session(session_id)
    if engine is None:
        return jsonify({"error": "The server was restarted since you generated — generate again."}), 404
    r = engine.word_recs[k]
    i = int((request.get_json() or {}).get("candidate", 0))
    cands = r.get("candidates") or []
    if not 0 <= i < len(cands):
        return jsonify({"error": "Pick one of the alternatives first"}), 400
    engine.rewrite_word(k, layer=cands[i])
    return _edited(engine, k)


@app.route("/api/edit/<session_id>/<int:k>/pad", methods=["POST"])
@_locked
def edit_pad(session_id, k):
    """Replace one word by the writer's own pad-written version (also saved to the bank)."""
    engine = _session(session_id)
    if engine is None:
        return jsonify({"error": "The server was restarted since you generated — generate again."}), 404
    if engine.bank is None or not engine.cfg.glyph_bank:
        return jsonify({"error": "Select your handwriting bank to use the pad"}), 400
    data = request.get_json() or {}
    word = engine.word_recs[k]["text"]
    # the editor shows the word with its brackets / punctuation and the writer copies
    # it as shown, so the sample IS the whole word (labelling it without them made the
    # brackets appear twice: the drawn ones plus added ones)
    label = "".join(word.split())
    bank = Path(engine.cfg.glyph_bank)
    try:
        rec = glyphbank.add_pad_sample(bank, label, data.get("strokes", []),
                                       float(data["baseline"]), float(data["x_height"]), rebuild=False)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    meta = _use_new_sample(engine, bank, rec)
    if meta is None:
        return jsonify({"error": "Couldn't read that writing — try again"}), 400
    engine.rewrite_word(k, plan=[(label, meta)])
    return _edited(engine, k)


def _use_new_sample(engine, bank, rec):
    """Make a just-saved pad sample usable in this engine right away; the full bank
    rebuild runs in the background (it used to block the editor for seconds)."""
    meta = glyphbank.pad_entry(bank, rec)
    if meta is not None:
        engine.bank._meta.setdefault(meta["label"], []).append(meta)
        engine.bank.max_len = max(engine.bank.max_len, len(meta["label"]))
    glyphbank.schedule_build(bank)
    return meta


def _sentence_words(engine, k):
    """(word index, label) for the sentence around word k; words the pad can't write as
    one sample (no letters/digits, or e.g. "e.g.") are left as they are."""
    out = []
    for i in engine.sentence_range(k):
        found = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", engine.word_recs[i]["text"])
        if len(found) == 1:
            out.append((i, found[0]))
    return out


@app.route("/api/edit/<session_id>/<int:k>/sentence-range", methods=["POST"])
def edit_sentence_range(session_id, k):
    engine = _session(session_id)
    if engine is None:
        return jsonify({"error": "The server was restarted since you generated — generate again."}), 404
    sw = _sentence_words(engine, k)
    return jsonify({"ks": [i for i, _ in sw], "text": " ".join(w for _, w in sw)})


@app.route("/api/edit/<session_id>/<int:k>/sentence", methods=["POST"])
@_locked
def edit_sentence(session_id, k):
    """Replace a whole sentence by the writer's pad-written version: split into words,
    each saved to the bank, each word of the sentence on the page replaced."""
    engine = _session(session_id)
    if engine is None:
        return jsonify({"error": "The server was restarted since you generated — generate again."}), 404
    if engine.bank is None or not engine.cfg.glyph_bank:
        return jsonify({"error": "Select your handwriting bank to use the pad"}), 400
    data = request.get_json() or {}
    sw = _sentence_words(engine, k)
    bank = Path(engine.cfg.glyph_bank)
    try:
        ids = glyphbank.add_pad_sentence(bank, " ".join(w for _, w in sw), data.get("strokes", []),
                                         float(data["baseline"]), float(data["x_height"]))
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    skipped, pages, done = [], set(), []
    for (i, label), rec in zip(sw, ids):
        meta = _use_new_sample(engine, bank, rec)
        if meta is None:
            skipped.append(label)                       # the quality check left it out
            continue
        word = engine.word_recs[i]["text"]
        plan = [(label, meta)]
        if label != word:                               # keep punctuation around the word
            pre, post = word.split(label, 1) if label in word else ("", "")
            plan = [(c, None) for c in pre] + plan + [(c, None) for c in post]
        engine.rewrite_word(i, plan=plan, post=False)
        pages.add(engine.word_recs[i]["page"])
        done.append(i)
    engine.recompose(pages, words=done)
    pages |= {p - 1 for p in pages if p > 0}           # show-through: the sheet before changed too
    return jsonify({"pages": {p: _preview(engine.pages[p]) for p in pages},
                    "words": {p: _word_boxes(engine, p) for p in pages},
                    "page": engine.word_recs[k]["page"], "skipped": skipped})


@app.route("/api/download/<session_id>", methods=["GET"])
def download_pdf(session_id):
    """Download the generated PDF for a session."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found. Generate first."}), 404

    engine = sessions[session_id]
    buf = io.BytesIO(pages_to_pdf([_rgb(p) for p in engine.pages], photo=engine.cfg.scan_look))

    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="handscript_output.pdf",
    )


# ── Routes: Glyph bank labeling ──────────────────────────────────────────

def _bank_path(name):
    """banks/<name> if it has a built bank.json, else None."""
    if not name or name not in glyphbank.find_banks(BANKS_DIR):
        return None
    path = BANKS_DIR / name
    return str(path) if (path / "bank.json").exists() else None


def _bank_dir(name):
    if name not in glyphbank.find_banks(BANKS_DIR):
        return None
    return BANKS_DIR / name


@app.route("/label")
def label_page():
    return send_from_directory("ui", "label.html")


# ── Routes: Structured input (PDF import, local AI) ──────────────────────

@app.route("/api/llm/status", methods=["GET"])
def llm_status():
    import llm
    return jsonify(llm.available())


@app.route("/api/tidy", methods=["POST"])
def tidy_text():
    """Pasted text -> renderer Markdown with the local LLM."""
    import llm
    status = llm.available()
    if not status["ok"]:
        return jsonify({"error": status["error"]}), 503
    text = (request.get_json() or {}).get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is required"}), 400
    return jsonify({"markdown": llm.tidy(text)})


@app.route("/api/import-pdf", methods=["POST"])
def import_pdf():
    """PDF upload -> Markdown (layout kept; figures via local AI or pad placeholders)."""
    import pdfimport
    f = request.files.get("pdf")
    if f is None or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Upload a .pdf file"}), 400
    use_llm = request.form.get("use_llm") == "1"
    try:
        md, notes = pdfimport.pdf_to_markdown(f.read(), use_llm=use_llm)
    except Exception as e:  # malformed PDFs
        return jsonify({"error": f"Could not read the PDF: {e}"}), 400
    return jsonify({"markdown": md, "notes": notes})


@app.route("/api/banks/<name>/sketches", methods=["GET", "POST"])
def bank_sketches(name):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    if request.method == "GET":
        return jsonify(glyphbank.list_sketches(bank))
    data = request.get_json() or {}
    try:
        saved = glyphbank.add_sketch(bank, data.get("name", ""), data.get("strokes", []), data.get("x_height"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"name": saved, "markdown": f"![{saved}](pad:{saved})"})


@app.route("/bank")
def bank_page():
    return send_from_directory("ui", "bank.html")


@app.route("/api/banks/<name>/pieces", methods=["GET"])
def bank_pieces(name):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    return jsonify(glyphbank.pieces_by_label(bank))


@app.route("/banks/<name>/glyphs/<path:filename>")
def bank_glyph_image(name, filename):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    return send_from_directory(bank / "glyphs", filename)


@app.route("/api/banks/<name>/prefer-pad", methods=["GET", "POST"])
def bank_prefer_pad(name):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    if request.method == "POST":
        glyphbank.set_prefer_pad(bank, bool((request.get_json() or {}).get("on")))
    return jsonify({"on": glyphbank.prefer_pad(bank)})


@app.route("/api/banks/<name>/pieces/<path:file>", methods=["POST"])
def bank_piece_status(name, file):
    """{"removed": true|false} — remove a piece from use or bring it back."""
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    glyphbank.set_removed(bank, file, bool((request.get_json() or {}).get("removed")))
    return jsonify({"ok": True})


@app.route("/api/banks/<name>/fonts", methods=["GET", "POST"])
def bank_fonts(name):
    """Fallback fonts for this bank: list with on/off, or {"font": "x.ttf", "enabled": bool}."""
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    path = bank / "fonts.json"
    off = json.loads(path.read_text(encoding="utf-8")).get("disabled", []) if path.exists() else []
    if request.method == "POST":
        data = request.get_json() or {}
        f = data.get("font", "")
        off = [x for x in off if x != f] + ([] if data.get("enabled") else [f])
        path.write_text(json.dumps({"disabled": off}), encoding="utf-8")
    fonts = sorted(p.name for p in BUNDLED_FONTS_DIR.glob("*.[to]tf"))
    return jsonify([{"font": f, "enabled": f not in off} for f in fonts])


@app.route("/api/fonts/preview/<path:font>")
def font_preview(font):
    from PIL import ImageDraw, ImageFont
    p = BUNDLED_FONTS_DIR / font
    if not p.exists():
        return jsonify({"error": "Font not found"}), 404
    f = ImageFont.truetype(str(p), 34)
    img = Image.new("L", (360, 52), 255)
    ImageDraw.Draw(img).text((6, 4), "Aa Bb Kk Qq yz 123", font=f, fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/banks/<name>/words-to-write", methods=["POST"])
def bank_words_to_write(name):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    text = (request.get_json() or {}).get("text", "")
    return jsonify(glyphbank.words_to_write(bank, text))


@app.route("/api/banks/<name>/sentences-to-write", methods=["POST"])
def bank_sentences_to_write(name):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    text = (request.get_json() or {}).get("text", "")
    return jsonify(glyphbank.sentences_to_write(bank, text))


@app.route("/api/banks/<name>/sentences", methods=["POST"])
def add_sentence(name):
    """A sentence written on the pad -> one sample per word."""
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    data = request.get_json() or {}
    try:
        recs = glyphbank.add_pad_sentence(bank, data.get("text", ""), data.get("strokes", []),
                                          float(data["baseline"]), float(data["x_height"]))
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    glyphbank.schedule_build(bank)
    return jsonify({"ids": [r["id"] for r in recs]})


@app.route("/capture")
def capture_page():
    return send_from_directory("ui", "capture.html")


@app.route("/api/banks/<name>/prompts", methods=["GET"])
def bank_prompts(name):
    """Words to write on the pen tablet: missing letters first, then common words."""
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    return jsonify(glyphbank.prompts(bank))


@app.route("/api/banks/<name>/samples", methods=["GET"])
def list_samples(name):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    return jsonify(glyphbank.list_pad_samples(bank))


@app.route("/api/banks/<name>/samples/batch", methods=["POST"])
def add_samples_batch(name):
    """Many pad samples at once (character sheet): [{text, strokes, baseline, x_height}]."""
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    saved, skipped = [], 0
    for it in (request.get_json() or {}).get("items", []):
        try:
            rec = glyphbank.add_pad_sample(bank, it.get("text", ""), it.get("strokes", []),
                                           float(it["baseline"]), float(it["x_height"]), rebuild=False)
            saved.append(rec["id"])
        except (ValueError, KeyError):
            skipped += 1
    moved = glyphbank.fix_pad_baselines(bank, rebuild=False)   # written on the wrong guide line
    glyphbank.build(bank)
    return jsonify({"saved": len(saved), "skipped": skipped, "realigned": moved})


@app.route("/sheet")
def sheet_page():
    return send_from_directory("ui", "sheet.html")


@app.route("/api/banks/<name>/samples/<sample_id>", methods=["DELETE"])
def delete_sample(name, sample_id):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    if not glyphbank.delete_pad_sample(bank, sample_id):
        return jsonify({"error": "Sample not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/banks/<name>/samples", methods=["POST"])
def add_sample(name):
    """Save one word written on the pen tablet (strokes + guide geometry)."""
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    data = request.get_json()
    try:
        rec = glyphbank.add_pad_sample(bank, data.get("text", ""), data.get("strokes", []),
                                       float(data["baseline"]), float(data["x_height"]), rebuild=False)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    glyphbank.schedule_build(bank)          # rebuild once writing pauses; answer right away
    return jsonify({"id": rec["id"]})


@app.route("/api/banks", methods=["GET"])
def list_banks():
    out = []
    for name in glyphbank.find_banks(BANKS_DIR):
        words = glyphbank.load_words(BANKS_DIR / name)
        out.append({
            "name": name,
            "words": len(words),
            "done": sum(w["status"] != "todo" for w in words),
            "ready": (BANKS_DIR / name / "bank.json").exists(),
            "model_ready": wordmodel.bank_ready(str(BANKS_DIR / name)),
        })
    return jsonify(out)


@app.route("/api/banks/<name>/words", methods=["GET"])
def bank_words(name):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    return jsonify({"words": glyphbank.load_words(bank), "coverage": glyphbank.coverage(bank)})


@app.route("/banks/<name>/words/<path:filename>")
def bank_word_image(name, filename):
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    return send_from_directory(bank / "words", filename)


@app.route("/api/banks/<name>/words/<word_id>", methods=["POST"])
def save_word(name, word_id):
    """Save cuts + labels (or skip) for one word, then rebuild the bank."""
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    data = request.get_json()
    words = glyphbank.load_words(bank)
    word = next((w for w in words if w["id"] == word_id), None)
    if word is None:
        return jsonify({"error": "Word not found"}), 404

    status = data.get("status", "done")
    cuts = sorted(int(c) for c in data.get("cuts", word["cuts"]) if 0 < int(c) < word["w"])
    labels = data.get("labels")
    if status == "done" and (not labels or len(labels) != len(cuts) + 1):
        return jsonify({"error": f"Need {len(cuts) + 1} labels, one per piece"}), 400
    word.update(cuts=cuts, labels=labels if status == "done" else None, status=status)
    word.pop("auto", None)  # reviewed by a person
    glyphbank.save_words(bank, words)
    meta = glyphbank.build(bank)
    return jsonify({"coverage": glyphbank.coverage(bank), "slant_deg": meta["slant_deg"]})


# ── Main ─────────────────────────────────────────────────────────────────

# ── Practice notebook (/practice): write over grey guide text, check, save ──────

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"


def _notebook_path(name: str):
    safe = re.sub(r"[^A-Za-z0-9 _\-]", "", name or "").strip()[:60]
    return (NOTEBOOKS_DIR / f"{safe}.json") if safe else None


@app.route("/practice")
def practice_page():
    return send_from_directory("ui", "practice.html")


@app.route("/fonts/<path:filename>")
def bundled_font_file(filename):
    """The bundled .ttf files, for the practice page's guide-text font choice."""
    return send_from_directory(BUNDLED_FONTS_DIR, filename)


@app.route("/api/notebooks", methods=["GET"])
def notebooks_list():
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    items = sorted(NOTEBOOKS_DIR.glob("*.json"), key=lambda p: -p.stat().st_mtime)
    return jsonify([p.stem for p in items])


@app.route("/api/notebooks/<name>", methods=["GET", "PUT", "DELETE"])
def notebook(name):
    path = _notebook_path(name)
    if path is None:
        return jsonify({"error": "Give the notebook a name"}), 400
    if request.method == "GET":
        if not path.exists():
            return jsonify({"error": "No such notebook"}), 404
        return send_file(path, mimetype="application/json")
    if request.method == "DELETE":
        path.unlink(missing_ok=True)
        return jsonify({"ok": True})
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(request.get_json() or {}), encoding="utf-8")
    tmp.replace(path)                                    # never leave a half-written notebook
    return jsonify({"ok": True})


def _data_url_image(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


@app.route("/api/practice/check", methods=["POST"])
def practice_check():
    """Read written words back with the handwriting reader (TrOCR): which ones are hard
    to read. {items: [{text, image (data URL, dark ink on white)}]} ->
    [{read, miss}] in the same order (miss: 0 = read exactly, 1 = unreadable)."""
    import proofread
    items = (request.get_json() or {}).get("items", [])[:400]
    if not items:
        return jsonify([])
    try:
        reader = proofread.reader()
    except Exception as e:                               # torch / transformers missing
        return jsonify({"error": f"The handwriting reader isn't available: {e}"}), 503
    reads = reader.read_batch([_data_url_image(it["image"]) for it in items])
    return jsonify([{"read": r, "miss": round(proofread.mismatch(r, it["text"]), 3)}
                    for r, it in zip(reads, items)])


@app.route("/api/practice/read", methods=["POST"])
def practice_read():
    """Read one line of the writer's handwriting (to find where in the text they are)."""
    import proofread
    url = (request.get_json() or {}).get("image")
    if not url:
        return jsonify({"error": "No image"}), 400
    try:
        reader = proofread.reader()
    except Exception as e:
        return jsonify({"error": f"The handwriting reader isn't available: {e}"}), 503
    return jsonify({"text": reader.read_batch([_data_url_image(url)])[0]})


@app.route("/api/practice/pdf", methods=["POST"])
def practice_pdf():
    pages = [_data_url_image(u) for u in (request.get_json() or {}).get("pages", [])]
    if not pages:
        return jsonify({"error": "Nothing to export"}), 400
    return send_file(io.BytesIO(pages_to_pdf(pages, photo=False)), mimetype="application/pdf",
                     as_attachment=True, download_name="practice.pdf")


@app.route("/api/practice/to-bank/<name>", methods=["POST"])
def practice_to_bank(name):
    """Traced words that read correctly become pad samples of the writer's hand:
    {items: [{text, strokes, baseline, x_height}]} (page px)."""
    bank = _bank_dir(name)
    if bank is None:
        return jsonify({"error": "Bank not found"}), 404
    saved, failed = [], []
    for it in (request.get_json() or {}).get("items", [])[:200]:
        label = (re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", it.get("text", "")) or [""])[0]
        try:
            rec = glyphbank.add_pad_sample(bank, label, it.get("strokes", []), float(it["baseline"]),
                                           float(it["x_height"]), rebuild=False)
            saved.append(rec["id"])
        except (ValueError, KeyError):
            failed.append(it.get("text", ""))
    if saved:
        glyphbank.schedule_build(bank)
    return jsonify({"saved": len(saved), "failed": failed})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"HandScript server running at http://localhost:{port}")
    # load the reading-check model in the background so the first generate doesn't wait
    def _warm():
        try:
            import proofread
            proofread.reader()
        except Exception as e:                      # optional: no torch/transformers
            print("reading check unavailable:", e)
    threading.Thread(target=_warm, daemon=True).start()
    # No auto-reloader: loading the handwriting model touches site-packages, which
    # restarts the server mid-request and throws away the loaded model.
    app.run(debug=True, port=port, use_reloader=False)
