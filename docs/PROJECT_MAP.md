# HandScript map
Turns plain text into realistic handwritten-style multi-page PDFs (A4 at 150 DPI) by rendering each character with a randomly chosen handwriting font and per-character jitter (rotation, scale, opacity, baseline wave, slope), plus a progressive "fatigue" drift. Two front doors: a Flask web app with a single-page UI, and a CLI.

## Run it
- deps: `pip install -r requirements.txt flask flask-cors` (requirements.txt lists only `Pillow>=10.0`; app.py also imports flask, flask_cors)
- web: `python app.py` → http://localhost:5000 (`app.py:209`, debug=True)
- CLI: `python main.py --text "Hello" --fonts fonts/1.ttf fonts/2.ttf -o out.pdf` (args `main.py:17-45`)
- tests: none formal. `test_fatigue.py`, `test_run_cursive.py` are visual scripts with a stale fonts path (`test_fatigue.py:6`); `reproduce_error.py` POSTs to a running server.

## Modules
| module | responsibility | key files |
|---|---|---|
| engine | all rendering: config knobs, per-char rendering, layout, pagination, PDF save | `engine.py` |
| web backend | Flask routes: serve UI, font upload/list, generate previews, PDF download | `app.py` |
| CLI | argparse → EngineConfig → render → save_pdf | `main.py` |
| ui | single HTML page (Tailwind CDN, vanilla JS): sliders, font drop zone, preview pager, download | `ui/index.html` |
| glyphbank | cut the writer's own letters/joined pieces out of scans; label them; serve variants to the engine | `glyphbank.py`, `ui/label.html`, `banks/<name>/` |
| wordmodel | Emuru handwriting model: style line from labeled bank words → generated word chunks | `wordmodel.py` |
| fonts | bundled .ttf fonts, used when the request gives no font paths | `fonts/` |

## Main flows
### Web: generate preview
1. `ui/index.html:592` `generate()` POSTs `{text, font_paths: state.fontPaths, config: getConfig()}` to `/api/generate` (`ui/index.html:604`); config built from sliders in `getConfig()` `ui/index.html:560`
2. `app.py:83` `generate()`: empty font_paths → all `fonts/*.ttf|*.otf` (`app.py:92-95`)
3. `app.py:101` maps flat UI params to `EngineConfig` (symmetric ranges built as `(-abs(x), abs(x))`)
4. `app.py:154-155` `HandwritingEngine(...).render(text)`
5. `app.py:162-173` RGBA→RGB, thumbnail to 620x877 PNG, base64 data URLs
6. `app.py:176` pages kept in in-memory `sessions[session_id]` (`app.py:31`); JSON `{session_id, page_count, pages}` back to UI → `showPage()` `ui/index.html:635`

### Web: download PDF
1. `ui/index.html:670` `window.open('/api/download/' + sessionId)`
2. `app.py:186` `download_pdf()` reads `sessions[...]`, saves multi-page PDF into BytesIO, `send_file` as `handscript_output.pdf`

### Web: fonts
- list: `ui/index.html:491` `loadBundledFonts()` → `app.py:70` `list_bundled_fonts()` (globs `fonts/`)
- upload: `ui/index.html:541` `uploadFonts()` → `app.py:49` `upload_font()` saves to `%TEMP%/handscript_fonts/<uuid8>_<name>` (`app.py:25`), returns its server path, which the UI later sends back as `font_paths`

### Engine: render(text)
1. `engine.py:403` `render()` resets pages/fatigue, `_new_page()` `engine.py:166` (blank RGBA canvas, new line params `engine.py:178`)
2. split on `\n` into paragraphs, then on spaces into words (`engine.py:411-415`)
3. per word: `_measure_word` `engine.py:389` → wrap via `_line_break` `engine.py:373`, `_needs_page_break` `engine.py:380` → `_page_break` `engine.py:383`
4. per char: random font (`engine.py:435`) → `_render_char` `engine.py:236` (rotate/scale/opacity, advances fatigue `engine.py:288`) → `_paste_char` `engine.py:339` (y from `_compute_y_offset` `engine.py:303`) → `_advance_cursor` `engine.py:360`
5. paragraph end: line break + `paragraph_spacing`, partial fatigue recovery (`engine.py:450-457`)
6. `engine.py:460` last page saved; `save_pdf` `engine.py:465` flattens to RGB and writes PDF at 150 DPI

### Glyph bank
1. `python glyphbank.py extract scan.pdf --bank banks/mine`: page → ink mask → lines (components chained left→right) → per-line slant removed → words split on column gaps → baseline/x-height per word (outliers pulled to line median) → `words.json`
2. `/label` (`ui/label.html`) → `POST /api/banks/<name>/words/<id>` (`app.py` `save_word`) saves cuts+labels, calls `glyphbank.build` → `glyphs/*.png` + `bank.json` (median slant)
3. `layout.json` (`glyphbank.measure_layout`): writer's x-height, line pitch, margins, word gaps, drift, stroke; engine `_apply_writer_layout` uses it when `writer_layout` (default on)
4. engine: `EngineConfig.glyph_bank` → `GlyphBank.tokenize` (longest label first, e.g. `the`) → `_render_bank_glyph` scales to the font x-height, re-slants around the baseline, inks → `_paste_char` aligns baselines; missing letters fall back to fonts

### Structured documents
1. `engine.render` → `docrender.render_document`: markdown-it tokens → `DocWriter` blocks; text via `engine.write_word` / `newline` (region `_left`/`_right`), lines/boxes via `engine.hand_path` / `hand_line` / `hand_arrow`
2. tables: measured column widths (×1.18 headroom), wrapped cells, hand-ruled grid, page break between rows
3. ```mermaid → `flowchart.parse` → `layout` (ranks by longest path, barycenter order) → `draw`; $..$ / $$..$$ → `mathlayout.MathWriter` (boxes: frac, sqrt, scripts, big operators) → `engine.glyph`
4. `engine.fallback_used` → `needs` in /api/generate → UI links `/capture?words=…`
5. `/api/import-pdf` → `pdfimport.pdf_to_markdown`; `/api/tidy` → `llm.tidy`; sketches: `/api/banks/<name>/sketches` → `glyphbank.add_sketch` → `banks/<name>/sketches/*.json`

### Writing pad (`/capture`, `ui/capture.html`)
1. prompts from `glyphbank.prompts` (missing letters/capitals first, then common words with < 4 variants)
2. pen strokes + guide lines → `POST /api/banks/<name>/samples` → `glyphbank.add_pad_sample`: normalise to x-height units, deslant in vector space, save `strokes/<id>.json` + word record (`source: pad`, auto-labeled) → `build`
3. engine `_render_bank_glyph`: glyphs with strokes are redrawn by `render_strokes` with jitter each use; scanned pieces get `_warp`; `GlyphBank.pick` least-recently-used; `tokenize` sometimes splits joined pieces with few variants (`glyph_variation` scales all of it)

### Handwriting model (EngineConfig.handwriting_model)
1. `engine.py` `_render_with_model`: `StyleWriter(bank)` builds 3 style lines from random fully labeled words
2. text split into chunks ≤ `MAX_CHUNK_CHARS` (24) → `StyleWriter.write_line` → Emuru `generate` (model cached per process; retries once if width implausible)
3. chunk alpha normalized, scaled to font x-height (running reference), slanted, placed with word spacing / slope / wave; wraps on real widths

### CLI
`main.py:12` `main()` → text from `--text` (`\n` escapes) or `--file` → `EngineConfig` (font size, line height, uniform margin, ink color) → `render` → `save_pdf` → prints page count.

## Data and outside systems
- no database, no env vars. State: in-memory `sessions` dict (`app.py:31`), grows unbounded, lost on restart.
- uploaded fonts: OS temp dir `handscript_fonts` (`app.py:24-25`), never cleaned.
- UI loads Tailwind CDN, Google Fonts, a googleusercontent image (`ui/index.html:7-9`, `:404`).
- CORS enabled for all origins (`app.py:20`).

## Where to change what
- add/adjust a rendering knob: field on `EngineConfig` `engine.py:35`, use it in the engine; expose via `app.py:101` mapping and `getConfig()` `ui/index.html:560` + a slider in `SLIDERS` `ui/index.html:424`; CLI flag in `main.py:34`
- per-character look (rotation, scale, opacity, ink colour `engine.py:271`): `_render_char` `engine.py:236`
- baseline / slope / wave: `_compute_y_offset` `engine.py:303`, `_new_line_params` `engine.py:178`
- fatigue behaviour: `_fatigue_multiplier` `engine.py:206`, `_advance_fatigue` `engine.py:217`, knobs `engine.py:101`
- page size / margins: `EngineConfig` `engine.py:39`
- add an API route: follow `app.py:82`
- preview size / format: `app.py:168-171`

## Conventions
- `EngineConfig` is a dataclass of all knobs; ranges are `(lo, hi)` tuples, fatigue-scaled via `_fatigue_scale`.
- root `*.png` / `*.pdf` are test outputs, not source.

## Open questions
- fatigue knobs are accepted by `/api/generate` (`app.py:145-150`) but `getConfig()` never sends them, so the UI always uses defaults. Intended?
- `generate_handwritten_pdf` `engine.py:490` has no caller in this repo; possibly kept as a library API.
- `font_paths` from the client are opened as server paths without validation (`engine.py:150`).
<!-- project-map fingerprint (project-scan --changed compares against it)
built 2026-10-11
(root files) ee448c7e04d3
docs/ a8cd0080cd4a
ui/ 7a141d0e062f
-->
