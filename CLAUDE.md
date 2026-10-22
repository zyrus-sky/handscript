# HandScript (handwritten-text PDF renderer)
## Commands
- install: `pip install -r requirements.txt`
- run web app: `python app.py` → http://localhost:5000 (`PORT` env overrides); labeling UI at `/label`, pen-tablet capture at `/capture`
- glyph bank: `python glyphbank.py extract <scan.pdf> --bank banks/<name>` (segments words + writes layout.json; `measure` redoes only the layout; `--append` adds more pages/photos to an existing bank, colour photos on ruled paper are reduced to the ink); labeling rebuilds `banks/<name>/bank.json`; `python glyphbank.py build --bank banks/<name>` rebuilds by hand
- run CLI: `python main.py --file notes.md -o out.pdf --ink blue --preview generated/preview --json` (also `--text`, `--pdf`, `--file -`; `--bank none`, `--show-through`, `--proofread`, `--dpi`, `--letter-width`); same settings as the web generator via `genconfig.config_from_params`
- skill: `.claude/skills/handscript/SKILL.md` (write Markdown → main.py → check previews → deliver; cloud sessions commit `generated/*.pdf` with `git add -f`); SessionStart hook `.claude/hooks/ensure_deps.py` installs packages on fresh cloud machines
- repo: private github.com/zyrus-sky/handscript (includes banks/mine); push after adding pad samples so cloud sessions write with them
- handwriting model deps (optional): `pip install -r requirements-model.txt` (keep huggingface_hub<1.0 for transformers 4.x)
- tests: none formal; ad-hoc scripts `python test_fatigue.py`, `python test_run_cursive.py` (hardcode an old fonts path `d:/Projects_Grouped/DZypher/upcycle_ai/hand/fonts`), `reproduce_error.py` (needs server running)
## Live code
- entry points: `app.py` (Flask API + serves `ui/index.html`), `main.py` (CLI)
- `engine.py`: `EngineConfig`, `HandwritingEngine` (render, save_pdf), `generate_handwritten_pdf`
- `glyphbank.py`: scan → word crops (deslanted, baseline/x-height) → labeled pieces → `GlyphBank` used by the engine
- `docrender.py`: Markdown → document blocks (headings, lists, tables, ```mermaid via `flowchart.py`, $math$ via `mathlayout.py`, ![](pad:name) sketches); engine `render()` uses it when `structured` (default)
- `pdfimport.py`: PDF → Markdown (PyMuPDF text/tables, figures → local LLM or pad placeholders); `llm.py`: Ollama (`HANDSCRIPT_LLM`, default gemma4:12b, think off) for tidy / figure reading / OCR
- `vectorfont.py`: font letters from outlines (fontTools), smooth random deformation per instance; engine `_render_vector_char`
- spacing: engine `_place_by_ink` (closest ink approach = `layout.json` `letter_gap` from `glyphbank.letter_gaps`); quality check: `glyphbank.audit` (+ `_trim_tail` repair) on every build
- `morph.py`: learned font→writer deformations per letter/form (`banks/<name>/morph.json`, refit: `python morph.py --bank banks/mine`)
- joins: engine `_connect` (connecting strokes, `letter_joins` ≈ measured 0.27–0.3); reading check: `proofread.py` (TrOCR small, safetensors, GPU) via engine `_write_checked` (`proofread`); `paper.py` `scanify` (`scan_look`)
- UI pages: `/` generator, `/label`, `/capture` (word pad + recent samples), `/sheet` (character sheet), `/bank` (manage pieces, fonts, prefer-pad)
- `/practice` (`ui/practice.html`): handwriting practice notebook — text/PDF laid out as grey guide words (architect-lettering font) on ruled pages, pen/eraser, progress follows ink coverage of guide words, Tab reveals more; `/api/practice/check` (TrOCR readability), `/api/practice/to-bank/<bank>` (well-read traced words → pad samples), `/api/practice/pdf`; notebooks saved in `notebooks/<name>.json`
- editing: `EngineConfig.editable` keeps each word as a layer (`word_recs`, `page_bases`, `raw_pages`); `word_candidates` / `rewrite_word` + `/api/edit/<session>/<k>/{candidates,apply,pad}`; sessions keep the engine
- realism: `paper.py` `ink_texture` / `show_through` / `scanify` in engine `_postprocess_all`; per-word lean/tilt/size/weight (`word_*` config); page-level repeat avoidance (`GlyphBank.new_page`, `repeat_report`)
- pad strokes: `glyphbank.render_strokes` smooths (RDP + Catmull-Rom), carries ink amount (speed, landing/lift, turn-back/crossing layers → engine `_ink_amount`, Beer-Lambert) and trims tablet exit tails / caps descenders; engine `paste_multiply` layers ink multiplicatively; bank `pad_stretch` (paper vs pad width per letter) widens pad pieces, `EngineConfig.letter_width` on top
- bank audit reads multi-letter pieces with TrOCR (cached in `audit.json` `ocr`; background rebuilds read new pieces when the reader is loaded); `typ` = resemblance to other copies (planner cost)
- pad: saves rebuild in the background (`glyphbank.schedule_build`); sentences: `sentences_to_write`, `add_pad_sentence` (`/api/banks/<name>/sentences`)
- `wordmodel.py`: Emuru (HF `blowing-up-groundhogs/emuru`, ~3 GB VRAM) generates ≤24-char chunks styled by the bank's fully labeled words; engine `_render_with_model` lays them out
- `ui/index.html`: single-page frontend; `ui/label.html`: glyph labeling page
- `banks/<name>/`: words.json + words/*.png (labels live here), bank.json + glyphs/ (derived)
- `fonts/`: bundled .ttf fonts used by the server
## Conventions
- *.png / *.pdf in root are test outputs, not source
## Project map
- docs/PROJECT_MAP.md: modules, main flows, where to change what. Read the relevant section before working in an unfamiliar area. Check what changed since it was built: project-scan --changed docs/PROJECT_MAP.md.
## Compact instructions
When compacting, keep: files and line numbers touched, decisions made, failing tests, next steps.
