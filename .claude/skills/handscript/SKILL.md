---
name: handscript
description: Write documents out in the user's OWN handwriting as a PDF (assignments, records, notes, answers, copied-out PDFs) with the HandScript generator in this repo, including hand-drawn flowcharts, ruled tables and formulas. Use whenever the user asks for something "handwritten", "in my handwriting", "write this by hand", wants an assignment/record/notes written out, sends text or a PDF to be copied out by hand, or says handscript, even if they just paste text and say "make the pdf".
---

# HandScript: PDFs in the user's handwriting

This repo turns Markdown into pages written in the user's own handwriting (their samples
live in `banks/mine`). Your part is the content: write good Markdown (including the
figures), run the generator, look at the result, fix what looks wrong, deliver the PDF.

## Workflow

1. **Get the content.**
   - User pasted text / gave a file: use it as is, only adding Markdown structure
     (headings for questions, lists, tables) where it obviously helps.
   - User gave questions or a topic ("write answers for assignment 2: ..."): write the
     answers yourself, like a good student writing by hand: to the point, a heading per
     question, short paragraphs, bullet points, a table for comparisons, a flowchart for
     processes or architectures. Handwriting is slow to read; about 200 words fill a page.
   - User gave a PDF to copy out: read it yourself (it is better than the raw import) and
     write clean Markdown: fix broken lines/hyphenation, keep tables, turn figures into
     Mermaid flowcharts when they are flow/architecture diagrams. `--pdf file.pdf` is the
     quick route when the PDF is plain text.
2. **Write it to** `generated/<short-name>.md`.
3. **Generate** (from the repo root):
   ```
   python main.py --file generated/<name>.md -o generated/<name>.pdf --ink blue --preview generated/preview --json
   ```
   Takes ~5-10 s per page (longer on a CPU-only machine). Pick options from the table below.
4. **Look at the previews** (`generated/preview/<name>_p1.png`, ...; read 1-2 pages, all
   pages for short documents). Things to catch: a table cell overflowing into the next
   (shorten the cell text), a diagram that came out cramped (fewer nodes, shorter
   labels), a heading or word that reads wrong. Fix the Markdown and rerun; don't
   hand-edit the PDF.
5. **Deliver the PDF.**
   - Session on the user's PC (desktop app / terminal / Remote Control): send it with the
     file-sending tool (SendUserFile) and give the path.
   - Cloud session (claude.ai/code, e.g. from the phone; the repo was cloned into a
     fresh machine): `generated/` is git-ignored, so `git add -f generated/<name>.pdf`,
     commit, push the session's branch, and give the user the GitHub link to the file
     (`https://github.com/<owner>/<repo>/blob/<branch>/generated/<name>.pdf`). Opening
     it on GitHub shows the PDF; the download button saves it.
6. **Tell the user** pages written, and from the JSON summary: `font_letters` (letters
   their bank lacks, written with a font: suggest writing them on the pad at /capture),
   `hard_to_read`, `repeats` (words worth writing a few more copies of on the pad).

## Options (map the user's words)

| User says | Option |
|---|---|
| blue pen / ballpoint (default for assignments) | `--ink blue` |
| black pen | `--ink black` (default) |
| a specific colour | `--ink "#RRGGBB"` |
| will print it double-sided / make it look like a real sheet | `--show-through` (0.5; `--show-through 0.3` subtler) |
| looks scanned / photographed (not for printing) | `--scan-look` |
| more space between lines | `--line-spacing 1.2` (1.0 default) |
| wider / narrower letters | `--letter-width 1.1` / `0.9` |
| sharper print | `--dpi 450` or `600` (bigger file, slower) |
| make sure every word is readable | `--proofread` (rereads words, rewrites misread ones; slow without a GPU) |
| less/more ink texture | `--texture 0.2` / `0.6` (0.4 default) |

Leave `--model` (AI handwriting model) alone unless the user asks: it needs a big GPU.

## What the Markdown can contain

- `#`/`##` headings (written larger and underlined), paragraphs, `-` and `1.` lists
  (nested), `> quotes`, `---` (a hand-drawn line), ``` code ``` (lines kept).
- **Tables**: hand-ruled. Keep cells short (a few words); long words overflow the cell.
- **Flowcharts**: ```` ```mermaid ```` with `flowchart TD`/`LR` (or `graph`): nodes
  `A[box]`, `A(rounded)`, `A{decision}`, `A((circle))`, `A([stadium])`; edges `-->`,
  `---`, `-.->`, `==>`, labels `-->|yes|`; chains `A --> B --> C`. Other Mermaid types
  (sequence, class, pie, gantt) are not drawn: use a flowchart, a table or words instead.
  Keep labels short (1-3 words) and nodes under ~8 per chart: it's drawn by hand.
- **Math**: `$...$` inline, `$$...$$` display: LaTeX subset (`\frac`, `\sqrt`, `^ _`,
  `\sum \int \lim`, Greek, `\sin`...). Prefer `\cdot` over `\times` (the × comes out
  small).
- **A drawing that isn't a flowchart** (circuit, graph, sketch): `![caption](pad:name)`
  places a sketch the user drew on the pad (/capture, sketch mode). If it doesn't exist
  yet, say so and describe what to draw, or replace it with a short description.

## Notes

- One question's answer starting on a fresh line after a `?` is automatic.
- Words written by the user on the pad later improve every future PDF; if something
  reads badly in the preview, it is usually a weak sample: mention the word.
- Generated files stay out of git (`generated/` is ignored); commit a PDF only to hand it
  over from a cloud session.
