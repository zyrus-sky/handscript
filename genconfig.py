"""
Generator settings -> EngineConfig, shared by the web app (app.py /api/generate) and the
command line (main.py), so both produce the same handwriting for the same settings.
`params` are the web generator's settings (ui/index.html getConfig); anything missing
takes the generator's default.
"""

import json
from pathlib import Path
from typing import List, Optional

import glyphbank
from engine import EngineConfig

ROOT = Path(__file__).parent
BANKS_DIR = ROOT / "banks"
BUNDLED_FONTS_DIR = ROOT / "fonts"

# ink presets (the generator's Ink Color buttons)
INKS = {"blue": (12, 2, 118), "black": (20, 20, 22)}


def bank_path(name: Optional[str]) -> Optional[str]:
    """banks/<name> if it has a built bank.json, else None."""
    if not name or name not in glyphbank.find_banks(BANKS_DIR):
        return None
    path = BANKS_DIR / name
    return str(path) if (path / "bank.json").exists() else None


def config_from_params(params: dict) -> EngineConfig:
    cfg = EngineConfig(
        font_size=params.get("font_size", 42),
        line_height=params.get("line_height", 60),
        line_spacing=float(params.get("line_spacing", 1.0)),
        letter_width=float(params.get("letter_width", 1.0)),
        proofread=bool(params.get("proofread", False)),
        scan_look=bool(params.get("scan_look", False)),
        ink_texture=float(params.get("ink_texture", 0.4)),
        show_through=bool(params.get("show_through", False)),
        show_through_strength=float(params.get("show_through_strength", 0.5)),
        dpi=int(params.get("dpi", 300)),
        letter_spacing=float(params.get("letter_spacing", 1.0)),
        letter_spacing_jitter=(
            -abs(float(params.get("letter_spacing_jitter", 2.0))),
            abs(float(params.get("letter_spacing_jitter", 2.0))),
        ),
        word_spacing=(
            float(params.get("word_space_min", 10.0)),
            float(params.get("word_space_max", 18.0)),
        ),
        line_gap_jitter=(
            -abs(float(params.get("line_gap_jitter", 7.0))),
            abs(float(params.get("line_gap_jitter", 7.0))),
        ),
        margin_left_jitter=(
            -abs(float(params.get("margin_jitter", 4.0))),
            abs(float(params.get("margin_jitter", 4.0))),
        ),
        baseline_jitter=(
            -abs(float(params.get("baseline_jitter", 3.0))),
            abs(float(params.get("baseline_jitter", 3.0))),
        ),
        slope_range=(
            -abs(float(params.get("line_slope", 0.015))),
            abs(float(params.get("line_slope", 0.015))),
        ),
        wave_amplitude=float(params.get("wave_amplitude", 2.5)),
        rotation_range=(
            -abs(float(params.get("rotation_range", 3.5))),
            abs(float(params.get("rotation_range", 3.5))),
        ),
        opacity_range=(
            params.get("opacity_min", 180),
            params.get("opacity_max", 255),
        ),
        ink_color=tuple(params.get("ink_color", [20, 20, 22])),
        word_level_baseline=params.get("word_level_baseline", True),
        word_jitter_y=(
            -abs(float(params.get("word_jitter_y", 3.0))),
            abs(float(params.get("word_jitter_y", 3.0))),
        ),
        # Subtle drift (pattern-breaking)
        fatigue_enabled=params.get("fatigue_enabled", True),
        fatigue_rate=params.get("fatigue_rate", 0.002),
        fatigue_max=params.get("fatigue_max", 1.4),
        fatigue_paragraph_recovery=params.get("fatigue_paragraph_recovery", 0.5),
        fatigue_line_recovery=params.get("fatigue_line_recovery", 0.15),
        fatigue_burst_probability=params.get("fatigue_burst_probability", 0.02),
        glyph_bank=bank_path(params.get("glyph_bank")),
        handwriting_model=bool(params.get("handwriting_model")) and bank_path(params.get("glyph_bank")) is not None,
        slant_deg=params.get("slant_deg"),
    )
    return cfg


def font_paths(cfg: EngineConfig, given: Optional[List[str]] = None) -> List[str]:
    """The fonts to use: given ones or the bundled ones, minus those switched off in
    the bank manager (fonts.json)."""
    paths = list(given or [])
    if not paths:
        paths = [str(p) for p in BUNDLED_FONTS_DIR.glob("*.ttf")] + [str(p) for p in BUNDLED_FONTS_DIR.glob("*.otf")]
    if cfg.glyph_bank and (Path(cfg.glyph_bank) / "fonts.json").exists():
        off = set(json.loads((Path(cfg.glyph_bank) / "fonts.json").read_text(encoding="utf-8")).get("disabled", []))
        paths = [p for p in paths if Path(p).name not in off] or paths
    return paths
