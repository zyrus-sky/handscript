"""
"Scanned look" for finished pages: off-white paper with uneven tone and grain, a
little ink bleed, a tiny page rotation and soft shading like a scanner lid casts.
Meant for documents that should look scanned / photographed; leave it off when
printing on real paper (the paper is already there).
"""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def scanify(page: Image.Image, strength: float = 1.0, rng: Optional[random.Random] = None) -> Image.Image:
    rng = rng or random
    rgb = np.array(page.convert("RGB"), np.float32)
    H, W = rgb.shape[:2]
    gray = rgb.mean(2)
    ink = np.clip((245.0 - gray) / 200.0, 0, 1)                    # 1 = full ink

    # ink bleeds a touch into the paper fibres
    bleed = cv2.GaussianBlur(ink, (0, 0), 0.7 + 0.4 * strength)
    ink = np.maximum(ink, 0.85 * bleed)

    # paper: warm off-white, slow tone variation + fine grain
    paper = np.array([244, 242, 235], np.float32) + rng.uniform(-3, 3)
    low = cv2.resize(np.random.normal(0, 1, (8, 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    grain = np.random.normal(0, 1, (H, W)).astype(np.float32)
    tone = 1.0 + strength * (0.012 * low + 0.008 * grain)
    # soft shading: one side slightly darker, like a scanner lid / page curve
    side = rng.choice([-1, 1])
    xs = np.linspace(-1, 1, W, dtype=np.float32)[None, :] * side
    shade = 1.0 - strength * 0.035 * np.clip(xs, 0, 1) ** 2
    bgm = paper[None, None, :] * (tone * shade)[:, :, None]

    # the page's own ink colour (its darkest strokes), mixed over the paper by ink amount
    solid = ink > np.percentile(ink[ink > 0.05], 90)
    ink_col = np.median(rgb[solid], axis=0)
    out = bgm * (1 - ink[:, :, None]) + ink_col[None, None, :] * ink[:, :, None]

    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    # tiny rotation and offset, gaps filled with paper
    angle = rng.uniform(-0.35, 0.35) * strength
    fill = tuple(int(c) for c in paper)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill,
                     translate=(rng.uniform(-4, 4) * strength, rng.uniform(-4, 4) * strength))
    return img.convert("RGBA")


def ink_texture(page: Image.Image, amount: float = 0.5, strip: int = 512) -> Image.Image:
    """Ballpoint texture (see _ink_texture), done in horizontal strips: a full page at
    once needs ~1 GB of temporary arrays, which ran the server out of memory when
    several pages were textured in parallel. The texture is local, so strips with a
    small overlap give the same result."""
    W, H = page.size
    flow, patch = _lowfreq(W, H, 4, 3.0), _lowfreq(W, H, 8, 1.5)
    if H <= strip + 64:
        return _ink_texture(page, amount, flow, patch)
    out = page.convert("RGBA").copy()
    m = 24
    for y in range(0, H, strip):
        a, b = max(0, y - m), min(H, y + strip + m)
        part = page.crop((0, a, W, b))
        if part.convert("L").getextrema()[0] > 245:          # blank strip: nothing to do
            continue
        done = _ink_texture(part, amount, flow[a:b], patch[a:b])
        out.paste(done.crop((0, y - a, W, min(H, y + strip) - a)), (0, y))
    return out


def _lowfreq(W: int, H: int, step: int, sigma: float) -> np.ndarray:
    """Smooth random field over the whole page (zero mean, unit spread)."""
    f = cv2.GaussianBlur(np.random.rand(H // step + 1, W // step + 1).astype(np.float32), (0, 0), sigma)
    f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
    return (f - f.mean()) / (f.std() + 1e-6)


def _ink_texture(page: Image.Image, amount: float = 0.5, flow=None, patch=None) -> Image.Image:
    """Ballpoint ink as it looks on paper (forensic-document descriptions): oil-based
    paste sits on the surface (no feathering); fine pale striations run ALONG the
    stroke where the ball skips; small blobs ("gooping") at stroke ends; downstrokes
    heavier than upstrokes; light strokes speckled where the ball misses the valleys
    of the paper fibres. Stroke directions come from the ink itself (structure tensor)."""
    if amount <= 0:
        return page
    rgb = np.array(page.convert("RGB"), np.float32)
    H, W = rgb.shape[:2]
    ink = np.clip((250.0 - rgb.mean(2)) / 220.0, 0, 1)
    if not (ink > 0.05).any():
        return page
    solid = ink > np.percentile(ink[ink > 0.05], 90)
    ink_col = np.median(rgb[solid], axis=0)

    # stroke direction field: the stroke runs perpendicular to the ink gradient
    sm = cv2.GaussianBlur(ink, (0, 0), 1.5)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    j11 = cv2.GaussianBlur(gx * gx, (0, 0), 2.5)
    j22 = cv2.GaussianBlur(gy * gy, (0, 0), 2.5)
    j12 = cv2.GaussianBlur(gx * gy, (0, 0), 2.5)
    theta = 0.5 * np.arctan2(2 * j12, j11 - j22) + np.pi / 2          # tangent angle
    tx, ty = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)

    # striations: fine noise smeared far along the stroke direction -> thin tracks
    noise = np.random.rand(H, W).astype(np.float32)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    acc, n = noise.copy(), 1
    for step in (-9, -6, -3, 3, 6, 9):
        acc += cv2.remap(noise, xs + step * tx, ys + step * ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        n += 1
    streak = acc / n
    streak = (streak - streak.mean()) / (streak.std() + 1e-6)
    striation = np.clip((streak - 1.0) / 0.8, 0, 1)

    # only the inside of the line skips; the edges of a ballpoint line stay crisp
    core = cv2.distanceTransform((ink > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    inner = np.clip((core - 1.0) / 1.5, 0, 1)

    # pressure by direction: (near-)vertical strokes are the heavy downstrokes
    vert = np.abs(ty)
    pressure = 0.96 + 0.10 * vert

    # paper tooth: the ball misses fibre valleys on light (sideways/upward) strokes
    tooth = cv2.GaussianBlur(np.random.rand(H, W).astype(np.float32), (0, 0), 0.7)
    tooth = np.clip((tooth - 0.62) * 5.0, 0, 1)

    # ink flow: the line goes lighter and darker over a few letters (photo of the notebook)
    if flow is None or flow.shape != (H, W):
        flow = _lowfreq(W, H, 4, 3.0)
    pressure = pressure * (1 - 0.22 * amount * np.clip(flow, 0, 2) / 2)

    loss = amount * inner * (0.45 * striation + 0.25 * tooth * (1 - vert))
    density = pressure * (1 - np.clip(loss, 0, 0.6))

    # (landing blobs, lighter lift-offs and darker crossings come with the strokes
    # themselves now - engine._ink_amount / glyphbank.render_strokes)

    # the "aura": ballpoint dye creeping a little into the paper beside the line -
    # faint, leaning violet, and patchy (along some edges, not others), as in photos
    # of the writer's real pen next to a print
    on = (ink > 0.5).astype(np.float32)
    wpx = 2.0 * float(np.percentile(core[core > 0], 90)) if (core > 0).any() else 3.0
    halo = np.clip(cv2.GaussianBlur(on, (0, 0), max(0.6, 0.35 * wpx)) - ink, 0, 1)
    if patch is None or patch.shape != (H, W):
        patch = _lowfreq(W, H, 8, 1.5)
    halo = halo * np.clip(0.45 + 0.6 * patch, 0, 1) * 0.55 * amount
    tint = np.clip(ink_col + (255 - ink_col) * 0.55 + np.array([12, -6, 4], np.float32), 0, 255)

    # scale the existing darkness (keeps the pen colour and anti-aliased edges)
    dark = (255.0 - rgb) * np.clip(density, 0, 1.12)[..., None]
    out = 255.0 - dark
    out = out * (1 - halo[..., None]) + tint * halo[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")


def show_through(front: Image.Image, back: Image.Image, strength: float = 0.5) -> Image.Image:
    """The writing on the other side of the sheet faintly visible through the paper:
    the back page mirrored, softened and much lighter, under the front's own ink."""
    if strength <= 0:
        return front
    f = np.array(front.convert("RGB"), np.float32)
    b = np.array(back.convert("L").resize(front.size), np.float32)
    ink_b = np.clip((250.0 - b) / 220.0, 0, 1)[:, ::-1]                # seen from behind
    # paper diffuses the back side's strokes a little but keeps their shapes readable
    ink_b = cv2.GaussianBlur(ink_b, (0, 0), 1.0 + 0.8 * strength)
    a = np.clip(ink_b * 0.5 * strength, 0, 0.45)[..., None]
    ghost = np.array([125, 125, 135], np.float32)                      # greyish through paper
    f_ink = np.clip((250.0 - f.mean(2)) / 220.0, 0, 1)[..., None]
    mix = f * (1 - a * (1 - f_ink)) + ghost * a * (1 - f_ink)            # only where the front is paper
    return Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8)).convert("RGBA")
