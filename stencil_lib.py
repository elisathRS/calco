"""Calco: tattoo stencil engine built only with OpenCV + NumPy.

Each function maps to a concept from the course notebook:
  - adjust_light     -> image arithmetic: cv2.multiply, cv2.add, cv2.subtract and clipping
  - apply_threshold  -> thresholding: global, inverse, adaptive (mean / Gaussian) and Otsu
  - ink_mask         -> cv2.bitwise_not so that ink is always 255
  - make_rgba        -> alpha channel: cv2.split + cv2.merge([b, g, r, alpha])
  - build_sheet      -> drawing and annotation: cv2.line, cv2.circle, cv2.rectangle, cv2.putText
  - detect_objects   -> (bonus) YOLO
"""
from __future__ import annotations

import io
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Constants (colours are BGR, as OpenCV expects)
# ----------------------------------------------------------------------------
INK_BGR = {
    "Violet (classic stencil)": (154, 42, 75),
    "Black": (17, 17, 17),
    "Blue": (156, 63, 31),
    "Red": (31, 32, 179),
}
SKIN_BGR = {
    "Light skin": ((198, 220, 246), (159, 191, 233)),
    "Medium skin": ((124, 164, 217), (92, 133, 189)),
    "Dark skin": ((41, 67, 110), (25, 45, 79)),
}
PREVIEW_BACKGROUNDS = ["Transparent", "Paper", "Light skin", "Medium skin", "Dark skin"]
METHODS = {
    "Global": "global",
    "Global inverse": "global_inv",
    "Adaptive (mean)": "adaptive_mean",
    "Adaptive (Gaussian)": "adaptive_gauss",
    "Otsu (automatic)": "otsu",
}


@dataclass
class Params:
    contrast: float = 1.3
    brightness: int = 0
    blur: float = 1.0
    method: str = "adaptive_gauss"
    value: int = 127          # threshold for the global methods
    block: int = 31           # neighbourhood size for the adaptive methods
    C: int = 8                # constant subtracted by the adaptive methods
    clean: int = 30           # minimum area (px) of a blob to keep
    weight: int = 0           # > 0 thickens lines, < 0 thins them
    shade: int = 35           # 0-100, dot shading strength
    ink_bgr: tuple = (154, 42, 75)


# ----------------------------------------------------------------------------
# 1) Image arithmetic
# ----------------------------------------------------------------------------
def adjust_light(img_bgr: np.ndarray, contrast: float, brightness: int):
    """Contrast with cv2.multiply, brightness with cv2.add / cv2.subtract.

    Returns the uint8 image and two clipping fractions: lost shadows (values
    pushed below 0) and blown highlights (values pushed above 255).
    """
    factor = np.ones(img_bgr.shape, dtype="float64") * float(contrast)
    mult = cv2.multiply(np.float64(img_bgr), factor)
    total = mult + float(brightness)
    clipped = (float(np.mean(total < 0)), float(np.mean(total > 255)))

    out = np.uint8(np.clip(mult, 0, 255))
    b = int(brightness)
    if b != 0:
        shift = np.ones(img_bgr.shape, dtype="uint8") * abs(b)
        out = cv2.add(out, shift) if b > 0 else cv2.subtract(out, shift)
    return out, clipped


# ----------------------------------------------------------------------------
# 2) Thresholding
# ----------------------------------------------------------------------------
def apply_threshold(gray: np.ndarray, method: str, value=127, block=31, C=8):
    """Return (binary, already_inverted, threshold_used).

    `already_inverted` is True when OpenCV already put the INK at 255
    (THRESH_BINARY_INV). With every other method the paper is 255.
    """
    block = max(3, int(block)) | 1  # odd and >= 3
    if method == "global":
        t, b = cv2.threshold(gray, int(value), 255, cv2.THRESH_BINARY)
        return b, False, t
    if method == "global_inv":
        t, b = cv2.threshold(gray, int(value), 255, cv2.THRESH_BINARY_INV)
        return b, True, t
    if method == "adaptive_mean":
        b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, int(C))
        return b, False, None
    if method == "adaptive_gauss":
        b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(C))
        return b, False, None
    if method == "otsu":
        t, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b, False, t
    raise ValueError(f"Unknown method: {method}")


def ink_mask(binary: np.ndarray, already_inverted: bool) -> np.ndarray:
    """Ink mask: 255 wherever a line goes. Uses bitwise_not when needed."""
    return binary if already_inverted else cv2.bitwise_not(binary)


# ----------------------------------------------------------------------------
# 3) Mask cleanup
# ----------------------------------------------------------------------------
def remove_specks(mask: np.ndarray, min_area: int):
    """Remove connected blobs smaller than `min_area` pixels."""
    if min_area <= 0:
        return mask, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    out = np.where(keep[labels], 255, 0).astype(np.uint8)
    removed = int((n - 1) - keep.sum())
    return out, removed


def set_weight(mask: np.ndarray, weight: int):
    """Thicken (dilate) or thin (erode) the lines."""
    if weight == 0:
        return mask
    r = abs(int(weight))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, kernel) if weight > 0 else cv2.erode(mask, kernel)


def _bayer(n=3):
    m = np.array([[0]])
    for _ in range(n):
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    size = m.shape[0]
    return (m + 0.5) / (size * size)


_BAYER = _bayer(3)


def shade_dots(gray: np.ndarray, ink: np.ndarray, strength: int):
    """Turn mid-tones into dots (ordered dithering) without touching the lines."""
    if strength <= 0:
        return ink
    h, w = gray.shape
    cell = max(2, round(min(h, w) / 380))
    tone = (255 - gray.astype(np.float32)) / 255.0
    level = np.clip((tone - 0.18) / 0.82, 0, 1)
    ys, xs = np.indices((h, w))
    thr = _BAYER[(ys // cell) % 8, (xs // cell) % 8]
    dots = (level * (strength / 100.0) * 1.3 > thr) & (ink == 0)
    out = ink.copy()
    out[dots] = 255
    return out


# ----------------------------------------------------------------------------
# 4) Alpha channel
# ----------------------------------------------------------------------------
def make_rgba(mask: np.ndarray, ink_bgr: tuple, soften: float = 0.7) -> np.ndarray:
    """Single-colour ink plus the mask as the alpha channel (BGRA)."""
    alpha = cv2.GaussianBlur(mask, (0, 0), soften) if soften > 0 else mask
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[:] = ink_bgr
    b, g, r = cv2.split(color)
    return cv2.merge([b, g, r, alpha])


def process(img_bgr: np.ndarray, p: Params) -> dict:
    """Run the whole pipeline and also return the intermediate steps."""
    adj, clipped = adjust_light(img_bgr, p.contrast, p.brightness)
    gray = cv2.cvtColor(adj, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.GaussianBlur(gray, (0, 0), p.blur) if p.blur > 0 else gray

    binary, inverted, used_t = apply_threshold(gray_s, p.method, p.value, p.block, p.C)
    ink = ink_mask(binary, inverted)
    ink, removed = remove_specks(ink, p.clean)
    ink = set_weight(ink, p.weight)
    ink = shade_dots(gray_s, ink, p.shade)

    rgba = make_rgba(ink, p.ink_bgr)
    return {
        "adjusted": adj,
        "gray": gray_s,
        "binary": binary,
        "ink": ink,
        "rgba": rgba,
        "stats": {
            "clipped_low": clipped[0],
            "clipped_high": clipped[1],
            "removed": removed,
            "threshold": used_t,
            "coverage": float(np.count_nonzero(ink)) / ink.size,
        },
    }


# ----------------------------------------------------------------------------
# 5) Print sheet: mirror, marks, ruler, box and label (drawing and annotation)
# ----------------------------------------------------------------------------
def ascii_text(text: str) -> str:
    """cv2.putText cannot draw accents or non-Latin letters, so fold them to ASCII."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _text(img, txt, org, scale, color, thickness, anchor="left"):
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    if anchor == "center":
        x -= tw // 2
    elif anchor == "right":
        x -= tw
    cv2.putText(img, txt, (int(x), int(y + th // 2)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def build_sheet(rgba, width_cm, ink_bgr, mirror=False, marks=True, ruler=True, box=True, label=""):
    """Return (BGRA sheet, pixels per cm)."""
    design = cv2.flip(rgba, 1) if mirror else rgba
    h, w = design.shape[:2]
    px_cm = w / float(width_cm)
    label = ascii_text(label).strip()
    any_extra = marks or ruler or box or bool(label)
    m = int(round(max(w, h) * 0.075)) if any_extra else 0   # margin around the design
    sheet = cv2.copyMakeBorder(design, m, m, m, m, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)) if m else design.copy()
    H, W = sheet.shape[:2]
    color = (int(ink_bgr[0]), int(ink_bgr[1]), int(ink_bgr[2]), 255)
    th = max(1, round(max(w, h) / 700))
    fscale = max(0.4, m / 90.0)
    ftk = max(1, int(fscale * 1.6))

    if box:
        pts = cv2.findNonZero((design[:, :, 3] > 40).astype(np.uint8))
        if pts is not None:
            x, y, bw, bh = cv2.boundingRect(pts)
            pad = max(4, th * 3)
            cv2.rectangle(sheet, (x + m - pad, y + m - pad), (x + m + bw + pad, y + m + bh + pad), color, th, cv2.LINE_AA)
            _text(sheet, f"{bw / px_cm:.1f} x {bh / px_cm:.1f} cm", (x + m, y + m - pad - int(fscale * 24)), fscale * 0.8, color, ftk)
    if marks:
        cs = int(m * 0.34)
        for (cx, cy) in [(m // 2, m // 2), (W - m // 2, m // 2), (m // 2, H - m // 2), (W - m // 2, H - m // 2)]:
            cv2.line(sheet, (cx - cs, cy), (cx + cs, cy), color, th, cv2.LINE_AA)
            cv2.line(sheet, (cx, cy - cs), (cx, cy + cs), color, th, cv2.LINE_AA)
            cv2.circle(sheet, (cx, cy), int(cs * 0.45), color, th, cv2.LINE_AA)
    if ruler:
        bar = 5 if width_cm >= 12 else 2 if width_cm >= 7 else 1
        y0 = H - m // 2
        x0 = m
        cv2.line(sheet, (x0, y0), (int(x0 + bar * px_cm), y0), color, th, cv2.LINE_AA)
        for i in range(bar + 1):
            tick = int(m * (0.16 if i in (0, bar) else 0.1))
            xx = int(x0 + i * px_cm)
            cv2.line(sheet, (xx, y0 - tick), (xx, y0), color, th, cv2.LINE_AA)
        _text(sheet, f"{bar} cm", (x0 + bar * px_cm + int(fscale * 14), y0 - int(fscale * 4)), fscale * 0.8, color, ftk)
    if marks or ruler or box:
        _text(sheet, f"{width_cm:.0f} x {h / px_cm:.1f} cm", (W // 2, m // 2), fscale * 0.8, color, ftk, "center")
    if label:
        _text(sheet, label, (W - m, H - m // 2 - int(fscale * 4)), fscale * 0.8, color, ftk, "right")
    return sheet, px_cm


# ----------------------------------------------------------------------------
# 6) Preview and export
# ----------------------------------------------------------------------------
def _checker(h, w, size=20):
    ys, xs = np.indices((h, w))
    c = (((ys // size) + (xs // size)) % 2).astype(np.float32)
    g = 255 - c * 40
    return np.dstack([g, g, g])


def _background(h, w, name):
    if name == "Paper":
        return np.full((h, w, 3), 255, dtype=np.float32)
    c0, c1 = SKIN_BGR[name]
    ys, xs = np.indices((h, w))
    t = ((xs / max(1, w)) + (ys / max(1, h))) / 2.0
    t = t[..., None].astype(np.float32)
    return np.array(c0, dtype=np.float32) * (1 - t) + np.array(c1, dtype=np.float32) * t


def preview(sheet_bgra: np.ndarray, bg_name: str) -> np.ndarray:
    """Composite for on-screen display (BGR)."""
    h, w = sheet_bgra.shape[:2]
    a = sheet_bgra[:, :, 3:4].astype(np.float32) / 255.0
    ink = sheet_bgra[:, :, :3].astype(np.float32)
    if bg_name == "Transparent":
        out = ink * a + _checker(h, w) * (1 - a)
    else:
        bg = _background(h, w, bg_name)
        out = bg * (1 - a) + a * (bg * ink / 255.0)   # ink is multiplied onto the background
    return np.clip(out, 0, 255).astype(np.uint8)


def export_png(sheet_bgra: np.ndarray, px_cm: float, white_background=False, dpi=300) -> bytes:
    """Real-size PNG: resample to `dpi` and store the DPI in the file."""
    from PIL import Image

    h, w = sheet_bgra.shape[:2]
    total_cm = w / px_cm
    target_w = int(min(6000, max(300, round(total_cm * dpi / 2.54))))
    scale = target_w / w
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    big = cv2.resize(sheet_bgra, (target_w, int(round(h * scale))), interpolation=interp)
    rgba = cv2.cvtColor(big, cv2.COLOR_BGRA2RGBA)
    img = Image.fromarray(rgba, "RGBA")
    if white_background:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


# ----------------------------------------------------------------------------
# 7) Input: decode, downscale and generate a sample (drawn with cv2)
# ----------------------------------------------------------------------------
def decode_image(data: bytes, max_side: int = 1600):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                       # transparent PNG: flatten onto white
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def sample_image(size: int = 900) -> np.ndarray:
    """A shaded mandala on noisy paper, like a phone photo of a drawing."""
    rng = np.random.default_rng(3)
    c = size // 2
    ys, xs = np.indices((size, size))
    d = np.hypot(xs - c, ys - c) / (size * 0.75)
    paper = 232 - 42 * d**2
    img = np.dstack([paper * 0.90, paper * 0.95, paper]).astype(np.uint8)   # slightly warm paper (BGR)
    dark, line = (40, 42, 45), (28, 27, 29)
    cv2.circle(img, (c, c), int(size * 0.455), line, 7, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(size * 0.43), line, 3, cv2.LINE_AA)
    for i in range(72):
        a = i / 72 * 2 * np.pi
        cv2.circle(img, (int(c + np.cos(a) * size * 0.406), int(c + np.sin(a) * size * 0.406)), 6, dark, -1, cv2.LINE_AA)

    def petals(n, dist, rx, ry, offset, tones):
        for i in range(n):
            a = offset + i * 2 * np.pi / n
            ang = np.degrees(a)
            for k, tone in enumerate(tones):          # stacked ellipses fake a gradient
                shrink = 1 - k * 0.22
                dd = dist + k * ry * 0.09
                ctr = (int(c + np.sin(a) * dd), int(c - np.cos(a) * dd))
                cv2.ellipse(img, ctr, (int(rx * shrink), int(ry * shrink)), ang, 0, 360, (tone, tone, tone), -1, cv2.LINE_AA)
            ctr = (int(c + np.sin(a) * dist), int(c - np.cos(a) * dist))
            cv2.ellipse(img, ctr, (rx, ry), ang, 0, 360, line, 5, cv2.LINE_AA)
            tip = (int(c + np.sin(a) * (dist + ry * 0.65)), int(c - np.cos(a) * (dist + ry * 0.65)))
            base = (int(c + np.sin(a) * (dist - ry * 0.75)), int(c - np.cos(a) * (dist - ry * 0.75)))
            cv2.line(img, base, tip, line, 3, cv2.LINE_AA)

    petals(12, int(size * 0.26), 66, 128, 0.0, (70, 115, 165, 205))
    petals(12, int(size * 0.167), 44, 88, np.pi / 12, (95, 140, 190))
    for r, tone in ((80, 80), (64, 125), (46, 175)):
        cv2.circle(img, (c, c), r, (tone, tone, tone), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), 80, line, 6, cv2.LINE_AA)
    star = []
    for i in range(16):
        a = i / 16 * 2 * np.pi - np.pi / 2
        r = 26 if i % 2 else 66
        star.append([int(c + np.cos(a) * r), int(c + np.sin(a) * r)])
    cv2.polylines(img, [np.array(star, np.int32)], True, line, 4, cv2.LINE_AA)
    cv2.circle(img, (c, c), 14, line, -1, cv2.LINE_AA)
    noise = rng.normal(0, 7, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# 8) Optional bonus: YOLO
# ----------------------------------------------------------------------------
def detect_objects(img_bgr: np.ndarray, model, conf: float = 0.25):
    """Return [(x1, y1, x2, y2, class_name, confidence)] using an ultralytics model."""
    res = model(img_bgr, verbose=False, conf=conf)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        out.append((x1, y1, x2, y2, res.names[int(b.cls)], float(b.conf)))
    return out


def draw_detections(img_bgr: np.ndarray, dets) -> np.ndarray:
    out = img_bgr.copy()
    for x1, y1, x2, y2, name, conf in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"{name} {conf:.0%}", (x1 + 4, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 255), 2, cv2.LINE_AA)
    return out


def union_box(dets, shape, pad=0.05):
    h, w = shape[:2]
    x1 = min(d[0] for d in dets)
    y1 = min(d[1] for d in dets)
    x2 = max(d[2] for d in dets)
    y2 = max(d[3] for d in dets)
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
