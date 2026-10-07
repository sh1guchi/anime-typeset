"""Text placed on a plane in space (a sign at an angle, a board, a wall): the Russian text is laid into a
quadrilateral - the original text's outline in the frame - with an exact perspective map.

Why a drawing and not \\frx/\\fry: libass puts its 3D camera ~312 px from the screen *in the player's video
(storage) resolution*: the same \\frx\\fry line looks different on a 4K file, its 1080p web rendition and our
1080p check renders. Here the glyph outlines come from the font file (fontTools), every point goes through the
homography rectangle -> quad, and the result is a \\p drawing in PlayRes units: identical everywhere. The
style's colours, outline, shadow and blur apply to drawings as to text, so the preset layers stay as they are.

quad: [[x,y] top-left, top-right, bottom-right, bottom-left] in analysis px - the plane rectangle the original
text sits in (its ink box). The Russian text is set in that plane: baseline, cap height and centre the way the
place signs / cards are set (cap ~0.78 em of the kanji em), squeezed if wider than the plane allows."""
import numpy as np, cv2
from fontTools.ttLib import TTFont, TTCollection
from fontTools.pens.basePen import BasePen
from . import coords, fonts
from .coords import f2


class _Pen(BasePen):
    """outline -> [contour: [("m"|"l"|"b", [(x,y)...])]] in font units, quadratics as cubics"""
    def __init__(self, gs):
        super().__init__(gs)
        self.contours, self.cur, self.last = [], None, None

    def _moveTo(self, p):
        self.cur = [("m", [p])]; self.contours.append(self.cur); self.last = p

    def _lineTo(self, p):
        self.cur.append(("l", [p])); self.last = p

    def _curveToOne(self, a, b, c):
        self.cur.append(("b", [a, b, c])); self.last = c

    def _qCurveToOne(self, a, b):
        p0 = self.last
        c1 = (p0[0] + 2 / 3 * (a[0] - p0[0]), p0[1] + 2 / 3 * (a[1] - p0[1]))
        c2 = (b[0] + 2 / 3 * (a[0] - b[0]), b[1] + 2 / 3 * (a[1] - b[1]))
        self._curveToOne(c1, c2, b)

    def _closePath(self):
        pass


_fonts = {}


def _font(rec):
    k = (rec["path"], rec["index"])
    if k not in _fonts:
        p = rec["path"]
        ft = TTCollection(p).fonts[rec["index"]] if p.lower().endswith(".ttc") else TTFont(p)
        _fonts[k] = (ft, ft.getGlyphSet(), ft.getBestCmap() or {}, ft["hmtx"], ft["head"].unitsPerEm)
    return _fonts[k]


def layout(rec, text, em, spacing=0.0):
    """contours of `text` (\\N = new line, centred lines) in px at em size `em`: x from 0, y down, first
    baseline at 0; returns (contours, widths of lines, line advance)"""
    ft, gs, cmap, hmtx, upm = _font(rec)
    met = fonts.metrics(rec)
    s = em / upm
    adv_line = em * met["ratio"]                        # libass line height = the font size (winA + winD)
    lines = text.split("\\N")
    widths = []
    for ln in lines:
        w = 0.0
        for ch in ln:
            g = cmap.get(ord(ch))
            w += (hmtx[g][0] if g else upm / 2) * s + spacing
        widths.append(w - (spacing if ln else 0))
    W = max(widths) if widths else 0
    out = []
    for li, ln in enumerate(lines):
        x = (W - widths[li]) / 2
        y0 = li * adv_line
        for ch in ln:
            g = cmap.get(ord(ch))
            if g is None:
                x += upm / 2 * s + spacing; continue
            pen = _Pen(gs)
            gs[g].draw(pen)
            for c in pen.contours:
                out.append([(op, [(x + px * s, y0 - py * s) for px, py in pts]) for op, pts in c])
            x += hmtx[g][0] * s + spacing
    return out, widths, adv_line


def homography(src, dst):
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))


def _apply(H, pts):
    p = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def plane_size(quad):
    q = np.asarray(quad, np.float64)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return w, h


def fit(rec, text, quad, em=None, cap=0.78, base=None, maxw=0.96, spacing=0.0, align="center", min_squeeze=0.85):
    """the text box inside the plane of `quad`: returns (em, H text px -> analysis px, contours, info)
    em: em size in plane px (default: cap * the kanji em, the kanji em = quad height / 0.92);
    base: baseline at this fraction of the quad height (default: the capital letters centred in the plane);
    maxw: max width share - a wider text is squeezed down to min_squeeze, then set smaller."""
    Wp, Hp = plane_size(quad)
    em = em or cap * Hp / 0.92
    met = fonts.metrics(rec)
    for _ in range(3):
        cont, widths, adv = layout(rec, text, em, spacing)
        tw = max(widths) if widths else 1.0
        sq = min(1.0, maxw * Wp / max(tw, 1e-6))
        if sq >= min_squeeze:
            break
        em *= sq / min_squeeze
    n = len(widths)
    x_off = {"left": (1 - maxw) / 2 * Wp, "right": Wp - (1 - maxw) / 2 * Wp - tw * sq}.get(align, (Wp - tw * sq) / 2)
    capk = met["cap"] * em                                  # capital height, px
    block = capk + (n - 1) * adv                            # first cap top -> last baseline
    y_base_last = base * Hp if base is not None else (Hp + block) / 2
    y0 = y_base_last - (n - 1) * adv                        # first baseline
    # text px (x from 0, first baseline 0) -> plane px -> quad (analysis px)
    A = np.array([[sq, 0, x_off], [0, 1, y0], [0, 0, 1]], np.float64)
    Hq = homography([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]], quad)
    return em, Hq @ A, cont, dict(squeeze=sq, plane=(Wp, Hp), width=tw, em=em)


def drawing(contours, H, prec=8):
    """'m x y l x y b ...' of the mapped contours in PlayRes units; \\p3 -> coordinates in 1/4 px (prec 4),
    \\p4 -> 1/8 px (prec 8). Long curves are split so the projective map bends them right."""
    K = coords.K
    out = []
    for c in contours:
        pts_all = []
        for op, pts in c:
            pts_all += pts
        P = _apply(H, pts_all) * K * prec
        k = 0
        for op, pts in c:
            q = P[k:k + len(pts)]; k += len(pts)
            out.append(op + " " + " ".join(f"{int(round(x))} {int(round(y))}" for x, y in q))
    return " ".join(out)


def pbits(prec):
    return {1: 1, 2: 2, 4: 3, 8: 4}[prec]


def emit(ctx, spec, f0, f1, layers, actor="Надпись", tags=""):
    """lines of a text spec with "quad": every layer as a drawing of the mapped glyph outlines"""
    main = spec.get("style") or layers[-1]["style"]
    rec, met = ctx.styles.font(main)
    text = spec["text"]
    miss = fonts.missing_glyphs(rec, text.replace("\\N", ""))
    if miss:
        print(f"    warning: style '{main}' font lacks glyphs {''.join(miss)}", flush=True)
    em, H, cont, info = fit(rec, text, spec["quad"], em=spec.get("em"), cap=spec.get("cap", 0.78),
                            base=spec.get("base_frac"), maxw=spec.get("maxw_frac", 0.96),
                            spacing=spec.get("spacing", 0.0), align=spec.get("align", "center"))
    prec = int(spec.get("prec", 8))
    d = drawing(cont, H, prec)
    st, en = ctx.video.atime(f0), ctx.video.atime(f1 + 1)
    out = []
    for L in layers:
        ox, oy = L.get("offset", [0, 0])
        K = coords.K
        out.append(f"Dialogue: {L['layer']},{st},{en},{L['style']},{actor},0,0,0,,"
                   f"{{{tags}\\an7\\pos({f2(ox * K)},{f2(oy * K)}){L.get('tags', '')}{spec.get('tags', '')}"
                   f"\\p{pbits(prec)}}}{d}")
    if info["squeeze"] < 0.999:
        print(f"    quad text: {info['squeeze'] * 100:.0f}% width, em {info['em']:.0f} px to fit the plane", flush=True)
    return out, (em, 100)          # the squeeze is in the drawing already: callers must not scale it again


# ---------------------------------------------------------------- the original text's quad
def _trimmed_line(xs, ys, keep=0.7, iters=4):
    """y = a x + b: least squares, then refit on the best `keep` share of the points (outliers: short glyphs)"""
    sel = np.ones(len(xs), bool)
    a, b = np.polyfit(xs, ys, 1)
    for _ in range(iters):
        r = np.abs(a * xs + b - ys)
        thr = np.quantile(r, keep)
        sel = r <= thr + 1e-9
        if sel.sum() < 3:
            break
        a, b = np.polyfit(xs[sel], ys[sel], 1)
    return a, b, float(np.abs(a * xs[sel] + b - ys[sel]).mean()) if sel.any() else 0.0


def auto_quad(mask, x0=0, y0=0, keep=0.7, perspective=False, snap_deg=2.0):
    """quad of one line of text from its glyph mask (box offset x0, y0). The line is cut into windows half a
    glyph wide; the highest and the lowest ink point of each window (tall glyphs reach the envelope, short ones
    like 'ー' are dropped by a trimmed fit) give the top and bottom lines - converging with perspective, tilted
    with rotation. The ends: perpendicular to the mean line direction, through the first / last ink.
    Returns ([[x,y]]*4 TL TR BR BL, info)."""
    m = mask.astype(bool)
    ys_all, xs_all = np.nonzero(m)
    if len(xs_all) < 30:
        raise SystemExit("too little text in the box for a quad")
    # a rough direction first (PCA of the ink): windows are cut along it, not along x
    pts = np.c_[xs_all, ys_all].astype(np.float64)
    c = pts.mean(axis=0)
    u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0] if vt[0][0] >= 0 else -vt[0]
    nrm = np.array([-d[1], d[0]])                     # points "down" the text when d points right
    if nrm[1] < 0:
        nrm = -nrm
    t = (pts - c) @ d; s_ = (pts - c) @ nrm
    hgt = np.quantile(s_, 0.98) - np.quantile(s_, 0.02)
    win = max(4.0, 0.5 * hgt)
    edges = np.arange(t.min(), t.max() + win, win)
    T, TOP, BOT = [], [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        sel = (t >= a_) & (t < b_)
        if sel.sum() < 5:
            continue
        T.append((a_ + b_) / 2); TOP.append(s_[sel].min()); BOT.append(s_[sel].max())
    T, TOP, BOT = map(np.asarray, (T, TOP, BOT))
    if len(T) < 3:
        raise SystemExit("text line too short for a quad")
    at, bt, et = _trimmed_line(T, TOP, keep)
    ab, bb, eb = _trimmed_line(T, BOT, keep)
    t0, t1 = t.min(), t.max()
    if not perspective:
        # one line of kana/kanji has a ragged bottom (and top): converging lines from it are guesses. Default:
        # a rotated rectangle - the direction of the cleaner envelope, the extents from the ink itself.
        # A tilt under snap_deg is glyph-height noise (a tall 城 at one end), not a rotated sign: horizontal
        a = at if et <= eb else ab
        if abs(np.degrees(np.arctan2(d[1] + nrm[1] * a, d[0] + nrm[0] * a))) < snap_deg:
            ang0 = np.arctan2(d[1], d[0])
            a = -np.tan(ang0) if abs(np.cos(ang0)) > 1e-6 else 0.0          # undo the PCA tilt: exactly level
        rot = np.array([[1, a], [-a, 1]]) / np.hypot(1, a)          # re-express (t, s) along the slope a
        d2 = (d * 1.0 + nrm * a) / np.hypot(1, a)
        n2 = np.array([-d2[1], d2[0]])
        if n2 @ nrm < 0:
            n2 = -n2
        d, nrm = d2, n2
        t = (pts - c) @ d; s_ = (pts - c) @ nrm
        t0, t1 = t.min(), t.max()
        top, bot = np.quantile(s_, 0.01), np.quantile(s_, 0.99)
        at = ab = 0.0; bt, bb = top, bot
    # corners in the (t, s) frame -> image
    def img(tt, ss):
        p = c + tt * d + ss * nrm
        return [float(p[0]) + x0, float(p[1]) + y0]
    # ends perpendicular to the mean line: the end segment of each line at t0 / t1 (in a rotated frame the
    # perpendicular of the mean direction is the t axis' normal - straight in t)
    q = [img(t0, at * t0 + bt), img(t1, at * t1 + bt), img(t1, ab * t1 + bb), img(t0, ab * t0 + bb)]
    ang = np.degrees(np.arctan2(d[1], d[0]))
    return q, dict(angle=float(ang), top_slope=float(at), bottom_slope=float(ab), top_err=et, bottom_err=eb,
                   top_fit=keep, bottom_fit=keep, perspective=perspective)
