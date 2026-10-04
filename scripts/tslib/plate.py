"""Clean plates for moving backgrounds and their encoding as ASS drawings.

Translating background (cards over a pan). Model, screen s, frame f, background shift d_f
(a background point p is seen at s = p + d_f):
    I_f(s) = B(s - d_f) * (1 - a(s)) + C * a(s)
a(s) is the static dark glow around the card's white elements (glow_alpha), C its colour.
B is recovered by a median over frames of (I - C a) / (1 - a), excluding the white elements themselves.
The text region m is then redrawn as B moving with the camera (\\move) plus a static glow layer whose
alpha is a smooth fill of a over m - the Japanese text and its glow disappear, the card's own glow stays.

Zooms (affine): the clean plate P comes from frames before the text appears (or a smooth fill), is
re-projected with the tracked affine per segment (\\move + \\t(\\fscx\\fscy)) and colour-matched over time
with linear per-channel fits (\\t(\\c)).
"""
import warnings
import numpy as np, cv2
from . import coords
from .coords import f2
from .masks import laplace, dilate
from .motion import shift_img, warp_to, warp_from, affine_params
from .draw import (ass_color, rects_from_labels, draw_rects, draw_rects_units, quant_rects, layered_rects,
                   mask_polygon, edge_smooth)


def glow_alpha(Wcard, model):
    """static alpha of the dark glow around white card elements: sum_k w_k * gauss_{sig_k}(dilated mask)"""
    dil = model.get("dil", 1)
    cov = cv2.dilate(Wcard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1,) * 2)).astype(np.float32)
    a = sum(wk * cv2.GaussianBlur(cov, (0, 0), s) for s, wk in zip(model["sig"], model["w"]) if wk)
    return np.clip(a, 0, 0.95)


def fit_glow(med, Wcard, box, base, halo=45):
    """Glow model fitted on this card itself (the preset one was fitted on one sky; on a much lighter or darker
    background its alpha is off and the patch edges show). Background under the halo = smooth fill from
    outside it (fine for skies/gradients/soft bokeh); per-sigma weights and a dark colour C by least squares.
    Returns (model, rms, rms_without_glow); falls back to `base` when the fit does not explain the halo."""
    from .masks import harmonic_fill as hfill, dilate as dil
    x0, y0, x1, y1 = [int(v) for v in box]
    reg = np.zeros(Wcard.shape, bool); reg[y0:y1, x0:x1] = True
    H = dil(Wcard, halo) & reg
    B = hfill(med, H | dil(Wcard, 2), (x0, y0, x1, y1), iters=3000, presmooth=5)
    sel = H & ~dil(Wcard, 2)
    ys, xs = np.where(sel)
    if len(ys) < 500:
        return base, None, None
    if len(ys) > 60000:
        k = np.random.default_rng(0).choice(len(ys), 60000, replace=False); ys, xs = ys[k], xs[k]
    I, Bv = med[ys, xs].astype(np.float64), B[ys, xs].astype(np.float64)
    cov = cv2.dilate(Wcard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(np.float32)
    sig = base.get("sig", [1.5, 3, 6, 12, 24])
    As = [cv2.GaussianBlur(cov, (0, 0), s)[ys, xs].astype(np.float64) for s in sig]
    y = (I - Bv).ravel(order="F")
    rms0 = float(np.sqrt(np.mean(y ** 2)))
    best = None
    for c in range(0, 121, 8):
        D = c - Bv
        X = np.concatenate([np.stack([A * D[:, ch] for A in As], 1) for ch in range(3)], 0)
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = float(np.sqrt(np.mean((X @ w - y) ** 2)))
        if best is None or r < best[0]:
            best = (r, c, w)
    r, c, w = best
    if r > 0.8 * rms0:
        return base, r, rms0
    return {"sig": list(sig), "w": [float(v) for v in w], "C": [c, c, c], "dil": 1}, r, rms0


def estimate_B(strip, sx0, sy0, d, a, C, Wcard, win, excl_dil=3, amax=0.35):
    """Clean moving background over screen window `win` from a frame strip.
    strip: uint8 (N,h,w,3) frames cropped at screen offset (sx0, sy0); d: (N,2) background shift.
    amax: background samples are taken only where the card glow is weaker than this - under strong glow any
    error of the glow model biases the de-glowed value (light/dark blotches that later slide under the patch);
    while the camera moves every point is also seen under weak glow. Points never seen so: smooth fill.
    Returns canvas B in background coords (origin px0, py0)."""
    x0, y0, x1, y1 = win
    N = len(strip); dx, dy = d[:, 0], d[:, 1]
    px0 = int(np.floor(x0 - dx.max())) - 1; px1 = int(np.ceil(x1 - dx.min())) + 1
    py0 = int(np.floor(y0 - dy.max())) - 1; py1 = int(np.ceil(y1 - dy.min())) + 1
    CW, CH = px1 - px0, py1 - py0
    excl = cv2.dilate(Wcard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * excl_dil + 1,) * 2))
    inwin = np.zeros(Wcard.shape, np.uint8); inwin[y0:y1, x0:x1] = 1
    validS = (inwin & (1 - excl)).astype(np.float32)
    if amax is not None and np.abs(d).max() > 4:      # only when the background really moves
        validS *= (a < amax).astype(np.float32)
    sh, sw = strip.shape[1:3]
    a_s = a[sy0:sy0 + sh, sx0:sx0 + sw]; v_s = validS[sy0:sy0 + sh, sx0:sx0 + sw]
    samples = []
    for i in range(N):
        U = (strip[i].astype(np.float32) - C * a_s[..., None]) / (1 - a_s[..., None])
        im = shift_img(U, dx[i], dy[i], CW, CH, px0 - sx0, py0 - sy0)
        v = shift_img(v_s, dx[i], dy[i], CW, CH, px0 - sx0, py0 - sy0) > 0.99
        samples.append((im, v))
    B = np.zeros((CH, CW, 3), np.float32); valid = np.zeros((CH, CW), bool)
    for r0 in range(0, CH, 64):   # median over frames, chunked to bound memory
        r1 = min(CH, r0 + 64)
        stack = np.full((N, r1 - r0, CW, 3), np.nan, np.float32)
        for i, (im, v) in enumerate(samples):
            vv = v[r0:r1]
            stack[i][vv] = im[r0:r1][vv]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns are expected (never uncovered)
            med = np.nanmedian(stack, axis=0)
        ok = ~np.isnan(med[..., 0])
        B[r0:r1][ok] = med[ok]; valid[r0:r1] = ok
    hole = ~valid   # never uncovered (under long strokes): smooth fill
    if hole.any():
        Uh = hole.copy(); Uh[0, :] = Uh[-1, :] = Uh[:, 0] = Uh[:, -1] = False
        B = laplace(B, Uh, 600)
    return dict(B=B, Bvalid=valid, px0=px0, py0=py0, CW=CW, CH=CH)


def observed_alpha(strip, sx0, sy0, d, est, C, win, minsep=30.0):
    """Glow alpha actually seen over the estimated moving background, per screen pixel of `win`: median over
    frames of the projection of (B - F) on (B - C). The glow model is a fit (rms ~10); taking the cloud's
    boundary values from what the frames really show makes the patch meet the real halo without a step.
    NaN where the background is off the canvas or too close to the glow colour."""
    x0, y0, x1, y1 = win
    ww, wh = x1 - x0, y1 - y0
    ones = est["Bvalid"].astype(np.float32)
    vals = np.empty((len(strip), wh, ww), np.float32)
    for i in range(len(strip)):
        Bi = shift_img(est["B"], -d[i, 0], -d[i, 1], ww, wh, x0 - est["px0"], y0 - est["py0"])
        vi = shift_img(ones, -d[i, 0], -d[i, 1], ww, wh, x0 - est["px0"], y0 - est["py0"]) > 0.99
        Fi = strip[i][y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0].astype(np.float32)
        D = Bi - C
        n2 = (D * D).sum(-1)
        ai = ((Bi - Fi) * D).sum(-1) / np.maximum(n2, 1.0)
        ai[(n2 < minsep ** 2) | ~vi] = np.nan
        vals[i] = ai
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        am = np.nanmedian(vals, axis=0)
    out = np.full(coords.shape(), np.nan, np.float32)
    out[y0:y1, x0:x1] = am
    return out


def B_at_frame(est, dxy):
    return shift_img(est["B"], -(est["px0"] + dxy[0]), -(est["py0"] + dxy[1]), coords.AW, coords.AH, 0, 0)


def _ms(video, n, st):
    h, m, s = st.split(":")
    return round(video.tms(n) - (int(h) * 3600000 + int(m) * 60000 + round(float(s) * 1000)))


def mask_zone_rects(m, zones=None):
    """bounding boxes of mask m: one per zone (or one for all), overlapping boxes merged -> [(x0, y0, x1, y1)]"""
    parts = [m]
    if zones:
        parts = []
        for z in zones:
            x0, y0, x1, y1 = [max(0, int(v)) for v in z]
            sub = np.zeros_like(m); sub[y0:y1, x0:x1] = m[y0:y1, x0:x1]
            parts.append(sub)
    rects = []
    for p in parts:
        if p.any():
            yy, xx = np.where(p)
            rects.append([int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1])
    if not rects:
        yy, xx = np.where(m)
        rects = [[int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]]
    merged = True
    while merged:   # a vector clip of overlapping boxes may render the overlap as a hole
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del rects[j]; merged = True
                    break
            if merged:
                break
    return rects


def moving_lines(video, est, d, m, an, C, f0, segments, layer=10, block=3, maxK=64, style="Маска",
                 rect_clip=False, seg_block=None, cloud_block=3, smooth=False, layered=True, tau=4.0, feather=0,
                 feather_w=4, keep=None):
    """Lines for a translating plate: per segment the background drawing moves linearly; plus a static glow
    cloud (alpha `an`, colour C). rect_clip: clip to m's bounding box (cheap for many segments); a list of
    zones clips to the union of m's bounding box inside each zone (a wide title zone above a narrow name zone
    must not take the card's side flourishes into one big box)."""
    K = coords.K
    if rect_clip:
        rects = mask_zone_rects(m, rect_clip if isinstance(rect_clip, (list, tuple)) else None)
        if len(rects) == 1:
            bx0, by0, bx1, by1 = rects[0]
            clip = f"\\clip({f2(bx0 * K)},{f2(by0 * K)},{f2(bx1 * K)},{f2(by1 * K)})"
        else:
            clip = "\\clip(" + " ".join(f"m {f2(x0 * K)} {f2(y0 * K)} l {f2(x1 * K)} {f2(y0 * K)} {f2(x1 * K)} {f2(y1 * K)} "
                                        f"{f2(x0 * K)} {f2(y1 * K)}" for x0, y0, x1, y1 in rects) + ")"
        m = np.zeros_like(m)
        for bx0, by0, bx1, by1 in rects:
            m[by0:by1, bx0:bx1] = True
    else:
        clip = "\\clip(" + mask_polygon(m, holes=True) + ")"
    # feather: `feather` rings (feather_w px each) around the mask fade from the patch into the real frame,
    # so a slight colour mismatch at the mask edge does not draw a visible outline. The opaque patch keeps
    # the full mask; the rings lie outside it (minus `keep`: card elements stay untouched). Under the patch
    # (layer-2) lie `feather` stacked copies of the background, copy j clipped to the mask dilated
    # (feather-j) rings with alpha 1/(feather+1-j) - stacked, ring r (1 = next to the mask) ends up at
    # opacity (feather+1-r)/(feather+1). The copies only carry the part of the background that passes under
    # the rings (one k-means palette, no coarse/fine layering: semi-transparent layers over each other would
    # double the opacity where fine blocks sit).
    rings, ring_masks, ring_m = [], [], None
    if feather:
        k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather_w + 1,) * 2)
        free = ~keep if keep is not None else np.ones_like(m)
        grown = [m]
        for r in range(feather):
            grown.append(cv2.dilate(grown[-1].astype(np.uint8), k_).astype(bool) & (free | m))
        ring_masks = [grown[r] & ~grown[r - 1] for r in range(1, feather + 1)]
        ring_m = grown[-1] & ~m
        if ring_m.any():
            for j in range(feather):
                rings.append(("\\clip(" + mask_polygon(grown[feather - j], holes=True) + ")",
                              int(round(255 * (1 - 1.0 / (feather + 1 - j))))))
        else:
            ring_m = None
    out = []
    Bimg = edge_smooth(est["B"]) if smooth else est["B"]

    def bg_region(mm, ia, ib):
        """background-canvas pixels that pass under screen mask mm during frames ia..ib"""
        ys, xs = np.where(mm)
        reg = np.zeros((est["CH"], est["CW"]), bool)
        for i in range(ia, ib + 1):
            bx = np.round(xs - d[i, 0] - est["px0"]).astype(int); by = np.round(ys - d[i, 1] - est["py0"]).astype(int)
            ok = (bx >= 0) & (by >= 0) & (bx < est["CW"]) & (by < est["CH"])
            reg[by[ok], bx[ok]] = True
        return cv2.dilate(reg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    for fa, fb in segments:
        b0 = seg_block(fa, fb) if seg_block else block
        ia, ib = fa - f0, fb - f0
        st, en = video.atime(fa), video.atime(fb + 1)
        t1, t2 = _ms(video, fa, st), _ms(video, fb, st)
        reg = bg_region(m, ia, ib)
        ox, oy = est["px0"], est["py0"]
        xa, ya, xb, yb = (ox + d[ia, 0]) * K, (oy + d[ia, 1]) * K, (ox + d[ib, 0]) * K, (oy + d[ib, 1]) * K
        mv = f"\\move({f2(xa)},{f2(ya)},{f2(xb)},{f2(yb)},{t1},{t2})" if ib > ia else f"\\pos({f2(xa)},{f2(ya)})"
        if layered:
            items = [(b, col, rl, layer - 1 if b != b0 else layer)
                     for b, col, rl in layered_rects(Bimg, reg, fine=b0, coarse=b0 * 4, tau=tau, maxK=maxK,
                                                     outside_free=True)]
        else:
            rects, cent = quant_rects(Bimg, reg, b0, maxK)
            items = [(b0, cent[c], rl, layer) for c, rl in rects.items() if rl]
        for b, col, rl, lay in items:
            s = coords.scl(b)
            tags = "{\\an7" + mv + "\\fscx" + s + "\\fscy" + s + clip + "\\c" + ass_color(col) + "\\p1}"
            out.append(f"Dialogue: {lay},{st},{en},{style},mask,0,0,0,,{tags}" + draw_rects_units(rl, ext=round(1.0 / b, 2)))
        if rings:
            rr, rc = quant_rects(Bimg, bg_region(ring_m, ia, ib), b0, max(8, maxK // 2))
            s = coords.scl(b0)
            for c, rl in rr.items():
                if not rl:
                    continue
                body = draw_rects_units(rl, ext=round(1.0 / b0, 2))
                for rclip, ra in rings:
                    tags = ("{\\an7" + mv + "\\fscx" + s + "\\fscy" + s + rclip + "\\c" + ass_color(rc[c])
                            + f"\\1a&H{ra:02X}&" + "\\p1}")
                    out.append(f"Dialogue: {layer - 2},{st},{en},{style},mask,0,0,0,,{tags}" + body)
    # static glow cloud
    st, en = video.atime(segments[0][0]), video.atime(segments[-1][1] + 1)
    out += cloud_lines(an, m, clip, C, st, en, style, layer + 1)
    if rings:    # the glow fades out with the patch: ring r gets alpha a*(F+1-r)/(F+1) (static: one copy per ring)
        F = len(rings)
        for r, ring in enumerate(ring_masks, 1):
            if ring.any():
                out += cloud_lines(an * (F + 1 - r) / (F + 1), ring, "\\clip(" + mask_polygon(ring, holes=True) + ")",
                                   C, st, en, style, layer + 1)
    return out


def cloud_lines(an, m, clip, C, st, en, style, layer, levels=48, blur=0.5, eps=1.0):
    """A smooth static alpha field `an` (colour C) over mask m as stacked nested level sets: layer k covers
    {an >= k/levels} with the alpha that brings the stack to exactly k/levels (same colour, so the order of
    stacking does not matter). Outlines are polygons from the smooth field and get a light \\blur, so the
    result has no block staircase (a rect mosaic with 2% alpha steps shows as a mesh on a steep halo)."""
    region = dilate(m, 3)              # level sets reach a bit past the clip: no soft edge inside it
    q = np.clip(an, 0, 0.95)
    out, keep = [], 1.0                # keep = transmittance of the stack so far
    for k in range(1, levels + 1):
        S = (q >= (k - 0.5) / levels) & region
        if S.sum() < 4:
            break
        A = int(round(255 * (1 - k / levels) / keep))      # \1a byte: 255 = invisible
        A = max(0, min(254, A))
        keep *= A / 255.0
        tags = ("{\\an7\\pos(0,0)" + (f"\\blur{blur}" if blur else "") + clip + "\\c" + ass_color(C)
                + f"\\1a&H{A:02X}&" + "\\p1}")
        out.append(f"Dialogue: {layer},{st},{en},{style},mask,0,0,0,,{tags}" + mask_polygon(S, eps=eps, holes=True))
    return out


def colfit(src, dst, sel):
    """per-channel linear map src -> dst over pixels sel: returns (3,2) [gain, offset]"""
    out = []
    for c in range(3):
        x = src[..., c][sel]; y = dst[..., c][sel]
        A = np.stack([x, np.ones_like(x)], 1)
        (g, o), *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append((g, o))
    return np.array(out, np.float32)


def mapc(c, cm):
    return np.clip(c * cm[:, 0] + cm[:, 1], 0, 255)


def affine_lines(video, P, AFF, need, m, segments, color_maps, layer=10, style="Маска", block=3, maxK=48, tau=4.0,
                 strips=None):
    """Plate P (ref coords) re-projected per segment. AFF: {frame: 2x3 ref->frame}; need: region of P to encode;
    m: screen-space clip mask; color_maps: {frame: (3,2)} sorted keys used as colour keyframes.
    strips: [(rgb, [rects px])] of plates filled to their edges (platefill), carried by the same track."""
    K = coords.K
    items = layered_rects(edge_smooth(P, 1), need, fine=block, coarse=block * 4, tau=tau, maxK=maxK) if need.any() else []
    items = [(b, col, rl, round(1.0 / b, 2), layer if b == block else layer - 1) for b, col, rl in items]
    items += [(1, np.float32(col), rl, 0, layer) for col, rl in (strips or [])]
    clip = ("\\clip(" + mask_polygon(m) + ")") if m is not None else ""   # None: the patch moves with the scene
    keys = sorted(color_maps)
    out = []
    # rotation of the track (paper / phone turning in a hand): \frz about the \move point (the image of the ref
    # origin); only written when the track really turns, so plain zooms stay as before
    rot = any(abs(affine_params(AFF[f])[4]) > 0.02 for s in segments for f in s)
    for fa, fb in segments:
        st, en = video.atime(fa), video.atime(fb + 1)
        t1, t2 = _ms(video, fa, st), _ms(video, fb, st)
        Aa, Ab = AFF[fa], AFF[fb]
        _, _, sxa, sya, ra = affine_params(Aa)
        _, _, sxb, syb, rb = affine_params(Ab)
        mv = f"\\move({f2(Aa[0, 2] * K)},{f2(Aa[1, 2] * K)},{f2(Ab[0, 2] * K)},{f2(Ab[1, 2] * K)},{t1},{t2})"
        seg_keys = [fa] + [k for k in keys if fa < k < fb] + ([fb] if fb > fa else [])
        for b, col, rl, ext, lay in items:
            k = 100 * b * K
            sc = f"\\fscx{k * sxa:.4f}\\fscy{k * sya:.4f}" + (f"\\frz{-ra:.3f}" if rot else "")
            if fb > fa:
                sc += (f"\\t({t1},{t2},\\fscx{k * sxb:.4f}\\fscy{k * syb:.4f}"
                       + (f"\\frz{-rb:.3f}" if rot else "") + ")")
            cols = [mapc(col, color_maps[x]) for x in seg_keys]
            ctag = "\\c" + ass_color(cols[0])
            for j in range(1, len(seg_keys)):
                ta, tb = _ms(video, seg_keys[j - 1], st), _ms(video, seg_keys[j], st)
                ctag += f"\\t({ta},{tb},\\c" + ass_color(cols[j]) + ")"
            tags = "{\\an7" + mv + sc + clip + ctag + "\\p1}"
            out.append(f"Dialogue: {lay},{st},{en},{style},mask,0,0,0,,{tags}" + draw_rects_units(rl, ext=ext))
    return out
