"""Camera / background motion: translation tracks, piecewise-linear segmentation, ECC affine, scale template."""
import numpy as np, cv2
from . import coords


def shift_img(img, dx, dy, W, H, ox, oy, interp=cv2.INTER_LINEAR):
    """sample img at (x + ox + dx, y + oy + dy) for canvas pixel (x, y)"""
    M = np.float32([[1, 0, ox + dx], [0, 1, oy + dy]])
    return cv2.warpAffine(img, M, (W, H), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def phase_track(gray_frames, axis=None):
    """cumulative translation between consecutive frames (N,2) via phase correlation.
    axis "y"/"x": track along one axis only, on the row/column mean profile (a camera tilt/pan over
    periodic detail such as rows of buttons, where 2D correlation can lock onto a wrong sideways peak)"""
    g = [f.astype(np.float32) for f in gray_frames]
    if axis == "y":
        g = [np.repeat(f.mean(axis=1)[:, None], 16, axis=1) for f in g]
    elif axis == "x":
        g = [np.repeat(f.mean(axis=0)[None, :], 16, axis=0) for f in g]
    h, w = g[0].shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    pos = [np.zeros(2)]
    for i in range(1, len(g)):
        (dx, dy), _ = cv2.phaseCorrelate(g[i - 1], g[i], win)
        if axis == "y":
            dx = 0.0   # the correlation surface is flat across the replicated profile
        elif axis == "x":
            dy = 0.0
        pos.append(pos[-1] + (dx, dy))
    return np.array(pos, np.float32)


def track_roi(video, f0, f1, roi, scale=1.0, axis=None):
    """translation of the content inside roi=(x, y, w, h) (analysis px) over frames f0..f1.
    scale < 1 tracks on downscaled frames: more robust for big per-frame shifts and repetitive
    detail (rows of buttons), where full-res phase correlation can lock onto a wrong peak."""
    x, y, w, h = roi
    if scale == 1.0:
        G = video.grab(f0, f1 - f0 + 1, crop=(x, y, x + w, y + h), gray=True)
        return phase_track(G, axis)
    W, H = int(round(coords.AW * scale)), int(round(coords.AH * scale))
    c = [int(round(v * scale)) for v in (x, y, x + w, y + h)]
    G = video.grab(f0, f1 - f0 + 1, crop=c, w=W, h=H, gray=True)
    return phase_track(G, axis) / scale


def linear_path(p):
    """fit p (N,2) with a straight constant-speed path starting at 0; returns (d, residual_max, velocity)"""
    t = np.arange(len(p))
    vx = np.polyfit(t, p[:, 0], 1); vy = np.polyfit(t, p[:, 1], 1)
    d = np.stack([np.polyval(vx, t) - np.polyval(vx, 0), np.polyval(vy, t) - np.polyval(vy, 0)], 1).astype(np.float32)
    res = float(max(np.abs(np.polyval(vx, t) - p[:, 0]).max(), np.abs(np.polyval(vy, t) - p[:, 1]).max()))
    return d, res, (float(vx[0]), float(vy[0]))


def smooth_path(d, sigma=1.5):
    """smooth per-frame velocities (tracking noise) and re-integrate"""
    if len(d) < 3:
        return d.astype(np.float32)
    v = np.diff(d, axis=0)
    k = np.exp(-0.5 * (np.arange(-4, 5) / sigma) ** 2); k /= k.sum()
    vs = np.stack([np.convolve(np.pad(v[:, c], 4, mode="edge"), k, mode="valid") for c in range(2)], 1)
    return np.vstack([d[:1], d[:1] + np.cumsum(vs, axis=0)]).astype(np.float32)


def fit_segments(d, f0, tol=0.35, rel=0.15):
    """Like segments_for, but each segment is the least-squares line through its frames instead of the chord
    between its end frames, so under the same per-frame bound (tol + rel * local speed) segments run longer and
    there are fewer of them (every segment re-draws the whole moving background: fewer segments = smaller
    file, same motion accuracy). Neighbouring segments need not meet - every frame is still within the bound.
    Returns ([(fa, fb)] inclusive and disjoint, fitted per-frame path to draw with)."""
    N = len(d)
    d = np.asarray(d, np.float64)
    if N == 1:
        return [(f0, f0)], d.astype(np.float32)
    sp = np.r_[np.linalg.norm(np.diff(d, axis=0), axis=1), 0]
    out = d.copy()
    segs, a = [], 0

    def fit(a, b):
        t = np.arange(b - a + 1, dtype=np.float64)
        A = np.stack([np.ones_like(t), t], 1)
        coef, *_ = np.linalg.lstsq(A, d[a:b + 1], rcond=None)
        return A @ coef

    while a < N:
        b, best = a, d[a:a + 1]
        while b + 1 < N:
            f = fit(a, b + 1)
            if (np.linalg.norm(f - d[a:b + 2], axis=1) > tol + rel * sp[a:b + 2]).any():
                break
            b, best = b + 1, f
        out[a:b + 1] = best
        segs.append((f0 + a, f0 + b))
        a = b + 1
    return segs, out.astype(np.float32)


def segments_for(d, f0, tol=0.35, rel=0.15):
    """greedy piecewise-linear segmentation of per-frame shifts -> [(fa, fb)] inclusive, non-overlapping.
    allowed error = tol + rel * local speed (fast motion hides small errors)"""
    N = len(d); segs = []; a = 0
    if N == 1:
        return [(f0, f0)]
    sp = np.r_[np.linalg.norm(np.diff(d, axis=0), axis=1), 0]
    while a < N - 1:
        b = a + 1
        while b + 1 < N:
            nb = b + 1
            t = (np.arange(a, nb + 1) - a) / (nb - a)
            lin = d[a][None] + t[:, None] * (d[nb] - d[a])[None]
            err = np.linalg.norm(lin - d[a:nb + 1], axis=1)
            if (err > tol + rel * sp[a:nb + 1]).any():
                break
            b = nb
        segs.append((f0 + a, f0 + b))
        a = b
    # consecutive segments share their boundary frame -> make them disjoint
    return [(x, y - 1) for x, y in segs[:-1]] + [segs[-1]]


def _prep(g, denoise):
    g = g if g.dtype == np.uint8 else np.clip(g, 0, 255).astype(np.uint8)
    if denoise:
        g = cv2.medianBlur(g, 5)
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)


def affine_params(A):
    """2x3 affine -> (tx, ty, sx, sy, angle_deg) as libass draws it: \\move point, \\fscx/\\fscy, rotation.
    angle is the screen rotation (clockwise positive in image coords, y down); \\frz takes -angle."""
    sx, sy = float(np.hypot(A[0, 0], A[1, 0])), float(np.hypot(A[0, 1], A[1, 1]))
    return float(A[0, 2]), float(A[1, 2]), sx, sy, float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))


def _apply_params(p, pts):
    tx, ty, sx, sy, ang = p
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    x, y = pts[:, 0] * sx, pts[:, 1] * sy
    return np.stack([tx + c * x - s * y, ty + s * x + c * y], 1)


def affine_segments(AFF, f0, f1, pts, tol=0.6, rel=0.1, cuts=()):
    """Greedy piecewise segmentation of an affine track for \\move + \\t(\\fscx\\fscy\\frz) lines: libass
    interpolates position, scale and angle linearly inside a segment, so a segment grows while that
    interpolation keeps `pts` (ref coords, e.g. the patch's bbox corners) within tol + rel * (their speed,
    px/frame) of where the track puts them. `cuts`: frames that must start a segment (fade ends...).
    Returns [(fa, fb)] inclusive, disjoint, covering f0..f1."""
    pts = np.asarray(pts, np.float64)
    P = {f: np.array(affine_params(AFF[f])) for f in range(f0, f1 + 1)}
    true = {f: _apply_params(P[f], pts) for f in P}
    if f1 == f0:
        return [(f0, f0)]
    cuts = {c for c in cuts if f0 < c <= f1}
    segs, a = [], f0
    while a < f1:
        b = a + 1
        while b + 1 <= f1 and b not in cuts:      # a cut frame ends this segment and starts the next
            nb = b + 1
            ok = True
            for f in range(a + 1, nb):
                t = (f - a) / (nb - a)
                pred = _apply_params(P[a] + t * (P[nb] - P[a]), pts)
                sp = np.linalg.norm(true[min(f + 1, f1)] - true[f], axis=1)
                if (np.linalg.norm(pred - true[f], axis=1) > tol + rel * sp).any():
                    ok = False; break
            if not ok:
                break
            b = nb
        segs.append((a, b))
        a = b
    # consecutive segments share their boundary frame -> make them disjoint
    out = [(x, y - 1) for x, y in segs[:-1]] + [segs[-1]]
    return [s for s in out if s[1] >= s[0]]


def ecc_affines(G, ref_i, mask=None, rows=None, denoise=True, iters=200):
    """affine A_i (2x3, ref coords -> frame i coords) for every frame of G (N,H,W uint8 gray), chained from ref.
    mask: uint8 (H,W), 0 = ignore (static overlays, the text itself). rows=(y0,y1): only use this band."""
    N, H, W = G.shape
    y0, y1 = rows or (0, H)
    R = _prep(G[ref_i], denoise)[y0:y1]
    M = None if mask is None else mask[y0:y1].astype(np.uint8)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    out = [None] * N
    out[ref_i] = np.eye(2, 3, dtype=np.float32)
    for rng in (range(ref_i + 1, N), range(ref_i - 1, -1, -1)):
        A = np.eye(2, 3, dtype=np.float32)
        for i in rng:
            try:
                _, A = cv2.findTransformECC(R, _prep(G[i], denoise)[y0:y1], A, cv2.MOTION_AFFINE, crit, M, 5)
            except cv2.error:
                print(f"    ECC did not converge at index {i}, reusing previous transform", flush=True)
            out[i] = A.copy()
    for i in range(N):  # crop-band coords -> full-frame coords
        A = out[i].copy()
        A[:, 2] += np.array([0, y0]) - A[:, :2] @ np.array([0, y0])
        out[i] = A
    return out


def warp_to(img, A, size=None):
    """image in ref coords -> frame coords"""
    return cv2.warpAffine(img, A, size or (coords.AW, coords.AH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def warp_from(img, A, size=None):
    """frame coords -> ref coords"""
    return cv2.warpAffine(img, A, size or (coords.AW, coords.AH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def scale_template_track(G, ref_i, search, dark=90, scales=(0.86, 1.06, 0.005)):
    """Track a dark logo by multi-scale template matching on gradient images.
    search=(x0,y0,x1,y1) region of the ref frame holding the logo. Returns (N,4): score, scale, cx, cy."""
    def grad(im):
        im = im.astype(np.float32)
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    x0, y0, x1, y1 = search
    T = G[ref_i].astype(np.float32)
    ys, xs = np.where(T[y0:y1, x0:x1] < dark)
    bx0, bx1, by0, by1 = xs.min() + x0 - 10, xs.max() + x0 + 10, ys.min() + y0 - 10, ys.max() + y0 + 10
    tmpl = T[by0:by1, bx0:bx1]
    S = np.arange(scales[0], scales[1] + 1e-9, scales[2])
    gts = []                                   # template gradients per scale: computed once, not per frame
    for s in S:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gts.append(grad(t))
    H, W = G[0].shape

    def match(i, cand, center, margin):
        best = None
        if center is None:                      # full-frame search
            gi = grad(G[i]); ox = oy = 0
        else:                                   # only around the previous position (logos move a few px/frame)
            hmax = max(gts[k].shape[0] for k in cand); wmax = max(gts[k].shape[1] for k in cand)
            ox = int(max(0, center[0] - wmax / 2 - margin)); oy = int(max(0, center[1] - hmax / 2 - margin))
            ex = int(min(W, center[0] + wmax / 2 + margin)); ey = int(min(H, center[1] + hmax / 2 + margin))
            gi = grad(G[i][oy:ey, ox:ex])
        for k in cand:
            gt = gts[k]
            if gt.shape[0] > gi.shape[0] or gt.shape[1] > gi.shape[1]:
                continue
            r = cv2.matchTemplate(gi, gt, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(r)
            if best is None or mv > best[0]:
                best = (mv, S[k], ox + ml[0] + gt.shape[1] / 2, oy + ml[1] + gt.shape[0] / 2)
        return best

    res = [None] * len(G)
    allk = list(range(len(S)))
    res[ref_i] = match(ref_i, allk, None, 0)
    for order in (range(ref_i + 1, len(G)), range(ref_i - 1, -1, -1)):
        prev = res[ref_i]
        for i in order:
            k0 = int(np.argmin(np.abs(S - prev[1])))
            near = [k for k in allk if abs(k - k0) <= 4]       # +-0.02 scale per step from the last good frame
            b = match(i, near, (prev[2], prev[3]), 60)
            if b is None or b[0] < 0.5:                        # lost (cut, flash, occluded): full search
                full = match(i, allk, None, 0)
                b = full if b is None or full[0] > b[0] else b
            res[i] = b
            if b[0] >= 0.5:
                prev = b
    return np.array(res, np.float64)
