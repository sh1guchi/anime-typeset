"""Encoding pixels as ASS vector drawings (\\p1) and other drawing helpers.

Drawings are authored in analysis px (1920x1080) and scaled to PlayRes with \\fscx/\\fscy (coords.scl).
"""
import numpy as np, cv2
from . import coords

f2 = coords.f2


def ass_color(rgb):
    r, g, b = [int(round(min(255, max(0, v)))) for v in rgb]
    return f"&H{b:02X}{g:02X}{r:02X}&"


def rects_from_labels(L, K):
    """L: label grid (-1 = empty). returns {c: [(x0,y0,x1,y1)]} - horizontal runs merged vertically (grid units)"""
    hb, wb = L.shape
    rects = {c: [] for c in range(K)}
    open_ = {}
    for y in range(hb):
        row = L[y]; cur = set(); x = 0
        while x < wb:
            c = row[x]
            if c < 0:
                x += 1; continue
            x0 = x
            while x < wb and row[x] == c:
                x += 1
            key = (x0, x, int(c)); cur.add(key)
            if key in open_ and open_[key][1] == y - 1:
                open_[key][1] = y
            else:
                if key in open_:
                    rects[key[2]].append((x0, open_[key][0], x, open_[key][1] + 1))
                open_[key] = [y, y]
        for key in list(open_):
            if key not in cur:
                x0, x1, c = key
                rects[c].append((x0, open_[key][0], x1, open_[key][1] + 1)); del open_[key]
    for key, (a, b) in open_.items():
        x0, x1, c = key
        rects[c].append((x0, a, x1, b + 1))
    return rects


def cover_rects(G):
    """G: 1 = must cover, 2 = may cover, 0 = must not. Greedy: from each uncovered 1 grow right as far as
    allowed then down, and down then right; keep the one covering more needed cells, trimmed to its last needed
    column/row. -> [(x0, y0, x1, y1)]"""
    H, W = G.shape
    need = G == 1
    ok = G > 0
    out = []
    for y, x in zip(*np.nonzero(need)):
        if not need[y, x]:
            continue
        x2 = x + 1
        while x2 < W and ok[y, x2]:
            x2 += 1
        y2 = y + 1
        while y2 < H and ok[y2, x:x2].all():
            y2 += 1
        y3 = y + 1
        while y3 < H and ok[y3, x]:
            y3 += 1
        x3 = x + 1
        while x3 < W and ok[y:y3, x3].all():
            x3 += 1
        if need[y:y3, x:x3].sum() > need[y:y2, x:x2].sum():
            x2, y2 = x3, y3
        sub = need[y:y2, x:x2]
        x2 = x + int(np.nonzero(sub.any(axis=0))[0][-1]) + 1
        y2 = y + int(np.nonzero(sub.any(axis=1))[0][-1]) + 1
        need[y:y2, x:x2] = False
        out.append((int(x), int(y), int(x2), int(y2)))
    return out


def painter_rects(L, free=None, under=None):
    """Opaque mosaic in painter's order: [(label, rects)], to be drawn in this order on one layer. A colour drawn
    earlier may run under the cells of colours drawn later (they paint over it) and over `free` cells (never
    visible, e.g. outside the clipped region), so the big colours become a few big rects. Same mosaic as the
    exact partition of rects_from_labels with fewer rects (-15..25% on real patches; only the 1-px overlap
    strips between neighbouring blocks may take the other neighbour's colour).
    under: labels of a layer beneath (same palette): an empty cell may take colour c where c is already below."""
    if not (L >= 0).any():
        return []
    labels = np.unique(L[L >= 0])
    # bottom first: the colours spread over the biggest box (they gain most from running under the others)
    ys, xs = np.nonzero(L >= 0)
    lab = L[ys, xs]
    span = [(np.ptp(ys[lab == c]) + 1) * (np.ptp(xs[lab == c]) + 1) for c in labels]
    order = labels[np.argsort(-np.array(span), kind="stable")]
    rank = np.full(int(L.max()) + 1, -1, int)
    rank[order] = np.arange(len(order))
    R = np.where(L >= 0, rank[np.maximum(L, 0)], -1)
    fr = (free & (L < 0)) if free is not None else None
    out = []
    for r, c in enumerate(order):
        G = np.where(R > r, 2, 0).astype(np.uint8)
        if fr is not None:
            G[fr] = 2
        if under is not None:
            G[(L < 0) & (under == c)] = 2
        G[R == r] = 1
        out.append((int(c), cover_rects(G)))
    return out


def rect_path(X0, Y0, X1, Y1):
    return f"m {f2(X0)} {f2(Y0)} l {f2(X1)} {f2(Y0)} {f2(X1)} {f2(Y1)} {f2(X0)} {f2(Y1)}"


def patch_lines(img, mask, start, end, block=3, maxK=64, per=6, layer=10, style="Маска", feather=6,
                alevels=4, ext=1.0, extra=""):
    """Static patch: pixels of `img` under `mask` as k-means coloured rectangles.
    feather: alpha ramps up over this many px from the mask edge (soft transition into the real frame)."""
    H, W = mask.shape
    hb, wb = (H + block - 1) // block, (W + block - 1) // block
    pm = np.zeros((hb * block, wb * block), bool); pm[:H, :W] = mask
    pi = np.zeros((hb * block, wb * block, 3), np.float32); pi[:H, :W] = img
    bm = pm.reshape(hb, block, wb, block).any(axis=(1, 3))
    bc = pi.reshape(hb, block, wb, block, 3).mean(axis=(1, 3))
    ys, xs = np.where(bm)
    if len(ys) == 0:
        return []
    cols = bc[ys, xs].astype(np.float32)
    K = int(min(maxK, max(2, len(cols) // per)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    cv2.setRNGSeed(1234)   # deterministic palette regardless of build order
    _, lab, cent = cv2.kmeans(cols, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    alv = np.full(len(lab), alevels, int)
    if feather > 0:
        dist = cv2.distanceTransform(np.pad(bm, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1] * block
        a = np.clip((dist[ys, xs] - block * 0.5) / feather, 0, 1)
        alv = np.clip(np.ceil(a * alevels), 1, alevels).astype(int)
    combo = lab * (alevels + 1) + alv
    uniq, inv = np.unique(combo, return_inverse=True)
    L = -np.ones((hb, wb), int); L[ys, xs] = inv
    # opaque cells: painter's order (colours may run under later ones); semi-transparent edge cells must tile
    # exactly and are never run under (an opaque colour beneath would show through them)
    opaque = np.array([int(cb) % (alevels + 1) == alevels for cb in uniq])
    Lo = np.where((L >= 0) & opaque[np.maximum(L, 0)], L, -1)
    La = np.where((L >= 0) & ~opaque[np.maximum(L, 0)], L, -1)
    ordered = painter_rects(Lo) + [(i, rl) for i, rl in rects_from_labels(La, len(uniq)).items() if rl]
    out = []
    s = coords.scl()
    for i, rl in ordered:
        c, al = divmod(int(uniq[i]), alevels + 1)
        if not rl:
            continue
        # opaque blocks overlap neighbours by `ext` (no seams); semi-transparent ones must tile exactly,
        # otherwise the overlaps show up as a darker grid
        e = ext if al == alevels else 0
        parts = [rect_path(x0 * block, y0 * block, x1 * block + e, y1 * block + e) for x0, y0, x1, y1 in rl]
        aa = int(round(255 * (1 - al / alevels)))
        atag = f"\\1a&H{aa:02X}&" if aa else ""
        tags = "{\\an7\\pos(0,0)\\fscx" + s + "\\fscy" + s + extra + "\\c" + ass_color(cent[c]) + atag + "\\p1}"
        out.append(f"Dialogue: {layer},{start},{end},{style},mask,0,0,0,,{tags}" + " ".join(parts))
    return out


def mask_polygon(m, eps=2.0, holes=False):
    """outline of mask m as an ASS vector clip in PlayRes units (0.1 precision).
    holes=True keeps inner contours too (OpenCV gives them the opposite orientation, so they cut holes)."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP if holes else cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    g = lambda v: f"{v * coords.K:.1f}".rstrip("0").rstrip(".")
    parts = []
    for c in cs:
        c = cv2.approxPolyDP(c, eps, True)[:, 0, :]
        if len(c) < 3:
            continue
        parts.append(f"m {g(c[0][0])} {g(c[0][1])} l " + " ".join(f"{g(x)} {g(y)}" for x, y in c[1:]))
    return " ".join(parts)


def quant_rects(img, region, block, maxK, per=6):
    """k-means colours of img over region -> (rects per cluster in block units, centers)"""
    H, W = region.shape
    hb, wb = (H + block - 1) // block, (W + block - 1) // block
    pm = np.zeros((hb * block, wb * block), bool); pm[:H, :W] = region
    pi = np.zeros((hb * block, wb * block, 3), np.float32); pi[:H, :W] = img
    bm = pm.reshape(hb, block, wb, block).any(axis=(1, 3))
    bc = pi.reshape(hb, block, wb, block, 3).mean(axis=(1, 3))
    ys, xs = np.where(bm)
    cols = bc[ys, xs].astype(np.float32)
    K = int(min(maxK, max(2, len(cols) // per)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    cv2.setRNGSeed(1234)   # deterministic palette regardless of build order
    _, lab, cent = cv2.kmeans(cols, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    L = -np.ones((hb, wb), int); L[ys, xs] = lab.ravel()
    return rects_from_labels(L, K), cent


def layered_rects(img, region, fine=3, coarse=12, tau=5.0, maxK=32, per=6, outside_free=False):
    """Two-layer encoding: coarse blocks everywhere, fine blocks only where they differ by > tau.
    Roughly halves the size of moving plates. Returns [(block, colour, rects_in_block_units)], coarse first,
    each layer in painter's order (draw in this order). outside_free: coarse cells outside `region` are never
    visible (a clipped moving patch: region = everything that passes under the clip), so colours may run there."""
    H, W = region.shape

    def grid(b):
        hb, wb = (H + b - 1) // b, (W + b - 1) // b
        pm = np.zeros((hb * b, wb * b), bool); pm[:H, :W] = region
        pi = np.zeros((hb * b, wb * b, 3), np.float32); pi[:H, :W] = img
        pw = pm.astype(np.float32)
        cnt = pw.reshape(hb, b, wb, b).sum(axis=(1, 3))
        col = (pi * pw[..., None]).reshape(hb, b, wb, b, 3).sum(axis=(1, 3)) / np.maximum(cnt, 1)[..., None]
        return cnt > 0, col

    cm, cc = grid(coarse)
    fm, fc = grid(fine)
    cols = np.concatenate([cc[cm], fc[fm]]).astype(np.float32)
    K = int(min(maxK, max(2, len(cols) // per)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    cv2.setRNGSeed(1234)   # deterministic palette regardless of build order
    _, _, cent = cv2.kmeans(cols, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)

    def nearest(c):
        d = ((c[:, None, :] - cent[None]) ** 2).sum(-1); return d.argmin(1)

    Lc = -np.ones(cm.shape, int); Lc[cm] = nearest(cc[cm])
    r = coarse // fine
    up = np.repeat(np.repeat(Lc, r, 0), r, 1)[:fm.shape[0], :fm.shape[1]]
    if up.shape != fm.shape:
        pad = np.full(fm.shape, -1, int); pad[:up.shape[0], :up.shape[1]] = up; up = pad
    upc = np.where(up[..., None] >= 0, cent[np.maximum(up, 0)], 0)
    Lf_all = nearest(fc[fm])
    Lf = -np.ones(fm.shape, int); Lf[fm] = Lf_all
    diff = np.zeros(fm.shape, bool)
    diff[fm] = (np.abs(cent[Lf_all] - upc[fm]).max(1) > tau) | (up[fm] < 0)
    Lf[~diff] = -1
    out = []
    # fine layer: a cell without a fine block shows the coarse colour below, so a fine colour may run over it
    # only where that coarse colour is the same palette entry (or over later fine colours)
    for b, L, fr, und in ((coarse, Lc, ~cm if outside_free else None, None), (fine, Lf, None, up)):
        for c, rl in painter_rects(L, fr, und):
            if rl:
                out.append((b, cent[c], rl))
    return out


def draw_rects_units(rects, ext=0.3):
    """rects in block units (drawing scaled by block) - short integer coordinates"""
    parts = []
    for x0, y0, x1, y1 in rects:
        X1, Y1 = f2(x1 + ext), f2(y1 + ext)
        parts.append(f"m {x0} {y0} l {X1} {y0} {X1} {Y1} {x0} {Y1}")
    return " ".join(parts)


def draw_rects(rects, block, ox, oy, ext=1.0):
    return " ".join(rect_path(x0 * block + ox, y0 * block + oy, x1 * block + ext + ox, y1 * block + ext + oy)
                    for x0, y0, x1, y1 in rects)


def edge_smooth(img, passes=2):
    out = np.clip(img, 0, 255).astype(np.float32)
    for _ in range(passes):
        out = cv2.bilateralFilter(out, 7, 10, 4)
    return out


def circle(cx, cy, r):
    k = 0.5523 * r
    return (f"m {f2(cx)} {f2(cy-r)} b {f2(cx+k)} {f2(cy-r)} {f2(cx+r)} {f2(cy-k)} {f2(cx+r)} {f2(cy)} "
            f"b {f2(cx+r)} {f2(cy+k)} {f2(cx+k)} {f2(cy+r)} {f2(cx)} {f2(cy+r)} "
            f"b {f2(cx-k)} {f2(cy+r)} {f2(cx-r)} {f2(cy+k)} {f2(cx-r)} {f2(cy)} "
            f"b {f2(cx-r)} {f2(cy-k)} {f2(cx-k)} {f2(cy-r)} {f2(cx)} {f2(cy-r)}")


def dark_iclip(gray, region, thr=128, min_px=50):
    """vector clip (PlayRes units) of dark shapes (silhouettes passing in front of a sign) inside region"""
    x0, y0, x1, y1 = [int(v) for v in region]
    x0, y0 = max(0, x0), max(0, y0)
    sub = gray[y0:y1, x0:x1]
    dark = (sub < thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if dark.sum() < min_px:
        return None
    dark = cv2.dilate(dark, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    parts = []
    for c in cs:
        if cv2.contourArea(c) < 20:
            continue
        c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :]
        pts = [(x + x0, y + y0) for x, y in c]
        parts.append(f"m {coords.P(pts[0][0])} {coords.P(pts[0][1])} l " + " ".join(f"{coords.P(x)} {coords.P(y)}" for x, y in pts[1:]))
    return " ".join(parts) if parts else None
