"""Plates filled to their edges.

A label / card / caption box with a frame (yellow recipe labels, white info cards) is not patched glyph by
glyph: its whole inside is replaced by a synthetic clean plate. From the Japanese text zone a flood fill by
similar colour (each pixel compared with its neighbour, so gradients pass) grows up to the frame; the text,
its outline and anti-aliasing are enclosed holes of that region and get filled too. The background model is
chosen by the residual on the plate's own pixels outside the text:
    const - one colour,  rows - one colour per row (vertical gradient),  cols - one colour per column;
too large a residual (texture, lighting) -> no plate, the zone falls back to the ordinary glyph mask.
The fill is drawn as straight strips (one rectangle per run of rows / columns of one colour and span; a
one-colour rectangular plate is one rectangle), not as a k-means mosaic: exact edges, a few KB."""
import numpy as np, cv2
from . import coords
from .masks import glyph_mask, dilate
from .draw import ass_color

MODELS = ("const", "rows", "cols")


def _local_std(gray, k=5):
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k)); m2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def _profile(img, F, axis):
    """median colour per row (axis 0) / column (axis 1) over the pixels F; lines without samples interpolated"""
    n = F.shape[axis]
    prof = np.full((n, 3), np.nan, np.float32)
    for i in range(n):
        sel = F[i] if axis == 0 else F[:, i]
        if sel.sum() >= 3:
            prof[i] = np.median((img[i] if axis == 0 else img[:, i])[sel], axis=0)
    ok = ~np.isnan(prof[:, 0])
    if not ok.any():
        return None
    idx = np.arange(n)
    for c in range(3):
        prof[:, c] = np.interp(idx, idx[ok], prof[ok, c])
    return prof


def model_image(model, prof, shape):
    h, w = shape
    if model == "const":
        return np.broadcast_to(np.asarray(prof, np.float32), (h, w, 3))
    if model == "rows":
        return np.broadcast_to(prof[:, None, :], (h, w, 3))
    return np.broadcast_to(prof[None, :, :], (h, w, 3))


def _resid(img, S, model, prof, k=7):
    """structural error of a model: image and model both averaged over k x k (only S pixels), so encoder
    dithering / banding inside a steep gradient (+-6 levels, invisible) does not count, while texture,
    lighting and a 2D gradient (what a flat fill would show) do"""
    if not S.any():
        return 1e9
    w = S.astype(np.float32)
    den = cv2.blur(w, (k, k))
    ok = S & (den > 0.3)
    if not ok.any():
        return 1e9
    mi = model_image(model, prof, img.shape[:2]).astype(np.float32)
    d = np.zeros(img.shape[:2], np.float32)
    for c in range(3):
        a = cv2.blur(img[..., c] * w, (k, k)) / np.maximum(den, 1e-6)
        b = cv2.blur(np.ascontiguousarray(mi[..., c]) * w, (k, k)) / np.maximum(den, 1e-6)
        d = np.maximum(d, np.abs(a - b))
    if model != "const":
        # banding contours of a steep gradient wander by a line or so: an error a 1-line shift of the profile
        # explains is not structure
        slope = np.abs(np.gradient(np.asarray(prof, np.float32), axis=0)).max(axis=1) * 1.5
        d = np.maximum(0, d - (slope[:, None] if model == "rows" else slope[None, :]))
    return float(np.percentile(d[ok], 95))


def fit_model(img, F, S, force=None, tol=5.0):
    """(model, profile, residuals): profiles from F (plate pixels away from the text and its glow), residual =
    p95 of the max-channel error on S (F away from the frame too). The simplest model wins when it is
    within 0.75 level of the best."""
    profs = {"const": np.median(img[S], axis=0) if S.any() else None,
             "rows": _profile(img, F, 0), "cols": _profile(img, F, 1)}
    res = {k: (_resid(img, S, k, p) if p is not None else 1e9) for k, p in profs.items()}
    if force:
        return force, profs[force], res
    best = min(res, key=res.get)
    pick = "const" if res["const"] <= res[best] + 0.75 else best
    if res[pick] > tol:
        return None, None, res
    return pick, profs[pick], res


def _span_fill(F):
    """F with every row and every column filled from its first to its last pixel: the plate's inside including
    the text, also where a glyph touches the plate's edge (fill_holes would leave such a glyph open)"""
    R = np.zeros_like(F, dtype=bool)
    for M, out in ((F, R), (F.T, R.T)):
        any_ = M.any(axis=1)
        first = M.argmax(axis=1)
        last = M.shape[1] - 1 - M[:, ::-1].argmax(axis=1)
        for i in np.flatnonzero(any_):
            out[i, first[i]:last[i] + 1] = True
    return R


def _contiguity(R):
    """share of R inside the per-row (and per-column) first..last spans: 1.0 = no gaps (convex-ish plate)"""
    def one(M):
        tot = 0
        for line in M:
            nz = np.flatnonzero(line)
            if len(nz):
                tot += nz[-1] - nz[0] + 1
        return M.sum() / max(1, tot)
    return min(one(R), one(R.T))


def full_mask(p, shape=None):
    m = np.zeros(shape or coords.shape(), bool)
    x0, y0 = p["off"]
    h, w = p["R"].shape
    m[y0:y0 + h, x0:x0 + w] = p["R"]
    return m


def describe(p):
    x0, y0 = p["off"]; h, w = p["R"].shape
    r = ", ".join(f"{k} {v:.1f}" for k, v in p["res"].items())
    return f"plate [{x0},{y0},{x0 + w},{y0 + h}] {p['model']} (residual {r})"


def _text_cores(med, box, polarity):
    """pixels that are surely glyphs (near-black / near-white neutral strokes, as glyph_zone looks for them),
    connected components without thin lines (frames, underlines): what 'the zone's text is covered' counts"""
    from .masks import white_mask, dark_mask
    x0, y0, x1, y1 = box
    sub = med[y0:y1, x0:x1]
    mn, mx = sub.min(axis=2), sub.max(axis=2)
    if polarity == "dark":
        core = dark_mask(med, [box], thr=25, maxc=150, win=61)[y0:y1, x0:x1] & (mx <= 70) & (mx - mn < 40)
    else:
        core = white_mask(med, [box], thr=20, minc=120, win=61)[y0:y1, x0:x1] & (mn >= 215) & (mx - mn < 40)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    out = np.zeros(med.shape[:2], bool)
    for i in range(1, n):
        cx, cy, cw, ch, a = st[i]
        if a < 15 or min(cw, ch) <= 3 or max(cw, ch) > 8 * max(1, min(cw, ch)) or a < 0.06 * cw * ch:
            continue                       # specks, thin long lines, frame rings
        out[y0:y1, x0:x1][lab == i] = True
    return out


def find_plates(med, zone, det, force=None):
    """Plates whose inside holds the zone's glyphs. Returns (plates, glyph coverage 0..1, message).
    det: the zone's detect keys (polarity/thr/maxc/minc for the glyphs) + fill_tol (5.0 levels of structural error;
    the whole inside is replaced, so there is no seam to show it),
    fill_step (3, flood tolerance per neighbour step), fill_reach (300 px search around the zone),
    fill_erase_all (also erase other elements inside the plate: icons, coloured text the glyph mask missed).
    force: 'const'|'rows'|'cols' - the zone box itself is the plate, model forced."""
    H, W = med.shape[:2]
    zx0, zy0, zx1, zy1 = [int(v) for v in zone]
    tol = det.get("fill_tol", 5.0)
    G = glyph_mask(med, [zone], {k: v for k, v in det.items() if not k.startswith("fill")})
    img_all = np.clip(med, 0, 255).astype(np.float32)
    if force:
        R = np.zeros((zy1 - zy0, zx1 - zx0), bool); R[:] = True
        g = G[zy0:zy1, zx0:zx1]
        F = ~dilate(g, 4)
        S = cv2.erode(F.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        img = img_all[zy0:zy1, zx0:zx1]
        model, prof, res = fit_model(img, F, S, force=force)
        p = dict(off=(zx0, zy0), R=R, F=F, S=S, model=model, prof=prof, res=res, glyph=int(g.sum()))
        return [p], 1.0, f"box {zone} as plate, {force} (residual {res[force]:.1f})"
    if int(G.sum()) < 30:
        return [], 0.0, "no glyphs in the zone"
    pol = det.get("polarity", "light")
    cores = _text_cores(med, [zx0, zy0, zx1, zy1], pol)
    total = int(cores.sum()) or int(G.sum())
    if not cores.any():
        cores = G
    pad = det.get("fill_reach", 300)
    wx0, wy0, wx1, wy1 = max(0, zx0 - pad), max(0, zy0 - pad), min(W, zx1 + pad), min(H, zy1 + pad)
    img = img_all[wy0:wy1, wx0:wx1]
    u8 = np.round(img).astype(np.uint8)
    g = G[wy0:wy1, wx0:wx1]
    # glyph-like pixels anywhere in the window: text of the same plate outside the zone (big kanji above the
    # translated line) is text too, not an "other element"
    gw = dilate(glyph_mask(med, [(wx0, wy0, wx1, wy1)], {k: v for k, v in det.items() if not k.startswith("fill")})
                [wy0:wy1, wx0:wx1], 6)
    h, w = g.shape
    step = det.get("fill_step", 3)
    glow = det.get("fill_glow", 10)
    cand = (_local_std(u8.mean(axis=2)) < 2.5) & ~dilate(g, 3)
    # seeds: around the zone too (big glyphs leave little plate inside a tight zone), nearest to it first
    sp = 60
    lx0, ly0, lx1, ly1 = max(0, zx0 - wx0 - sp), max(0, zy0 - wy0 - sp), min(w, zx1 - wx0 + sp), min(h, zy1 - wy0 + sp)
    ys, xs = np.mgrid[ly0:ly1:6, lx0:lx1:6]
    zc = ((zx0 + zx1) / 2 - wx0, (zy0 + zy1) / 2 - wy0)
    seeds = sorted(((int(y), int(x)) for y, x in zip(ys.ravel(), xs.ravel()) if cand[y, x]),
                   key=lambda q: max(0, abs(q[1] - zc[0]) - (zx1 - zx0) / 2) + max(0, abs(q[0] - zc[1]) - (zy1 - zy0) / 2))
    # the window edge is a leak unless it is the frame edge (a caption at the bottom of the screen)
    open_edges = (wy0 > 0, wy1 < H, wx0 > 0, wx1 < W)
    why = {}

    def evaluate(F):
        """(plate, None) or (None, reason) for one flood region F (window coords)"""
        R = _span_fill(F)
        gin = int((g & R & ~F).sum())        # the zone's glyphs lie in the plate's holes, not in its colour
        if gin < 30:
            return None, "no text inside"
        yy, xx = np.where(R)
        by0, by1, bx0, bx1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        Rb, Fb = R[by0:by1, bx0:bx1], F[by0:by1, bx0:bx1]
        if Rb.mean() < 0.85 or _contiguity(Rb) < 0.99 or Fb.sum() < 0.3 * Rb.sum():
            return None, "not a plate shape"
        # a real plate meets its frame with its own colour all round (a glyph touching the edge is a short
        # gap); a background cell wrapping around another plate has that plate on its edge
        edge = Rb & ~cv2.erode(np.pad(Rb, 1).astype(np.uint8), np.ones((3, 3), np.uint8))[1:-1, 1:-1].astype(bool)
        edge_f = float((edge & Fb).sum()) / max(1, edge.sum())
        if edge_f < 0.8:
            return None, "edge is not plate colour"
        hb = Rb & ~Fb
        n, lab, st, _ = cv2.connectedComponentsWithStats(hb.astype(np.uint8), connectivity=8)
        gwb = gw[by0:by1, bx0:bx1]
        foreign = [i for i in range(1, n) if st[i, 4] > 40 and not gwb[lab == i].any()]
        if foreign and not det.get("fill_erase_all"):
            return None, "other elements inside"
        # the text's soft glow / outline shading is not plate: fit away from the holes, check away from the frame
        Fc = Fb & ~dilate(hb, glow)
        S = cv2.erode(Fc.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if S.sum() < 200:
            return None, "too little plate around the text"
        model, prof, res = fit_model(img[by0:by1, bx0:bx1], Fc, S, tol=tol)
        if model is None:
            return None, "not flat: " + ", ".join(f"{k} {v:.1f}" for k, v in res.items())
        return dict(off=(int(wx0 + bx0), int(wy0 + by0)), R=Rb, F=Fb, S=S, model=model, prof=prof, res=res,
                    glyph=gin, edge=edge_f), None

    used = np.zeros((h, w), bool)
    found, failed = [], []
    tries = 0
    for y, x in seeds:
        if used[y, x]:
            continue
        tries += 1
        if tries > 400:
            break
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(u8, ff, (x, y), 0, (step,) * 3, (step,) * 3, 4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
        F = ff[1:-1, 1:-1].astype(bool)
        used |= F
        if (open_edges[0] and F[0].any()) or (open_edges[1] and F[-1].any()) or                 (open_edges[2] and F[:, 0].any()) or (open_edges[3] and F[:, -1].any()):
            why["not enclosed"] = why.get("not enclosed", 0) + 1; continue
        if F.sum() < 300:
            continue
        p, reason = evaluate(F) if F.sum() >= 1500 else (None, "small")
        if p:
            found.append(p)
        else:
            if reason in ("small", "no text inside", "not a plate shape", "edge is not plate colour"):
                failed.append(F)
            if reason not in ("small", "no text inside"):
                why[reason] = why.get(reason, 0) + 1
    # text touching both edges of a plate splits its colour into parts (left / right of a tall kanji): parts
    # side by side (or stacked) with the same colour per row (column) are tried together
    failed = sorted(failed, key=lambda F: -int(F.sum()))[:10]
    info = []
    for F in failed:
        yy, xx = np.where(F)
        info.append((yy.min(), yy.max() + 1, xx.min(), xx.max() + 1, _profile(img, F, 0), _profile(img, F, 1)))

    def compatible(i, j):
        (ay0, ay1, ax0, ax1, ar, ac), (by0, by1, bx0, bx1, br, bc) = info[i], info[j]
        oy = min(ay1, by1) - max(ay0, by0); ox = min(ax1, bx1) - max(ax0, bx0)
        if oy > 0.5 * min(ay1 - ay0, by1 - by0) and max(ax0, bx0) - min(ax1, bx1) < 2 * max(ay1 - ay0, by1 - by0):
            r0, r1 = max(ay0, by0), min(ay1, by1)
            d = np.abs(ar[r0:r1] - br[r0:r1]).max(axis=1)
            return float(np.median(d)) <= 6
        if ox > 0.5 * min(ax1 - ax0, bx1 - bx0) and max(ay0, by0) - min(ay1, by1) < 2 * max(ax1 - ax0, bx1 - bx0):
            c0, c1 = max(ax0, bx0), min(ax1, bx1)
            d = np.abs(ac[c0:c1] - bc[c0:c1]).max(axis=1)
            return float(np.median(d)) <= 6
        return False
    from itertools import combinations
    taken = set()
    for k in (2, 3):
        for grp in combinations(range(len(failed)), k):
            if taken & set(grp) or not all(compatible(i, j) for i, j in combinations(grp, 2)):
                continue
            Fu = np.zeros_like(failed[0])
            for i in grp:
                Fu |= failed[i]
            p, reason = evaluate(Fu)
            if p:
                found.append(p); taken |= set(grp)
    # a flood inside a glyph's outline sits within the real plate (as a hole): keep the outer one
    masks = [full_mask(p, (H, W)) for p in found]
    keep = [i for i in range(len(found))
            if not any(j != i and (masks[i] & masks[j]).sum() >= 0.95 * masks[i].sum() and masks[j].sum() > masks[i].sum()
                       for j in range(len(found)))]
    # two plates that partly overlap cannot both be right: keep the one whose edge is cleaner
    keep = [i for i in keep
            if not any(j != i and (masks[i] & masks[j]).any() and
                       (found[j]["edge"], masks[j].sum()) > (found[i]["edge"], masks[i].sum()) for j in keep)]
    plates = [found[i] for i in keep]
    union = np.zeros((H, W), bool)
    for i in keep:
        union |= masks[i]
    # coverage of the zone's text: core components on a plate count as covered, the ones right next to a
    # plate (its frame, its outer outline) do not count at all, the rest are text no plate holds
    near = dilate(union, 14)
    n, lab = cv2.connectedComponents(cores.astype(np.uint8), connectivity=8)
    covered = uncovered = 0
    for i in range(1, n):
        comp = lab == i
        a = int(comp.sum())
        if (comp & union).any():
            covered += a
        elif not (comp & near).any():
            uncovered += a
    cov = covered / float(max(1, covered + uncovered)) if (covered or uncovered) else 0.0
    msg = (f"{len(plates)} plate(s): " + "; ".join(describe(p) for p in plates) if plates else "no plate") + \
        f", glyphs covered {cov:.0%}" + (f" (rejected: {', '.join(why)})" if why and cov < 0.9 else "")
    return plates, cov, msg


def dedupe(plates):
    out, masks = [], []
    for p in plates:
        m = full_mask(p)
        if any((m & q).sum() > 0.9 * min(m.sum(), q.sum()) for q in masks):
            continue
        out.append(p); masks.append(m)
    return out


def paint(P, p):
    """the plate's model painted into image P (analysis px)"""
    x0, y0 = p["off"]
    h, w = p["R"].shape
    sub = P[y0:y0 + h, x0:x0 + w]
    sub[p["R"]] = model_image(p["model"], p["prof"], (h, w))[p["R"]]
    return P


def stable(p, frames, med, thr=6.0):
    """the plate's clean pixels stay put over the given frames (a static overlay or a static shot)"""
    x0, y0 = p["off"]
    h, w = p["R"].shape
    ref = med[y0:y0 + h, x0:x0 + w].astype(np.float32)
    for f in frames:
        d = np.abs(f[y0:y0 + h, x0:x0 + w].astype(np.float32) - ref).max(axis=2)[p["S"]]
        if d.size and float(np.percentile(d, 90)) > thr:
            return False
    return True


def strip_groups(plates):
    """[(rgb, [(x0, y0, x1, y1) analysis px])]: one rectangle per run of rows (cols for 'cols') with the
    same span and colour; each run also covers the first line of the next run (inside both spans) so that
    two strips never meet on an anti-aliased edge (a seam of the frame showing through)"""
    groups = {}
    for p in plates:
        x0, y0 = p["off"]
        R, model, prof = p["R"], p["model"], p["prof"]
        axis = 1 if model == "cols" else 0
        M = R if axis == 0 else R.T
        runs = []
        for i, line in enumerate(M):
            nz = np.flatnonzero(line)
            if not len(nz):
                continue
            c = prof if model == "const" else prof[i]
            key = (int(nz[0]), int(nz[-1]) + 1, tuple(int(round(float(v))) for v in c))
            if runs and runs[-1][1] == i and runs[-1][2] == key:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1, key])
        for k, (i0, i1, (a, b, col)) in enumerate(runs):
            rects = [(a, i0, b, i1)]
            if k + 1 < len(runs) and runs[k + 1][0] == i1:
                na, nb = runs[k + 1][2][:2]
                lo, hi = max(a, na), min(b, nb)
                if hi > lo:
                    rects.append((lo, i1, hi, i1 + 1))
            for (u0, v0, u1, v1) in rects:     # (along-line, line) -> screen
                r = (x0 + u0, y0 + v0, x0 + u1, y0 + v1) if axis == 0 else (x0 + v0, y0 + u0, x0 + v1, y0 + u1)
                groups.setdefault(col, []).append(r)
    return list(groups.items())


def strip_lines(plates, st, en, layer=10, style="Маска", extra=""):
    """static strips: one Dialogue line per colour"""
    s = coords.scl()
    out = []
    for col, rects in strip_groups(plates):
        body = " ".join(f"m {x0} {y0} l {x1} {y0} {x1} {y1} {x0} {y1}" for x0, y0, x1, y1 in rects)
        out.append(f"Dialogue: {layer},{st},{en},{style},mask,0,0,0,,{{\\an7\\pos(0,0)\\fscx{s}\\fscy{s}{extra}"
                   f"\\c{ass_color(col)}\\p1}}{body}")
    return out
