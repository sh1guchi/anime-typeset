"""Ornamental name cards (white text + top line + underline with flourishes, dark glow; e.g. Black Clover).

Geometry is detected automatically from a median frame (card_geom); the Japanese title/name are erased
inside zones derived from it; the Russian title/name are set on the same baselines and the top line is
redrawn to the Russian title's width.
"""
import os
import numpy as np, cv2
from . import coords, fonts
from .coords import f2
from .masks import white_mask, dilate, harmonic_fill, text_mask_zone, rect_mask, fill_holes
from .draw import patch_lines, circle
from .text import emit_text
from .plate import glow_alpha, estimate_B, moving_lines, fit_glow, observed_alpha
from .motion import track_roi, linear_path, smooth_path, segments_for, fit_segments


def longest_run(row):
    best = cur = 0; bs = be = 0; s = 0
    for i, v in enumerate(row):
        if v:
            if cur == 0:
                s = i
            cur += 1
            if cur > best:
                best, bs, be = cur, s, i + 1
        else:
            cur = 0
    return best, bs, be


def card_geom(med, box, has_title=True):
    """underline (y, ux0, ux1), outer/inner flourish x (L, R, fl_in, fr_in), top line (line, lx0, lx1),
    title/name glyph extents; cx = card centre. Raises on failure (supply 'geom' manually then)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    W = white_mask(med, [box], thr=22, minc=130)[y0:y1, x0:x1]
    runs = [longest_run(W[y]) for y in range(W.shape[0])]
    lens = np.array([r[0] for r in runs])
    u = int(np.argmax(lens))
    while u > 0 and lens[u - 1] > 0.8 * lens[u]:
        u -= 1
    ux0, ux1 = runs[u][1], runs[u][2]
    band = W[max(0, u - 46):u - 6]
    occ = band.any(axis=0)
    xs = np.where(occ)[0]
    L, R = xs.min(), xs.max()

    def inner(start, step):
        x = start; gap = 0; seen = 0
        while 0 <= x < len(occ):
            if occ[x]:
                if gap >= 7 and seen > 30:
                    return x - step * gap
                gap = 0; seen += 1
            else:
                gap += 1
            x += step
        return start

    fl_in = inner(L, 1); fr_in = inner(R, -1)
    g = dict(underline=y0 + u, ux0=x0 + ux0, ux1=x0 + ux1, L=x0 + L, R=x0 + R, fl_in=x0 + fl_in, fr_in=x0 + fr_in)
    nb = W[:u - 3, fl_in + 2:fr_in - 1]
    rows = np.where(nb.any(axis=1))[0]
    line = None
    if has_title:
        for y in range(u - 50, max(0, u - 110), -1):
            ln, a, b = runs[y]
            if ln > 120 and abs((a + b) / 2 - (ux0 + ux1) / 2) < 120:
                line = y; break
    if line is not None:
        seg = W[line - 4:line + 5].any(axis=0)
        xs2 = np.where(seg)[0]
        g.update(line=y0 + line, lx0=x0 + xs2.min(), lx1=x0 + xs2.max())
        nrows = [r for r in rows if r > line + 3]
        trows = np.where(W[:line - 2, xs2.min():xs2.max() + 1].any(axis=1))[0]
        trows = trows[trows > line - 70]
        tcols = np.where(W[trows.min():line - 2].any(axis=0))[0]
        g.update(t_top=y0 + trows.min(), t_bot=y0 + trows.max(), t_x0=x0 + tcols.min(), t_x1=x0 + tcols.max())
    else:
        nrows = [r for r in rows if r > u - 70]
    nrows = np.array(nrows)
    ncols = np.where(W[nrows.min():nrows.max() + 1, fl_in + 2:fr_in - 1].any(axis=0))[0] + fl_in + 2
    g.update(n_top=y0 + nrows.min(), n_bot=y0 + nrows.max(), n_x0=x0 + ncols.min(), n_x1=x0 + ncols.max())
    g["cx"] = (g["L"] + g["R"]) / 2
    return {k: float(v) for k, v in g.items()}


def find_cards(med, min_run=250):
    """candidate card boxes: rows with long white runs (underlines / top lines), validated with card_geom"""
    AW, AH = coords.AW, coords.AH
    W = white_mask(med, [(0, 0, AW, AH)], thr=22, minc=130)
    hits = []
    for y in range(AH):   # every long run of the row: several cards can share a baseline
        row = np.r_[False, W[y], False].astype(np.int8)
        d = np.diff(row)
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if b - a >= min_run:
                hits.append((y, a, b))
    found = []
    for y, a, b in hits:
        box = (max(0, a - 90), max(0, y - 170), min(AW, b + 90), min(AH, y + 75))
        try:
            g = card_geom(med, box, True)
        except Exception:
            continue
        if g["R"] - g["L"] < (g["ux1"] - g["ux0"]) + 40:
            continue
        # a card has an ornament under the underline centre (a sign's arrow line has nothing there)
        u, cx = int(g["underline"]), int(g["cx"])
        if W[u + 6:u + 45, max(0, cx - 60):cx + 60].sum() < 60:
            continue
        if any(abs(g["underline"] - h["underline"]) < 8 and abs(g["cx"] - h["cx"]) < 40 for h in found):
            continue
        g["box"] = [int(g["L"] - 40), int((g.get("line", g["n_top"] + 30)) - 100), int(g["R"] + 40), int(g["underline"] + 60)]
        found.append(g)
    return found


def zones_from_geom(g):
    if "line" in g:
        lx = g["lx0"] if g["lx0"] > g["L"] + 40 else g["t_x0"]
        return [(int(min(g["t_x0"], lx) - 55), int(g["line"] - 78), int(g["t_x1"] + 55), int(g["line"] + 6)),
                (int(g["fl_in"] + 7), int(g["line"] + 6), int(g["fr_in"] - 7), int(g["underline"] - 5))]
    return [(int(g["fl_in"] + 7), int(g["n_top"] - 30), int(g["fr_in"] - 7), int(g["underline"] - 5))]


def card_box(g):
    top = g["line"] - 70 if "line" in g else g["n_top"] - 30
    return (int(g["L"]) - 10, int(top) - 20, int(g["R"]) + 10, int(g["underline"]) + 60)


def refine_lines(med, g, name="", reach=8):
    """Snap the top line / underline y to the actual thin bright line (a ridge: brighter than the rows 3 px
    above and below), within +-reach px. Hand-measured or auto values can be a few px off (bright skies,
    glow), and the redrawn line and the glow model then sit next to the real one."""
    if not all(k in g for k in ("L", "R")):
        return g
    lum = med.mean(axis=2)
    x0, x1 = int(g["L"] + 40), int(g["R"] - 40)
    if x1 - x0 < 60:
        return g
    for key in ("line", "underline"):
        if key not in g:
            continue
        y = int(round(g[key]))
        best, by = None, y
        for yy in range(max(3, y - reach), min(lum.shape[0] - 3, y + reach + 1)):
            r = lum[yy, x0:x1] - 0.5 * (lum[yy - 3, x0:x1] + lum[yy + 3, x0:x1])
            v = float(np.percentile(r, 60))      # the line covers most of the span, glyphs only parts of it
            if best is None or v > best:
                best, by = v, yy
        if best is not None and best > 25 and by != y:
            if abs(by - y) > 1:
                print(f"    {name}: {key} {y} -> {by} (snapped to the real line)", flush=True)
            g[key] = float(by)
    return g


def resolve(card, med):
    """geometry + erase zones for one card config"""
    g = {}
    if card.get("box") and not card.get("geom_only"):
        try:
            g = card_geom(med, card["box"], bool(card.get("title")))
        except Exception as e:
            if not card.get("geom"):
                raise SystemExit(f"card '{card.get('name')}': geometry not found ({type(e).__name__}) - light "
                                 "background or the box misses the card; give geom_only + geom + zones + glow_box")
    g.update(card.get("geom", {}))
    refine_lines(med, g, card.get("name", ""))
    zones = [tuple(z) for z in card.get("zones") or zones_from_geom(g)]
    lim = card.get("zone_limit")
    if lim:
        # one box for all zones, or a list with one box (or null) per zone
        lims = lim if isinstance(lim[0], (list, tuple)) or lim[0] is None else [lim] * len(zones)
        zones = [z if L is None else (max(z[0], L[0]), max(z[1], L[1]), min(z[2], L[2]), min(z[3], L[3]))
                 for z, L in zip(zones, lims)]
    return g, zones


def card_text(ctx, card, g, f0, f1):
    cp = ctx.preset["card"]
    cx = g["cx"]
    out = []
    spec = {"text": card["name"], "em": card.get("name_em", cp["name_em"]), "x": cx,
            "base": g["underline"] - cp.get("name_base_off", 14)}
    if "fl_in" in g:
        spec.update(maxw=(g["fr_in"] - g["fl_in"]) - 40, fit="squeeze")
    lines, _ = emit_text(ctx, spec, f0, f1, layers=cp["layers"])
    out += lines
    if card.get("title"):
        rec, met = ctx.styles.font(cp["layers"][-1]["style"])
        tem = card.get("title_em", cp["title_em"])
        avail = card.get("title_maxw") or ((g["R"] - g["L"]) - 2 * (cp["line_pad"] + 6))
        tw = fonts.text_width(rec, card["title"], tem)
        if tw > avail:
            tem *= avail / tw
        tw = fonts.text_width(rec, card["title"], tem)
        lines, _ = emit_text(ctx, {"text": card["title"], "em": tem, "x": cx, "base": g["line"] - cp.get("title_base_off", 9)},
                             f0, f1, layers=cp["layers"])
        out += lines
        half = tw / 2 + cp["line_pad"]
        xl, xr, y, rr = cx - half, cx + half, g["line"], cp.get("line_r", 3.1)
        gap = rr + 2.2; th = cp.get("line_th", 0.65)
        shape = " ".join([circle(xl, y, rr), circle(xr, y, rr),
                          f"m {f2(xl+gap)} {f2(y-th*1.3)} l {f2(xl+gap+40)} {f2(y-th)} {f2(xr-gap-40)} {f2(y-th)} {f2(xr-gap)} {f2(y-th*1.3)} "
                          f"{f2(xr-gap)} {f2(y+th*1.3)} {f2(xr-gap-40)} {f2(y+th)} {f2(xl+gap+40)} {f2(y+th)} {f2(xl+gap)} {f2(y+th*1.3)}"])
        s = coords.scl()
        st, en = ctx.video.atime(f0), ctx.video.atime(f1 + 1)
        out.append(f"Dialogue: 20,{st},{en},{cp['line_style']},Надпись,0,0,0,,{{\\an7\\pos(0,0)\\blur1\\fscx{s}\\fscy{s}\\p1}}{shape}")
    return out


def build_card(ctx, it):
    v = ctx.video
    f0, f1 = it["frames"]
    st, en = v.atime(f0), v.atime(f1 + 1)
    motion = it.get("motion", {"mode": "static"})
    if isinstance(motion, str):
        motion = {"mode": motion}
    mode = motion.get("mode", "static")
    enc = it.get("encode", {})
    dil = it.get("mask_dil", 28)
    cards = it["cards"]
    lines, areas = [], []
    if mode == "static":
        a, b = it.get("plate", [f0, f1])
        med = ctx.median(a, b)
        gmed = ctx.median(*it["geom_frames"]) if it.get("geom_frames") else med
        for card in cards:
            g, zones = resolve(card, gmed)
            m = text_mask_zone(med, zones, dil=dil)
            bx0 = max(0, min(z[0] for z in zones) - 90); by0 = max(0, min(z[1] for z in zones) - 60)
            bx1 = min(coords.AW, max(z[2] for z in zones) + 90); by1 = min(coords.AH, max(z[3] for z in zones) + 60)
            allw = dilate(white_mask(med, [(bx0, by0, bx1, by1)], thr=20, minc=120), 3)
            for p in card.get("protect", []):
                x0, y0, x1, y1 = p; allw[y0:y1, x0:x1] = False
            box = (bx0, by0, bx1, by1)
            if it.get("glow_band", ctx.preset.get("card", {}).get("glow_band", False)):
                # keep the card's dark glow continuous along the line (a shorter Russian name must not leave
                # lighter gaps where the Japanese glyphs' glow was): background without glow + smooth glow alpha
                Wc = white_mask(med, [box], thr=20, minc=120)
                gm = glow_model(ctx, it, med, Wc, box)
                a = glow_alpha(Wc, gm); C = np.array(gm.get("C", [0, 0, 0]), np.float32)
                U = (med - C * a[..., None]) / (1 - a[..., None])
                unknown = m | dilate(Wc, 2)
                Bc = harmonic_fill(U, unknown, box, iters=1500)
                an = harmonic_fill(np.repeat(a[..., None], 3, 2) * 255, unknown, box, iters=1500, presmooth=0)[..., 0] / 255
                clean = Bc * (1 - an[..., None]) + C * an[..., None]
            else:
                clean = harmonic_fill(med, m | allw, box, iters=1500)
            card_lines_ = patch_lines(clean, m, st, en, block=enc.get("block", 3), maxK=enc.get("maxK", 64), per=6,
                                      layer=10, feather=enc.get("feather", 6))
            if not it.get("source_text"):
                card_lines_ += card_text(ctx, card, g, f0, f1)
            if it.get("occluders"):
                card_lines_ = occlude(ctx, it, card_lines_, m, med)
            lines += card_lines_
            areas.append(card_box(g))
        return _with_source(ctx, it, lines, areas)
    # moving background
    from .tools import tracks, axis_suspect
    tr = tracks(v, f0, f1, motion["roi"], motion.get("scale", 1.0 if mode == "linear" else 0.5))
    p = tr[motion.get("axis") or "2d"]
    if not motion.get("axis"):        # sanity: 2D phase correlation can lock onto a wrong peak along one axis
        for ax, k in ((0, "x"), (1, "y")):
            two, one = float(tr["2d"][-1, ax]), float(tr[k][-1, ax])
            if axis_suspect(two, one):
                other = "y" if k == "x" else "x"
                print(f"    ! track along {k}: 2D {two:.1f} px vs 1D profile {one:.1f} px - check with "
                      f"`TS track --roi ...`; if the camera moves along {other} only, set \"axis\": \"{other}\"", flush=True)
    if mode == "linear":
        d, res, vel = linear_path(p)
        print(f"    linear motion v=({vel[0]:.3f},{vel[1]:.3f}) px/frame, max residual {res:.2f}px", flush=True)
        segs = [(f0, f1)]
        d_draw = d
    else:
        d = smooth_path(p - p[0])
        # the true track estimates the background; the patch is drawn along per-segment fitted lines
        segs, d_draw = fit_segments(d, f0, tol=motion.get("tol", 0.35), rel=motion.get("rel", 0.15))
        print(f"    path motion, total shift {d[-1].round(1)}, {len(segs)} segments "
              f"(chords would need {len(segments_for(d, f0, motion.get('tol', 0.35), motion.get('rel', 0.15)))})",
              flush=True)
    gmed_full = None
    geoms = []
    # band of the screen that holds the cards
    if it.get("geom_frames"):
        gmed_full = ctx.median(*it["geom_frames"])
    if gmed_full is None:
        gmed_full = ctx.median(f0, f1)
    for card in cards:
        geoms.append(resolve(card, gmed_full))
    gboxes = [tuple(c.get("glow_box") or card_box(g)) for c, (g, _) in zip(cards, geoms)]
    by0 = max(0, min(b[1] for b in gboxes) - 50); by1 = min(coords.AH, max(b[3] for b in gboxes) + 40)
    strip = v.grab(f0, f1 - f0 + 1, crop=(0, by0, coords.AW, by1))
    med = np.zeros(coords.shape() + (3,), np.float32)
    med[by0:by1] = np.median(strip, axis=0)
    Wc = white_mask(med, gboxes, thr=20, minc=120)
    gm = glow_model(ctx, it, med, Wc, (max(0, min(b[0] for b in gboxes) - 60), by0,
                                       min(coords.AW, max(b[2] for b in gboxes) + 60), by1))
    a = glow_alpha(Wc, gm)
    C = np.array(gm.get("C", [0, 0, 0]), np.float32)
    path = mode != "linear"
    if path:
        speed = lambda fa, fb: abs(d[fb - f0] - d[fa - f0]).max() / max(1, fb - fa)
        seg_block = lambda fa, fb: 4 if speed(fa, fb) > 4 else 3
    for card, (g, zones), gb in zip(cards, geoms, gboxes):
        m = fill_holes(text_mask_zone(med, zones, dil=dil))
        win = (max(0, gb[0] - 40), max(by0, gb[1] - 40), min(coords.AW, gb[2] + 40), min(by1, gb[3] + 30))
        inwin = np.zeros(m.shape, bool); inwin[win[1]:win[3], win[0]:win[2]] = True
        est = estimate_B(strip, 0, by0, d, a, C, Wc, win)
        ccfg = ctx.preset.get("card", {})
        a_in = a                  # glow the patch carries
        clip_zones = list(zones)
        seam = it.get("seam", ccfg.get("seam", True))
        halo_thr = it.get("halo", ccfg.get("halo", 0.015))
        ao = None
        if seam or halo_thr:
            # the glow the frames really show over the estimated background (the model is only a fit)
            ao = observed_alpha(strip, 0, by0, d, est, C, win)
            okao = np.isfinite(ao)
            ao = np.where(okao, cv2.medianBlur(np.nan_to_num(ao, nan=0).astype(np.float32), 5), a)
        if halo_thr:
            # the Japanese glyphs' own dark halo reaches past the glyph mask; left outside the patch it draws
            # a dark rim around a lighter box (the halo gradient breaks at the mask edge). The patch takes the
            # halo in, and inside it carries only the glow of the card elements that stay.
            n_, lab, stats, _ = cv2.connectedComponentsWithStats((Wc & ~m).astype(np.uint8), connectivity=8)
            keep_el = np.isin(lab, [i for i in range(1, n_) if stats[i, cv2.CC_STAT_AREA] >= 40])  # no specks
            a_in = glow_alpha(keep_el, gm)
            # only halo that belongs to the Japanese text: the model sees some of it there, and the pixel is
            # nearer the text than any card element (the model underrates the ornaments' halo too)
            dm = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
            dk = cv2.distanceTransform((~keep_el).astype(np.uint8), cv2.DIST_L2, 5)
            hl = ((ao - a_in > halo_thr) & (a - a_in > 0.003) & (dm < dk) & (dm <= it.get("halo_reach", 40))
                  & inwin & ~dilate(keep_el, 2))
            hl = cv2.morphologyEx(hl.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            hl = hl & _connected_to(hl | m, m)        # only halo attached to the text, not stray dark bits
            grow = int((hl & ~m).sum())
            if grow:
                m = fill_holes(m | hl) & ~dilate(keep_el, 2)   # holes only where card elements stay visible
                clip_zones = [[z[0] - 60, z[1] - 60, z[2] + 60, z[3] + 60] for z in zones]
                print(f"    halo: patch grows by {grow} px over the Japanese glyphs' glow", flush=True)
        a_edge = a
        if seam:
            # boundary values of the cloud from the frames right outside the patch
            band = dilate(m, 8) & ~m & ~dilate(Wc, 3) & okao
            if band.any():
                a_edge = a.copy(); a_edge[band] = np.clip(ao[band], 0, 0.95)
                print(f"    seam: glow model vs frames at the mask edge {np.abs(a[band] - a_edge[band]).mean():.3f}", flush=True)
        # cloud = glow of what stays + the mismatch at the patch edge, interpolated smoothly inwards
        res = np.repeat((a_edge - a_in)[..., None], 3, 2) * 255
        an = a_in + harmonic_fill(res, m | dilate(Wc, 2), win, iters=1500, presmooth=0)[..., 0] / 255
        if os.environ.get("TS_DEBUG"):
            np.savez_compressed(ctx.path("check", f"_{it['id']}_glow.npz"), a=a, a_in=a_in, a_edge=a_edge, an=an, m=m,
                                ao=ao if ao is not None else a, Wc=Wc)
        lines += moving_lines(v, est, d_draw, m, an, C, f0, segs, maxK=enc.get("maxK", 28 if path else 48),
                              tau=enc.get("tau", 5.0 if path else 4.0), layered=True,
                              rect_clip=clip_zones if path and motion.get("clip", "rect") == "rect" else False, smooth=path,
                              seg_block=seg_block if path else None, feather=motion.get("feather", 0),
                              feather_w=motion.get("feather_w", 4), keep=dilate(Wc, 3))
        if not it.get("source_text"):
            lines += card_text(ctx, card, g, f0, f1)
        areas.append(card_box(g))
    return _with_source(ctx, it, lines, areas)


def _connected_to(region, seed):
    """pixels of `region` in connected components that touch `seed`"""
    n, lab = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    ids = np.unique(lab[seed & region])
    return np.isin(lab, ids[ids > 0])


def glow_model(ctx, it, med, Wc, box):
    """item "glow": "fit" (fit on this card) | {sig, w, C} | absent (preset; preset card.glow_fit: true -> fit)"""
    base = ctx.preset.get("glow", {"sig": [3, 6, 12], "w": [0, 0.7, 0.3], "C": [0, 0, 0]})
    opt = it.get("glow", "fit" if ctx.preset.get("card", {}).get("glow_fit") else None)
    if isinstance(opt, dict):
        return opt
    if opt == "fit":
        gm, r, r0 = fit_glow(med, Wc, box, base)
        if r is None:
            print("    glow fit: not enough halo pixels - preset glow", flush=True)
        else:
            print(f"    glow fit: rms {r:.1f} vs {r0:.1f} without glow" + ("" if gm is not base else " - poor fit, preset glow"),
                  flush=True)
        return gm
    return base


def occlude(ctx, it, lines, m, med):
    """Something passes in front of a static card (a hand, a character): frames whose pixels under the patch
    differ from the clean plate get the patch and the text cut away there (\\iclip), so the object stays visible.
    it["occluders"]: {"thr": 35, "min_px": 400, "dil": 6}"""
    from .draw import mask_polygon
    import cv2 as _cv2
    o = it["occluders"] if isinstance(it["occluders"], dict) else {}
    thr, min_px, dl = o.get("thr", 35), o.get("min_px", 400), o.get("dil", 6)
    f0, f1 = it["frames"]
    reg = dilate(m, 12)
    ys, xs = np.where(reg)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    fr = ctx.video.grab(f0, f1 - f0 + 1, crop=(x0, y0, x1, y1)).astype(np.float32)
    ref = med[y0:y1, x0:x1]
    occ = {}
    for i, f in enumerate(range(f0, f1 + 1)):
        d = (np.abs(fr[i] - ref).max(axis=2) > thr) & reg[y0:y1, x0:x1]
        d = _cv2.morphologyEx(d.astype(np.uint8), _cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if d.sum() < min_px:
            continue
        full = np.zeros(m.shape, bool)
        full[y0:y1, x0:x1] = dilate(d, dl)
        occ[f] = mask_polygon(full, eps=1.5)
    if not occ:
        return lines
    print(f"    occluders: frames {min(occ)}-{max(occ)} ({len(occ)} frames)", flush=True)
    v = ctx.video
    out = []
    for l in lines:
        kind, rest = l.split(":", 1)
        p = rest.strip().split(",", 9)
        a0 = v.frame_at(int(p[1].split(":")[0]) * 3600 + int(p[1].split(":")[1]) * 60 + float(p[1].split(":")[2]))
        b0 = v.frame_at(int(p[2].split(":")[0]) * 3600 + int(p[2].split(":")[1]) * 60 + float(p[2].split(":")[2])) - 1
        run = None
        for f in range(a0, b0 + 2):
            if f <= b0 and f not in occ:
                run = run if run is not None else f
                continue
            if run is not None:
                q = list(p); q[1], q[2] = v.atime(run), v.atime(f)
                out.append(f"{kind}: " + ",".join(q)); run = None
            if f <= b0:
                q = list(p); q[1], q[2] = v.atime(f), v.atime(f + 1)
                txt = q[9]
                tag = f"\\iclip({occ[f]})"
                q[9] = ("{" + tag + txt[1:]) if txt.startswith("{") else ("{" + tag + "}" + txt)
                out.append(f"{kind}: " + ",".join(q))
    return out


def _with_source(ctx, it, lines, areas):
    """source_text: keep the translator's typeset text over our clean card (masks only from us)"""
    if it.get("source_text"):
        src, _ = emit_text(ctx, {"source": True, "lines": it.get("source_lines", [])}, *it["frames"])
        lines = lines + src
    return dict(lines=lines, area=_union(areas))


def _union(boxes):
    return [int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes)), int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes))]
