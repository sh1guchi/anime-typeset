"""Builders for the item types of episode.json: card, plate, follow, text."""
import numpy as np, cv2, os, re
from . import coords, fonts
from .coords import f2
from .masks import white_mask, glyph_mask, dilate, close, harmonic_fill, rect_mask
from .draw import patch_lines, dark_iclip
from .text import emit_text
from .motion import ecc_affines, warp_to, warp_from, scale_template_track, affine_params, affine_segments
from .plate import colfit, affine_lines
from .cards import build_card
from . import platefill

PLATE_FILLS = ("auto",) + platefill.MODELS


def build_item(ctx, it):
    t = it["type"]
    if t == "card":
        return build_card(ctx, it)
    if t == "plate":
        return build_plate(ctx, it)
    if t == "follow":
        return build_follow(ctx, it)
    if t == "text":
        return build_text(ctx, it)
    raise SystemExit(f"item {it['id']}: unknown type {t}")


def specs(it, many, one, default=None):
    """text specs of an item; {"source": true} specs get the item's source (translator's) lines"""
    ss = it[many] if many in it else ([it[one]] if one in it else (default or []))
    return [dict(sp, lines=it.get("source_lines", [])) if sp.get("source") and "lines" not in sp else sp for sp in ss]


# ---------------------------------------------------------------- text only
def build_text(ctx, it):
    f0, f1 = it["frames"]
    lines = []
    texts = specs(it, "texts", "text")
    if it.get("track"):
        # no mask, but the text rides the camera / an object: positions are given in the track's ref frame
        AFF, segs = text_track(ctx, it, texts)
        for spec in texts:
            lines += follow_affine(ctx, spec, segs, AFF)
        return dict(lines=lines, area=it.get("area"))
    for spec in texts:
        lines += emit_text(ctx, spec, f0, f1)[0]
    return dict(lines=lines, area=it.get("area"))


def text_track(ctx, it, texts):
    """Track for a text without a mask. track: {"roi": [x0,y0,x1,y1] - what the text rides (a board, a blade,
    the background), "ref": frame of the text positions (default f0), "mode": "shift" | "affine",
    "scale": 0.5 (tracking resolution), "axis": "x"|"y" (shift only), "exclude": [[x0,y0,x1,y1], ...] (affine),
    "seg_tol" 0.6 / "seg_rel" 0.1 (piecewise-linear \\move segments)}.
    shift  - cumulative frame-to-frame phase correlation of the roi: long pans/tilts where the end frame no longer
             overlaps the ref (a tall signboard scrolling through the frame); translation only.
    affine - ECC against the ref frame inside the roi: zoom / rotation; the motion must keep the roi in view.
    The track is cached (cache/ttrack_*.npy): changing only the text rebuilds in seconds."""
    from .motion import track_roi, smooth_path
    v = ctx.video
    f0, f1 = it["frames"]
    tr = it["track"]
    ref = int(tr.get("ref", f0))
    mode = tr.get("mode", "shift")
    s = float(tr.get("scale", 0.5))
    x0, y0, x1, y1 = [int(c) for c in tr["roi"]]
    key = f"ttrack_{mode}_{f0}_{f1}_{ref}_{x0}_{y0}_{x1}_{y1}_{s}_{tr.get('axis', '')}_{len(tr.get('exclude', []))}"
    cache = ctx.cache_path(key + ".npy")
    if os.path.exists(cache):
        A = np.load(cache)
    else:
        print(f"    tracking text ({mode}, roi {[x0, y0, x1, y1]})...", flush=True)
        if mode == "shift":
            p = track_roi(v, f0, f1, (x0, y0, x1 - x0, y1 - y0), scale=s, axis=tr.get("axis"))
            if tr.get("smooth", True):
                p = smooth_path(p)
            p = p - p[ref - f0]
            A = np.stack([np.float32([[1, 0, dx], [0, 1, dy]]) for dx, dy in p])
        elif mode == "affine":
            W, H = int(round(coords.AW * s)), int(round(coords.AH * s))
            G = v.grab(f0, f1 - f0 + 1, w=W, h=H, gray=True)
            m = np.zeros((H, W), np.uint8)
            m[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = 255
            for e in tr.get("exclude", []):
                m[int(e[1] * s):int(e[3] * s), int(e[0] * s):int(e[2] * s)] = 0
            A = np.stack(ecc_affines(G, ref - f0, m, denoise=tr.get("denoise", False)))
            A[:, :, 2] /= s
        else:
            raise SystemExit(f"item {it['id']}: track.mode must be shift or affine")
        np.save(cache + f".{os.getpid()}.tmp.npy", A.astype(np.float32))
        os.replace(cache + f".{os.getpid()}.tmp.npy", cache)
    AFF = {f0 + i: A[i] for i in range(len(A))}
    tx, ty, sc, _, rot = affine_params(AFF[f1])
    dx, dy = AFF[f1][:, :2] @ np.float32([(x0 + x1) / 2, (y0 + y1) / 2]) + AFF[f1][:, 2] - np.float32([(x0 + x1) / 2, (y0 + y1) / 2])
    print(f"    text track: roi centre moves {dx:+.1f},{dy:+.1f} px by the last frame, scale {sc:.3f}, rotation {rot:+.2f} deg",
          flush=True)
    pts = [(x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]
    cuts = [c for sp in texts for c in fade_cuts(v, sp, f0, f1)]
    segs = affine_segments(AFF, f0, f1, pts, tol=tr.get("seg_tol", 0.6), rel=tr.get("seg_rel", 0.1), cuts=cuts)
    print(f"    {len(segs)} segment(s)", flush=True)
    return AFF, segs


# ---------------------------------------------------------------- plate
def plate_mask(med, zone, det):
    if det.get("fill") is True:
        # the whole zone is replaced: flat / gradient plates whose glyphs are blurred, aberrated or sit on a
        # translucent box (the scene moving behind shows the glyph-shaped patch as ghosts)
        return rect_mask([zone], med.shape[:2])
    W = glyph_mask(med, [zone], det)
    ck = det.get("close", [25, 5])
    g = dilate(W, det.get("dil", 16))
    if det.get("shadow"):
        sh = det["shadow"]; ox, oy = sh.get("off", [7, 7])
        g = g | np.roll(np.roll(dilate(W, sh.get("dil", 10)), oy, 0), ox, 1)
    return close(g, ck[0], ck[1]) & rect_mask([zone], W.shape)


def build_plate(ctx, it):
    """Erase text inside `zone` (or several `zones`) and set the Russian text.
    clean.mode: 'inpaint' - smooth fill from the surroundings (plain/gradient backgrounds);
                'ref'     - clean plate = frame(s) before the text appears.
    motion: 'static' or 'affine' (ECC track of the frame).
    attached: true - the text belongs to the scene (screen, board, sign on a wall) and moves with it:
              mask, patch and Russian text all follow the track; detection/cleaning happen on the ref frame.
              false - the text is a static overlay (caption) over a moving background (clip stays put)."""
    v = ctx.video
    f0, f1 = it["frames"]
    det = it.get("detect", {})
    # a zone is a box, or {"box": [...], <detect keys>} with its own detection (e.g. "fill" for one plate
    # and glyph masks for the others in the same shot)
    zones, zdets = [], []
    for z in (it["zones"] if "zones" in it else [it["zone"]]):
        if isinstance(z, dict):
            zones.append([int(x) for x in z["box"]]); zdets.append({**det, **{k: v for k, v in z.items() if k != "box"}})
        else:
            zones.append([int(x) for x in z]); zdets.append(det)
    clean = it.get("clean", {"mode": "inpaint"})
    mode = clean.get("mode", "inpaint")
    motion = it.get("motion", {"mode": "static"})
    if isinstance(motion, str):
        motion = {"mode": motion}
    moving = motion.get("mode", "static") != "static"
    attached = bool(it.get("attached"))
    ref = clean.get("ref", (f0 + f1) // 2)
    n = f1 - f0
    if det.get("frames"):
        dfr = det["frames"]
    elif attached:
        dfr = [max(f0, ref - 1), min(f1, ref + 1)]      # text moves: detect on the ref frame itself
    else:
        dfr = [f0 + int(0.6 * n), min(f1, f0 + int(0.6 * n) + 10)]
    med_t = ctx.median(*dfr)
    m = np.zeros(coords.shape(), bool)        # glyph masks: k-means patch
    plates = []                               # plates filled to their edges: straight strips
    # a static plate is found (and filled) on the same median as the static patch: the whole plan, less noise
    med_f = ctx.median(*(it.get("plate") or [f0, f1])) if not moving else med_t
    for z, zd in zip(zones, zdets):
        if zd.get("fill") in PLATE_FILLS:
            f = zd["fill"]
            ps, cov, msg = platefill.find_plates(med_f, z, zd, force=None if f == "auto" else f)
            print(f"    zone {z}: {msg}" + ("" if cov >= 0.9 else " -> glyph mask for the rest"), flush=True)
            plates += ps
            if cov < 0.9:
                gm = plate_mask(med_t, z, {**zd, "fill": None})
                for p in ps:
                    gm &= ~platefill.full_mask(p)
                m |= gm
        else:
            m |= plate_mask(med_t, z, zd)
    surf = []          # inside shapes replaced whole by a smooth surface fitted to their own background
    if it.get("inside"):
        # tilted text inside a drawn frame (an oval on a map): boxes would cut into the frame, so the glyph mask
        # is kept inside these shapes - {"ellipse": [cx, cy, a, b, angle_deg]} | {"poly": [[x, y], ...]};
        # "fill": true - the whole shape is replaced by a quadratic surface fitted to its non-glyph pixels (paper
        # inside a ring: the fill never touches the ring or the glyph edges, so no grey seep from them)
        ins = it["inside"] if isinstance(it["inside"], list) else [it["inside"]]
        im = np.zeros(coords.shape(), bool)
        for s in ins:
            sm = np.zeros(coords.shape(), np.uint8)
            if "ellipse" in s:
                cx, cy, a, b, ang = s["ellipse"]
                cv2.ellipse(sm, (int(round(cx)), int(round(cy))), (int(round(a)), int(round(b))), float(ang), 0, 360, 1, -1)
            else:
                cv2.fillPoly(sm, [np.int32(s["poly"])], 1)
            if not s.get("fill"):
                im |= sm.astype(bool)
            else:
                # a filled shape replaces the glyph mask inside it (that mask holds the frame's own strokes)
                sm = sm.astype(bool)
                # the drawn ring may dip into the shape (hand-drawn ovals are not ellipses): dark strokes lying
                # mostly outside the shape are the frame - their pixels near the shape's edge are not filled
                raw = glyph_mask(med_t, zones, {**det, **{k: v for k, v in s.items() if k in ("thr", "maxc", "minc", "win")}})
                n, lab = cv2.connectedComponents(raw.astype(np.uint8))
                inside_px = np.bincount(lab[sm & raw], minlength=n)
                all_px = np.bincount(lab[raw], minlength=n)
                frame_ids = np.where((all_px > 0) & ((all_px - inside_px) > 0.5 * all_px))[0]
                ring = np.isin(lab, frame_ids) & raw
                band = sm & ~cv2.erode(sm.astype(np.uint8), np.ones((2 * int(s.get("band", 12)) + 1,) * 2, np.uint8)).astype(bool)
                # keep the fill a few px off the frame: the moving patch is drawn grown by 3 px and in blocks,
                # touching the frame would redraw its inner edge as steps
                sm = sm & ~(dilate(ring, int(s.get("ring_gap", 5))) & band)
                surf.append((sm, dilate(m & sm, int(s.get("glyph_pad", 4)))))
        m &= im
        for sm, _ in surf:
            m |= sm
    plates = platefill.dedupe(plates)
    enc = it.get("encode", {})
    st, en = v.atime(f0), v.atime(f1 + 1)
    boxes = zones + [[p["off"][0], p["off"][1], p["off"][0] + p["R"].shape[1], p["off"][1] + p["R"].shape[0]]
                     for p in plates]
    bb = [min(z[0] for z in boxes), min(z[1] for z in boxes), max(z[2] for z in boxes), max(z[3] for z in boxes)]

    def inpainted(base):
        keep = np.zeros(coords.shape(), bool)
        for (x0, y0, x1, y1), zd in zip(zones, zdets):
            if zd.get("fill"):
                continue
            # other glyph-like elements around (arrows, frames) must not bleed into the fill
            # (a "fill" zone is drawn inside the frame on purpose: its edge pixels are the boundary)
            keep |= dilate(glyph_mask(base, [(x0 - 80, y0 - 40, x1 + 80, y1 + 30)], {**zd, "thr": 20, "minc": 120}), 3)
        for (x0, y0, x1, y1), zd in zip(zones, zdets):     # a fill zone's own interior is never "kept"
            if zd.get("fill"):
                keep[y0:y1, x0:x1] = False
        P = base
        rest = m.copy()
        for sm, _ in surf:
            rest &= ~sm
        if rest.any():
            for x0, y0, x1, y1 in zones:
                P = harmonic_fill(P, rest | keep, (x0 - 120, y0 - 80, x1 + 120, y1 + 70), iters=2000)
        for sm, gl in surf:
            P = surface_fill(P.copy() if P is base else P, base, sm, gl)
        for p in plates:
            P = platefill.paint(P.copy() if P is base else P, p)
        return P

    if mode == "inpaint" and not moving:
        lines = []
        if m.any():
            P = inpainted(ctx.median(*(it.get("plate") or [f0, f1])))
            lines = patch_lines(P, m, st, en, block=enc.get("block", 3), maxK=enc.get("maxK", 64), per=6, layer=10,
                                feather=enc.get("feather", 8))
        lines += platefill.strip_lines(plates, st, en, layer=10)
        AFF = None
    else:
        pre = clean.get("pre")
        g0 = min([ref] + (list(range(pre[0], pre[1] + 1)) if pre else []) + [f0])
        G = v.grab(g0, f1 - g0 + 1, gray=True)
        emask = np.ones(coords.shape(), np.uint8) * 255
        auto_ex = [] if attached else [[x0 - 60, y0 - 40, x1 + 60, y1 + 20] for x0, y0, x1, y1 in zones]
        for r in motion.get("exclude", []) + auto_ex:
            emask[max(0, r[1]):r[3], max(0, r[0]):r[2]] = 0
        if moving:
            print("    tracking (ECC affine)...", flush=True)
            A = ecc_affines(G, ref - g0, emask, rows=motion.get("rows"), denoise=motion.get("denoise", True))
        else:
            A = [np.eye(2, 3, dtype=np.float32)] * len(G)
        AFF = {g0 + i: a for i, a in enumerate(A)}
        rg = it.get("ring", {})
        ox0, oy0, ox1, oy1 = rg.get("outer") or [bb[0] - 120, bb[1] - 80, bb[2] + 120, bb[3] + 80]
        ipx, ipy = rg.get("inner_pad", [20, 20])
        allm = m.copy()
        for p in plates:
            allm |= platefill.full_mask(p)
        ring = rect_mask([[ox0, oy0, ox1, oy1]]) & ~dilate(allm, 30)
        for x0, y0, x1, y1 in zones:
            ring &= ~rect_mask([[x0 - ipx, y0 - ipy, x1 + ipx, y1 + ipy]])
        if mode == "inpaint":                 # no clean frames: fill the ref frame
            P = inpainted(np.median(v.grab(max(f0, ref - 1), 3), axis=0).astype(np.float32))
        else:
            P = v.frame(ref)
            if pre:
                stack = [P]
                fr = v.grab(pre[0], pre[1] - pre[0] + 1)
                for i, f in enumerate(range(pre[0], pre[1] + 1)):
                    im = warp_from(fr[i].astype(np.float32), AFF[f])
                    cf = colfit(im, P, ring)
                    stack.append(im * cf[:, 0] + cf[:, 1])
                P = np.median(np.stack(stack), axis=0).astype(np.float32)
        mp = m.copy()
        for p in plates:
            mp |= platefill.full_mask(p)
        segs = plate_segments(ctx, it, AFF, mp, f0, f1, moving)
        keys = sorted({k for s in segs for k in s} | set(it.get("color_keys", [])))
        cms = {}
        for k in keys:
            a, b = max(f0, k - 2), min(f1, k + 2)
            tgt = np.median(v.grab(a, b - a + 1), axis=0).astype(np.float32)
            rk = (warp_to(ring.astype(np.float32), AFF[k]) > 0.5) if attached else ring
            cms[k] = colfit(warp_to(P, AFF[k]), tgt, rk)
        if attached:
            need = dilate(m, 3)                # patch shape itself moves with the scene, no clip
            clip = None
        else:
            need = np.zeros(coords.shape(), bool)
            for f in range(f0, f1 + 1, 3):
                need |= warp_from(m.astype(np.float32), AFF[f]) > 0.01
            need = dilate(need, 4)
            clip = m
        # plates: strips carried by the track when attached; a plate in a static overlay stays put
        lines = affine_lines(v, P, AFF, need, clip, segs, cms, block=enc.get("block", 3), maxK=enc.get("maxK", 48),
                             tau=enc.get("tau", 4.0), strips=platefill.strip_groups(plates) if attached else None)
        if plates and not attached:
            lines += platefill.strip_lines(plates, st, en, layer=10)
    texts = specs(it, "texts", "text")
    if attached and AFF is not None:
        for spec in texts:
            lines += follow_affine(ctx, spec, segs, AFF)
    else:
        for spec in texts:
            lines += emit_text(ctx, spec, f0, f1)[0]
    return dict(lines=lines, area=[bb[0] - 80, bb[1] - 80, bb[2] + 80, bb[3] + 80])


def surface_fill(P, base, shape, glyphs):
    """P with `shape` replaced by a quadratic colour surface (per channel) fitted by least squares to the shape's
    pixels outside `glyphs`, with two passes dropping outliers (stray ink, grain specks)"""
    ys, xs = np.where(shape & ~glyphs)
    if len(xs) < 50:
        print("    inside fill: too few background pixels, shape left to the glyph mask", flush=True)
        return P
    cx, cy, s = xs.mean(), ys.mean(), max(xs.std(), ys.std(), 1.0)

    def basis(x, y):
        u, v = (x - cx) / s, (y - cy) / s
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
    A = basis(xs.astype(np.float64), ys.astype(np.float64))
    vals = base[ys, xs].astype(np.float64)
    keep = np.ones(len(xs), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
        r = np.abs(A @ coef - vals).max(axis=1)
        sd = float(np.sqrt(np.mean(r[keep] ** 2))) + 1e-6
        keep = r < 2.5 * sd
    fy, fx = np.where(shape)
    P[fy, fx] = np.clip(basis(fx.astype(np.float64), fy.astype(np.float64)) @ coef, 0, 255).astype(P.dtype)
    print(f"    inside fill: surface over {shape.sum()} px, residual {sd:.1f} levels on {keep.sum()} background px", flush=True)
    return P


def plate_segments(ctx, it, AFF, m, f0, f1, moving):
    """segments of a tracked plate: given in the config, or "auto" - split wherever linear interpolation of
    the track (position, scale, angle) would move the patch's corners off by more than motion.seg_tol px
    (hand-held paper / phone: uneven motion and turning), plus breaks at the texts' fade ends"""
    sg = it.get("segments", [[f0, f1]])
    if sg != "auto":
        return [tuple(s) for s in sg]
    if not moving:
        return [(f0, f1)]
    ys, xs = np.where(m)
    if len(xs) == 0:
        return [(f0, f1)]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pts = [(x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]
    cuts = [c for sp in specs(it, "texts", "text") for c in fade_cuts(ctx.video, sp, f0, f1)]
    mo = it.get("motion", {}) if isinstance(it.get("motion"), dict) else {}
    segs = affine_segments(AFF, f0, f1, pts, tol=mo.get("seg_tol", 0.6), rel=mo.get("seg_rel", 0.1), cuts=cuts)
    print(f"    auto segments: {len(segs)}", flush=True)
    return segs


_POS = re.compile(r"\\pos\(([-\d.]+),([-\d.]+)\)")


_FRZ = re.compile(r"\\frz?(-?[\d.]+)")


def _fade_alpha(v, spec, f0, f1):
    """alpha (0 visible .. 255 hidden) at a time in ms for the item's fade spec, None if no fade"""
    if "fade_frames" in spec:
        a, b = spec["fade_frames"]; tin0, din, dout = v.tms(a), v.tms(b) - v.tms(a), 0.0
    elif "fade" in spec:
        din, dout = [float(x) for x in spec["fade"]]; tin0 = v.tms(f0)
    else:
        return None
    tend = v.tms(f1 + 1)

    def alpha(T):
        a = 0.0
        if din > 0 and T < tin0 + din:
            a = max(a, 1 - max(0.0, T - tin0) / din)
        if dout > 0 and T > tend - dout:
            a = max(a, 1 - max(0.0, tend - T) / dout)
        return int(round(255 * min(1.0, a)))
    return alpha


def fade_cuts(v, spec, f0, f1):
    """frames where a fade ramp starts/ends: segments must break there so the per-segment \\fade is linear"""
    if "fade_frames" in spec:
        return [spec["fade_frames"][1]]
    if "fade" in spec:
        din, dout = spec["fade"]
        out = []
        if din:
            out.append(next((f for f in range(f0, f1 + 1) if v.tms(f) - v.tms(f0) >= din), f1))
        if dout:
            out.append(next((f for f in range(f1, f0 - 1, -1) if v.tms(f1 + 1) - v.tms(f) >= dout), f0))
        return out
    return []


def follow_affine(ctx, spec, segs, AFF):
    """text placed in ref-frame coordinates, then carried by the track: per segment \\move + \\t(\\fscx\\fscy\\frz)
    (translation, scale and rotation of the track; shear is ignored). A fade runs over the whole item: each
    segment gets the part of it that falls in its time (\\fade), so short auto segments do not restart it."""
    v = ctx.video; K = coords.K
    out = []
    f0, f1 = segs[0][0], segs[-1][1]
    alpha = _fade_alpha(v, spec, f0, f1)
    rot = any(abs(affine_params(AFF[f])[4]) > 0.02 for sg in segs for f in sg)
    base = {k: val for k, val in spec.items() if k not in ("fade", "fade_frames")}
    for i, (fa, fb) in enumerate(segs):
        lines, (_, sx) = emit_text(ctx, base, fa, fb)
        Aa, Ab = AFF[fa], AFF[fb]
        _, _, sa, sya, ra = affine_params(Aa)
        _, _, sb, syb, rb = affine_params(Ab)
        st = v.atime(fa)
        h, mi, x = st.split(":")
        t0 = int(h) * 3600000 + int(mi) * 60000 + round(float(x) * 1000)
        t1, t2 = round(v.tms(fa) - t0), round(v.tms(fb) - t0)
        fade = ""
        if alpha is not None:
            ga, gb = alpha(v.tms(fa)), alpha(v.tms(fb + 1))
            if ga or gb:
                dur = round(v.tms(fb + 1) - t0)
                fade = f"\\fade({ga},{gb},{gb},{max(0, t1)},{dur},{dur},{dur})"
        for l in lines:
            mm = _POS.search(l)
            if mm is None:       # \\move or no position: leave the line as it is
                out.append(l); continue
            p = np.array([float(mm.group(1)) / K, float(mm.group(2)) / K, 1.0])
            (xa, ya), (xb, yb) = Aa @ p, Ab @ p
            mv = f"\\move({f2(xa * K)},{f2(ya * K)},{f2(xb * K)},{f2(yb * K)},{t1},{t2})"
            own = _FRZ.search(l[:l.index("}")])
            own = float(own.group(1)) if own else 0.0
            ta, tb = "", ""
            if abs(sa - 1) > 1e-3 or abs(sb - 1) > 1e-3 or abs(sya - 1) > 1e-3:
                ta += f"\\fscx{f2(sx * sa)}\\fscy{f2(100 * sya)}"; tb += f"\\fscx{f2(sx * sb)}\\fscy{f2(100 * syb)}"
            if rot:
                ta += f"\\frz{own - ra:.3f}"; tb += f"\\frz{own - rb:.3f}"
            tr = ta + (f"\\t({t1},{t2},{tb})" if tb and fb > fa else "")
            l = l[:mm.start()] + mv + l[mm.end():]
            k = l.index("}")
            out.append(l[:k] + tr + fade + l[k:])
    return out


# ---------------------------------------------------------------- follow (text riding a zooming logo)
def build_follow(ctx, it):
    """Text that follows a logo's zoom (scale-template track), going behind dark shapes passing in front."""
    v = ctx.video; K = coords.K
    f0, f1 = it["frames"]; tr = it["track"]; ref = tr["ref"]
    G = v.grab(f0, f1 - f0 + 1, gray=True)
    cache = ctx.cache_path(f"follow_{it['id']}_{f0}_{f1}_{ref}.npy")
    if os.path.exists(cache):
        res = np.load(cache)
    else:
        print("    tracking logo (multi-scale template)...", flush=True)
        res = scale_template_track(G, ref - f0, tr["search"], tr.get("dark", 90), tuple(tr.get("scales", [0.86, 1.06, 0.005])))
        np.save(cache + f".{os.getpid()}.tmp.npy", res)
        os.replace(cache + f".{os.getpid()}.tmp.npy", cache)
    fr = np.arange(f0, f1 + 1)
    ok = res[:, 0] > tr.get("min_score", 0.6)
    if "fit_frames" in tr:
        ok &= (fr >= tr["fit_frames"][0]) & (fr <= tr["fit_frames"][1])
    ps, pcx, pcy = (np.polyfit(fr[ok], res[ok, j], 1) for j in (1, 2, 3))
    sref = np.polyval(ps, ref)
    s = lambda f: np.polyval(ps, f) / sref
    c = lambda f: np.array([np.polyval(pcx, f), np.polyval(pcy, f)])
    t = it["text"]
    rec, met = ctx.styles.font(t["style"])
    em = t["cap"] / met["cap"] if "cap" in t else t["em"]
    fs = em * met["ratio"]; desc = fs * met["desc"]
    text = t["text"]
    natural = fonts.text_width(rec, text, em)
    if "span" in t:
        sx0, sx1 = t["span"]
        fsp = (sx1 - sx0 - natural) / max(1, len(text) - 1)
        cx = (sx0 + sx1) / 2 + fsp / 2      # libass adds spacing after the last glyph too
        tw = sx1 - sx0
    else:
        fsp = t.get("spacing", 0.0); cx = t["x"] + fsp / 2; tw = natural + fsp * (len(text) - 1)
    pref = np.array([cx, t["base"] + desc])
    anchor = lambda f: c(f) + s(f) * (pref - c(ref))
    layer = t.get("layer", 30); base_tags = t.get("tags", "")
    size = f"\\fs{f2(fs * K)}\\fsp{f2(fsp * K)}"
    occ = it.get("occluders")
    free_until = occ["from"] - 1 if occ else f1
    out = []

    def ms(n, st):
        h, mi, x = st.split(":")
        return round(v.tms(n) - (int(h) * 3600000 + int(mi) * 60000 + round(float(x) * 1000)))

    fa, fb = f0, min(free_until, f1)
    if fb >= fa:
        st, en = v.atime(fa), v.atime(fb + 1)
        t1, t2 = ms(fa, st), ms(fb, st)
        (xa, ya), (xb, yb) = anchor(fa), anchor(fb)
        sa, sb = 100 * s(fa), 100 * s(fb)
        out.append(f"Dialogue: {layer},{st},{en},{t['style']},Надпись,0,0,0,,{{\\move({f2(xa*K)},{f2(ya*K)},{f2(xb*K)},{f2(yb*K)},{t1},{t2})"
                   f"{base_tags}{size}\\fscx{f2(sa)}\\fscy{f2(sa)}\\t({t1},{t2},\\fscx{f2(sb)}\\fscy{f2(sb)})}}{text}")
    for f in range(free_until + 1, f1 + 1):
        k = s(f); x, y = anchor(f)
        reg = (x - tw * k / 2 - 60, y - desc - t.get("cap", em * 0.7) * k - 40, x + tw * k / 2 + 60, y + 30)
        clip = dark_iclip(G[f - f0], reg, thr=occ.get("thr", 128))
        st, en = v.atime(f), v.atime(f + 1)
        out.append(f"Dialogue: {layer},{st},{en},{t['style']},Надпись,0,0,0,,{{\\pos({f2(x*K)},{f2(y*K)}){base_tags}{size}"
                   f"\\fscx{f2(100*k)}\\fscy{f2(100*k)}" + (f"\\iclip({clip})" if clip else "") + f"}}{text}")
    x, y = pref
    return dict(lines=out, area=[int(x - tw / 2 - 150), int(y - 200), int(x + tw / 2 + 150), int(y + 80)])
