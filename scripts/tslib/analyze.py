"""Stage 1: translator's signs file + video -> per-sign analysis (cuts, motion, detected cards), contact
sheets, and a draft episode.json to be completed by looking at the sheets.

Source signs are often already typeset (Crunchyroll via Erai-raws: \\pos, \\frz, \\fad, \\clip...). Their
geometry is free information: every source line is rendered alone to get its bounding box (where to look
for the original text), and the source typeset is drawn over the middle frame on the sheet."""
import os, re, json, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw
from .video import Video
from .assdoc import read_lines, section, ev_fields, t2cs
from .cards import find_cards
from .platefill import find_plates, stable as plate_stable
from .motion import phase_track
from .inspect import label, tile
from .ctx import read_playres, read_wrapstyle
from . import coords

TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
              ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u",
               "f", "h", "c", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def slug(text, used):
    s = "".join(TR.get(ch, ch) for ch in text.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:24] or "sign"
    base, i = s, 2
    while s in used:
        s = f"{base}-{i}"; i += 1
    used.add(s)
    return s


def strip_tags(t):
    return re.sub(r"\{[^}]*\}", "", t)


def text_parts(t):
    return [p.strip() for p in re.split(r"\\[Nn]", strip_tags(t)) if p.strip()]


def pos_of(t):
    m = re.search(r"\\(?:pos|move)\(\s*([-\d.]+)\s*,\s*([-\d.]+)", t)
    return (float(m.group(1)), float(m.group(2))) if m else None


def detect_cuts(video, a, b):
    a = max(0, a)
    G = video.grab(a, b - a + 1, w=256, h=144, gray=True).astype(np.float32)
    G = 255.0 * np.sqrt(G / 255.0)      # lift shadows: cuts between dark shots count too
    if len(G) < 2:
        return []
    diffs = np.abs(np.diff(G, axis=0)).mean(axis=(1, 2))
    thr = max(18.0, 5 * float(np.median(diffs)))
    return [a + i + 1 for i, dv in enumerate(diffs) if dv > thr]


def motion_summary(video, a, b):
    """Dominant (background) motion: 4x3 tiles tracked frame to frame, a similarity transform fitted with
    RANSAC on the tiles' total displacement, so a moving character does not decide the answer.
    Advisory only - always confirm on the sheet / with `grid`."""
    sh = int(round(480 * coords.AH / coords.AW / 2) * 2)
    G = video.grab(a, b - a + 1, w=480, h=sh, gray=True)
    if len(G) < 3:
        return dict(kind="static", shift=[0, 0], scale=1.0, residual=0.0, inliers=1.0)
    tw, th = 120, 90
    centers, tracks = [], []
    for ty in range(max(1, sh // th)):
        for tx in range(4):
            sub = G[:, ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
            if sub[0].std() < 4:      # flat sky: no information
                continue
            centers.append((tx * tw + tw / 2, ty * th + th / 2))
            tracks.append(phase_track(sub))
    if len(tracks) < 3:
        return dict(kind="unknown", shift=[0, 0], scale=None, residual=None, inliers=0.0)
    src = np.float32(centers); disp = np.float32([t[-1] for t in tracks])
    A, inl = cv2.estimateAffinePartial2D(src, src + disp, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None:
        A = np.float32([[1, 0, np.median(disp[:, 0])], [0, 1, np.median(disp[:, 1])]]); inl = np.ones((len(src), 1))
    inl = inl.ravel().astype(bool)
    scale = float(np.hypot(A[0, 0], A[1, 0]))
    shift = (A @ np.float32([240, sh / 2, 1]) - np.float32([240, sh / 2])) * 4.0
    path = np.median(np.stack([t for t, k in zip(tracks, inl) if k]), axis=0) * 4.0
    tt = np.arange(len(path))
    fit = np.stack([np.polyval(np.polyfit(tt, path[:, c], 1), tt) for c in range(2)], 1)
    resid = float(np.abs(fit - path).max())
    mag = float(np.hypot(*shift))
    if inl.sum() < 4 or not (0.8 < scale < 1.25) or mag > 2000:
        kind = "unknown"   # dark / flat / flashing shot: estimate meaningless, look at the sheet
    elif abs(scale - 1) > 0.01:
        kind = "zoom"
    elif mag < 3:
        kind = "static"
    elif resid < 2.0:
        kind = "linear"
    else:
        kind = "path"
    return dict(kind=kind, shift=[round(float(v), 1) for v in shift], scale=round(scale, 4), residual=round(resid, 2),
                inliers=round(float(inl.mean()), 2))


def _source_header(lines):
    e0 = lines.index("[Events]")
    return lines[:e0 + 2]


def _retimed(raw, dur_cs):
    kind, rest = raw.split(":", 1)
    p = rest.strip().split(",", 9)
    p[1] = "0:00:00.00"
    p[2] = f"0:{dur_cs // 6000 % 60:02d}:{dur_cs // 100 % 60:02d}.{dur_cs % 100:02d}"
    return "Dialogue: " + ",".join(p)


def source_box(header, raw, work):
    """bbox (analysis px) of one already-typeset source line, rendered alone by libass on a grey canvas"""
    _, p = ev_fields(raw)
    dur = max(10, t2cs(p[2]) - t2cs(p[1]))
    tmp = os.path.join(work, "analysis", "_box.ass")
    with open(tmp, "w", encoding="utf-8-sig") as fh:
        fh.write("\n".join(header + [_retimed(raw, dur)]) + "\n")
    W, H = coords.AW, coords.AH
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=0x808080:s={W}x{H}:d={dur / 100 + 1:.2f}:r=24",
           "-vf", "subtitles=_box.ass", "-ss", f"{dur / 100 * 0.6:.3f}", "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, cwd=os.path.join(work, "analysis")).stdout
    if len(out) < W * H * 3:
        return None
    a = np.frombuffer(out[:W * H * 3], np.uint8).reshape(H, W, 3).astype(np.int16)
    ys, xs = np.where(np.abs(a - 128).max(axis=2) > 12)
    if len(ys) < 10:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def glyph_zone(med, box):
    """Where the original text near a translator's box really is (CR puts its line next to / above / below it).
    Near-white (or near-black) neutral strokes around the box are grouped into rows of same-height glyphs;
    a row scores by stroke mass, glyph count and density (kanji fill ~15% of their row box, scenery less),
    less with distance. A thin line under the glyphs (sign arrow line) bounds the zone from below.
    Returns (zone, polarity) or None."""
    from .masks import white_mask, dark_mask
    H, W = coords.AH, coords.AW
    x0, y0, x1, y1 = box
    sx0, sy0 = max(0, x0 - 300), max(0, y0 - 360)
    sx1, sy1 = min(W, x1 + 300), min(H, y1 + 360)
    sub = med[sy0:sy1, sx0:sx1]
    mn, mx = sub.min(axis=2), sub.max(axis=2)
    best = None
    for pol in ("light", "dark"):
        if pol == "light":
            g = white_mask(med, [(sx0, sy0, sx1, sy1)], thr=20, minc=120, win=61)[sy0:sy1, sx0:sx1]
            core = g & (mn >= 220) & (mx - mn < 40)
        else:
            g = dark_mask(med, [(sx0, sy0, sx1, sy1)], thr=25, maxc=150, win=61)[sy0:sy1, sx0:sx1]
            core = g & (mx <= 45) & (mx - mn < 40)
        lines_ = []                            # thin long horizontal lines: kept out of the glyphs
        n, lab, st, _ = cv2.connectedComponentsWithStats(
            cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 121), np.uint8)))
        for i in range(1, n):
            cx, cy, cw, ch, _ = st[i]
            if ch <= 7:
                lines_.append((cx + sx0, cy + sy0, cx + cw + sx0, cy + ch + sy0))
                core[lab == i] = False
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)))
        px = np.bincount(lab[core], minlength=n)
        comps = []
        for i in range(1, n):
            cx, cy, cw, ch, _ = st[i]
            if not (14 <= ch <= 400 and px[i] >= 60 and 0.08 <= px[i] / (cw * ch) <= 0.75 and cw <= 3 * ch):
                continue
            c = [cx + sx0, cy + sy0, cx + cw + sx0, cy + ch + sy0, int(px[i])]
            if not any(l[0] < c[2] and l[2] > c[0] and c[1] < l[1] - 2 and c[3] > l[3] + 2 for l in lines_):
                comps.append(c)                # (ornaments straddling a line are not glyphs)
        row = list(range(len(comps)))

        def root(i):
            while row[i] != i:
                i = row[i]
            return i
        for i, a in enumerate(comps):
            for j in range(i + 1, len(comps)):
                b = comps[j]
                h = max(a[3] - a[1], b[3] - b[1])
                if min(a[3], b[3]) - max(a[1], b[1]) > 0.6 * h and max(b[0] - a[2], a[0] - b[2]) < 1.6 * h:
                    row[root(i)] = root(j)
        rows = {}
        for i, c in enumerate(comps):
            rows.setdefault(root(i), []).append(c)
        for cs in rows.values():
            r = [min(c[0] for c in cs), min(c[1] for c in cs), max(c[2] for c in cs), max(c[3] for c in cs)]
            mass = sum(c[4] for c in cs)
            dens = mass / float((r[2] - r[0]) * (r[3] - r[1]))
            d = np.hypot(max(0, r[0] - x1, x0 - r[2]), max(0, r[1] - y1, y0 - r[3]))
            score = mass * min(len(cs), 4) * min(1.0, dens / 0.12) ** 2 / (1 + d / 100)
            if best is None or score > best[0]:
                best = (score, r, pol, lines_)
    if best is None:
        return None
    _, r, pol, lines_ = best
    pad = int(np.clip(0.15 * (r[3] - r[1]), 22, 60))
    z = [r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad]
    for lx0, ly0, lx1, ly1 in lines_:      # sign line under the glyphs: stop just above it and its soft edge row
        if lx0 < r[2] and lx1 > r[0] and r[3] - 6 <= ly0 <= r[3] + 25:
            z[3] = min(z[3], ly0 - 2)
    return [int(max(0, z[0])), int(max(0, z[1])), int(min(W, z[2])), int(min(H, z[3]))], pol


def steady_gain(sub, p, thr=3.0):
    """How much undoing a patch's own track p (N,2, patch px) steadies it: fraction of pixels within `thr` of
    their temporal median, compensated minus as is. Camera moves: large; background still and the track
    followed a moving character (hair, hand): ~0 or below. Uses the frames still overlapping the middle one."""
    N, h, w = sub.shape
    d = p - p[N // 2]
    use = [i for i in range(N) if abs(d[i, 0]) < 0.4 * w and abs(d[i, 1]) < 0.4 * h]
    if len(use) < 5:
        return 1.0
    du = d[use]
    x0, x1 = int(np.ceil(max(0, -du[:, 0].min()))), int(np.floor(min(w, w - du[:, 0].max())))
    y0, y1 = int(np.ceil(max(0, -du[:, 1].min()))), int(np.floor(min(h, h - du[:, 1].max())))
    S = [cv2.GaussianBlur(sub[i].astype(np.float32), (0, 0), 1.0) for i in use]
    C = [cv2.warpAffine(s, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
         for s, (dx, dy) in zip(S, du)]

    def steady(F):
        F = np.stack(F)[:, y0:y1, x0:x1]
        return float((np.percentile(np.abs(F - np.median(F, axis=0)), 90, axis=0) < thr).mean())
    return steady(C) - steady(S)


def card_motion(v, s0, s1, boxes):
    """Background motion next to the cards (not the whole frame: characters move, the background decides).
    Tracks candidate patches around the cards on half-size frames; static unless undoing the tracks steadies
    the patches (steady_gain); returns a draft motion dict."""
    if s1 - s0 < 3:
        return "static", []
    sw = 960; sh = int(round(sw * coords.AH / coords.AW / 2) * 2)
    G = v.grab(s0, s1 - s0 + 1, w=sw, h=sh, gray=True)
    k = sw / coords.AW
    cands = []
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        for r in ((x0, y0 - 1.3 * h, w, h), (x0, y1 + 10, w, 0.8 * h), (x0 - 0.6 * w, y0, 0.5 * w, h),
                  (x1 + 0.1 * w, y0, 0.5 * w, h)):
            rx, ry, rw, rh = r
            rx0, ry0 = max(0, rx), max(0, ry); rx1, ry1 = min(coords.AW, rx + rw), min(coords.AH, ry + rh)
            if rx1 - rx0 < 120 or ry1 - ry0 < 80:
                continue
            if any(rx0 < b[2] and rx1 > b[0] and ry0 < b[3] and ry1 > b[1] for b in boxes):
                continue
            cands.append([int(rx0), int(ry0), int(rx1 - rx0), int(ry1 - ry0)])
    res = []
    for roi in cands:
        x, y, w, h = [int(round(c * k)) for c in roi]
        sub = G[:, y:y + h, x:x + w]
        if sub[0].std() < 4:
            continue
        p = phase_track(sub) / k
        t = np.arange(len(p))
        fit = np.stack([np.polyval(np.polyfit(t, p[:, c], 1), t) for c in range(2)], 1)
        res.append((roi, p[-1], float(np.abs(fit - p).max()), steady_gain(sub, p * k)))
    if not res:
        return "TODO", []
    shifts = np.array([r[1] for r in res])
    med = np.median(shifts, axis=0)
    gain = float(np.median([r[3] for r in res]))
    if gain < 0.15:
        return "static", res
    best = min(res, key=lambda r: np.linalg.norm(r[1] - med))
    m = {"mode": "linear" if best[2] < 2.0 else "path", "roi": best[0],
         "_analyze": f"shift {np.round(best[1], 1).tolist()}, residual {best[2]:.2f}, {len(res)} patches agree on "
                     f"{np.round(med, 1).tolist()}, compensation steadies the background by {gain:+.2f}"}
    if abs(med[0]) < 5 and abs(med[1]) > 30:
        m["axis"] = "y"
    elif abs(med[1]) < 5 and abs(med[0]) > 30:
        m["axis"] = "x"
    else:
        for minor, major, ax in ((0, 1, "y"), (1, 0, "x")):
            # a small sideways drift on a long tilt/pan is often a wrong 2D peak (rows of buttons); forcing the
            # axis would zero a real diagonal drift, so only suggest it
            if abs(med[major]) > 200 and abs(med[minor]) < 0.02 * abs(med[major]):
                other = "x" if ax == "y" else "y"
                m["_analyze"] += (f"; drift along {other} {med[minor]:.1f} px is small - if `grid` shows none, "
                                  f"set \"axis\": \"{ax}\"")
    return m, res


def run(video_path, signs_path, work, force=False):
    os.makedirs(os.path.join(work, "analysis"), exist_ok=True)
    v = Video(video_path)
    px, py = read_playres(signs_path)
    px, py = px or 384, py or 288
    coords.set_playres(px)
    lines = read_lines(signs_path)
    header = _source_header(lines)
    e0, e1 = section(lines, "[Events]")
    raw_events = [l for l in lines[e0 + 2:e1] if l.startswith("Dialogue:")]
    styles = [l for l in lines if l.startswith("Style:")]
    groups = {}
    for l in raw_events:
        _, p = ev_fields(l)
        groups.setdefault((t2cs(p[1]), t2cs(p[2])), []).append(l)
    used, items, report = set(), [], []
    for (cs0, cs1), raws in sorted(groups.items()):
        evs = [ev_fields(l)[1] for l in raws]
        f0 = v.frame_at(cs0 / 100); f1 = v.frame_at(cs1 / 100) - 1
        cuts = detect_cuts(v, f0 - 24, f1 + 24)
        # snap only small timing errors; signs that start mid-shot (titles, fades) keep the translator's timing
        near = lambda x: min((c for c in cuts if abs(c - x) <= 4), key=lambda c: abs(c - x), default=None)
        c0, c1 = near(f0), near(f1 + 1)
        s0 = c0 if c0 is not None else f0
        s1 = (c1 - 1) if c1 is not None else f1
        inner = [c for c in cuts if s0 < c <= s1]
        mot = motion_summary(v, s0, s1)
        mid = (s0 + s1) // 2
        med = np.median(v.grab(max(s0, mid - 6), min(12, s1 - max(s0, mid - 6) + 1)), axis=0).astype(np.float32)
        cards = find_cards(med)
        texts = [dict(parts=[] if re.search(r"\\p[1-9]", p[9]) else text_parts(p[9]), pos=pos_of(p[9]), style=p[3])
                 for p in evs]          # drawings (CR's own plates / boxes) have no text: no "m-0-0-l-608" ids
        boxes = [source_box(header, l, work) for l in raws]
        typeset = any(re.search(r"\\(pos|move|fr[xyz]?|fax|fay|clip|fn)", p[9]) for p in evs)
        name_for_id = next((t["parts"][-1] for t in texts if t["parts"]), "sign")
        iid = slug(name_for_id, used)
        # contact sheet: around both ends + middle, plus the source typeset rendered over the middle frame
        tw_, th_ = 640, int(round(640 * coords.AH / coords.AW))
        frames = [s0 - 1, s0, mid, s1, s1 + 1]
        ims = []
        raw = {}
        for f in frames:
            raw[f] = v.grab(max(0, f), 1)[0]
            im = Image.fromarray(raw[f]).resize((tw_, th_))
            if f == mid:
                d = ImageDraw.Draw(im)
                for g in cards:
                    d.rectangle([x * tw_ / coords.AW for x in g["box"]], outline=(0, 255, 0), width=2)
                for b in boxes:
                    if b:
                        d.rectangle([x * tw_ / coords.AW for x in b], outline=(255, 128, 0), width=1)
            tag = {s0 - 1: "before", s0: "first", mid: "mid", s1: "last", s1 + 1: "after"}[f]
            ims.append(label(im, f"{f} {tag}"))
        tmp = os.path.join(work, "analysis", "_src.ass")
        with open(tmp, "w", encoding="utf-8-sig") as fh:
            fh.write("\n".join(header + raws) + "\n")
        try:
            p_ = v.render(tmp, [mid], os.path.join(work, "analysis"), prefix="_src")[0]
            ims.append(label(Image.open(p_).convert("RGB").resize((tw_, th_)), f"{mid} source typeset", (255, 160, 0)))
            os.remove(p_)
        except Exception as e:
            print(f"    could not render the source lines: {e}")
        sheet = os.path.join(work, "analysis", f"{iid}.jpg")
        tile(ims, 3, sheet)
        # draft item
        it = {"id": iid, "name": " / ".join(t["parts"][-1] for t in texts if t["parts"]), "frames": [s0, s1],
              "source_frames": [f0, f1], "source_lines": raws, "source_boxes": boxes}
        if cards and all(len(t["parts"]) in (1, 2) for t in texts):
            assigned = []
            pool = list(cards)
            for t, b in zip(texts, boxes):
                if b:
                    x, y = (b[0] + b[2]) / 2, b[3]
                elif t["pos"]:
                    x, y = t["pos"][0] * coords.AW / px, t["pos"][1] * coords.AH / py
                else:
                    x = y = None
                if not pool:
                    break
                g = min(pool, key=lambda g: abs(g["cx"] - x) + 0.5 * abs(g["underline"] - y)) if x is not None else pool[0]
                pool.remove(g)
                card = {"box": g["box"], "name": t["parts"][-1]}
                if len(t["parts"]) == 2:
                    card["title"] = t["parts"][0]
                assigned.append(card)
            if len(assigned) == len(texts):
                it.update(type="card", cards=assigned)
        if it.get("type") == "card":
            # cards: judge the motion by the background around them (moving characters fool the whole frame)
            cm, _ = card_motion(v, s0, s1, [c["box"] for c in it["cards"]])
            it["motion"] = cm
        else:
            # the translator often puts the text next to the original (CR: above / below it): find the glyphs
            zones, pols, zpol = [], [], {}
            for b in boxes:
                if not b:
                    continue
                gz = glyph_zone(med, b)
                if gz:
                    zones.append(gz[0]); pols.append(gz[1]); zpol[tuple(gz[0])] = gz[1]
            zones = [list(z) for z in {tuple(z) for z in zones}]
            # framed plates (labels, cards) that stay put over the shot: fill the whole inside (platefill)
            hinted, n_fill = [], 0
            for z in zones:
                ps, cov, msg = find_plates(med, z, {"polarity": zpol[tuple(z)]})
                st_ = [plate_stable(p, [raw[s0], raw[s1]], med) for p in ps]
                if ps and cov >= 0.9 and all(st_):
                    hinted.append({"box": z, "fill": "auto", "_analyze": msg}); n_fill += 1
                    print(f"    plate fill hint: {msg}", flush=True)
                else:
                    hinted.append(z)
                    if ps and cov >= 0.9:
                        print(f"    plate found but it changes between frames {s0} and {s1} (timing runs into the "
                              f"next sign / the plate moves): {msg}. Fix `frames`, then {{\"box\": ..., \"fill\": "
                              f"\"auto\"}} fills it", flush=True)
            all_fill = bool(zones) and n_fill == len(zones)
            it.update(type="plate" if zones else "TODO plate|text", zones=hinted or "TODO",
                      text={"source": True} if typeset else {"text": "\\N".join(texts[0]["parts"]), "preset": "sign"},
                      detect={"polarity": max(set(pols), key=pols.count) if pols else "TODO light|dark"},
                      clean={"mode": "inpaint" if mot["kind"] == "static" or all_fill else "TODO inpaint|ref (+ref frame)"})
            # a plate that stays put is a static overlay / static shot, whatever the rest of the frame does
            it["motion"] = "static" if all_fill else \
                {"static": "static", "zoom": {"mode": "affine"}, "linear": {"mode": "affine"},
                 "path": {"mode": "affine"}}.get(mot["kind"], "TODO")
        items.append(it)
        report.append((iid, f0, f1, s0, s1, inner, mot, len(cards), typeset,
                       " | ".join(" / ".join(t["parts"]) for t in texts)))
        print(f"  {iid}: frames {s0}-{s1} (source {f0}-{f1}), motion {mot['kind']} {mot['shift']} scale {mot['scale']}, "
              f"cards {len(cards)}, cuts inside {inner}", flush=True)
    for f in ("_box.ass", "_src.ass"):
        try:
            os.remove(os.path.join(work, "analysis", f))
        except OSError:
            pass
    from .ctx import guess_preset
    preset = guess_preset(os.path.basename(v.path), os.path.basename(signs_path), os.path.basename(os.path.dirname(v.path)))
    print(f"  preset: {preset}" + ("  (no title preset matched - set one or start from default)" if preset == "default" else ""))
    cfg = {"video": v.path, "source_signs": os.path.abspath(signs_path), "preset": preset,
           "playres": [px, py], "out": "TODO", "source_styles": styles, "source_events": raw_events,
           "source_wrapstyle": read_wrapstyle(signs_path), "items": items}
    dst = os.path.join(work, "episode.json")
    if os.path.exists(dst) and not force:
        dst = os.path.join(work, "episode.draft.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)
    md = ["| id | frames (src) | snapped | cuts inside | motion | cards | src typeset | text |",
          "|---|---|---|---|---|---|---|---|"]
    for iid, f0, f1, s0, s1, inner, mot, nc, ts, txt in report:
        md.append(f"| {iid} | {f0}-{f1} | {s0}-{s1} | {inner or ''} | {mot['kind']} {mot['shift']} s={mot['scale']} "
                  f"r={mot['residual']} in={mot.get('inliers')} | {nc} | {'yes' if ts else ''} | {txt} |")
    with open(os.path.join(work, "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write(f"video: {v.path}\n\n{v.width}x{v.height} @ {float(v.fps):.3f} fps, matrix {v.ycbcr}\n\n" + "\n".join(md) + "\n")
    return dst
