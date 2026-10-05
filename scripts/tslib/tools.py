"""Inspection and config helpers: track, probe, set, split, sheet."""
import os, json, re
import numpy as np, cv2
from PIL import Image
from . import coords
from .motion import phase_track
from .masks import white_mask, dark_mask
from .inspect import label, tile


# ---------------------------------------------------------------- track
def tracks(video, f0, f1, roi, scale):
    """2D and per-axis (profile) tracks of one ROI from a single decode -> {"2d","x","y"}: (N,2) arrays"""
    x, y, w, h = roi
    W, H = int(round(coords.AW * scale)), int(round(coords.AH * scale))
    c = [int(round(v * scale)) for v in (x, y, x + w, y + h)]
    G = video.grab(f0, f1 - f0 + 1, crop=c, w=W, h=H, gray=True)
    return {k: phase_track(G, None if k == "2d" else k) / scale for k in ("2d", "x", "y")}


def lin_res(p):
    t = np.arange(len(p))
    fit = np.stack([np.polyval(np.polyfit(t, p[:, c], 1), t) for c in range(2)], 1)
    return float(np.abs(fit - p).max())


def axis_suspect(two, one):
    """2D phase correlation claims a real drift along an axis that the 1D profile of that axis does not see -
    the typical wrong sideways peak on a tilt over periodic detail. (The other way round is no evidence: a 1D
    profile is unreliable along an axis while the picture scrolls along the other one.)"""
    return abs(two) > 15 and abs(one) < 0.5 * abs(two)


def track_report(video, f0, f1, rois, scales=(0.5, 1.0)):
    """table of total shifts per ROI / scale / method + warnings when they disagree"""
    rows = []
    for roi in rois:
        for s in scales:
            tr = tracks(video, f0, f1, roi, s)
            for k, p in tr.items():
                rows.append((roi, s, k, p[-1], lin_res(p)))
    print(f"frames {f0}-{f1}: total shift (dx, dy) of the background, linear-fit residual")
    print(f"{'roi':>26} {'scale':>5} {'track':>5} {'dx':>8} {'dy':>8} {'resid':>6}")
    for roi, s, k, d, r in rows:
        print(f"{str(roi):>26} {s:5.2f} {k:>5} {d[0]:8.1f} {d[1]:8.1f} {r:6.2f}")
    two = np.array([r[3] for r in rows if r[2] == "2d"])
    xs = np.array([r[3][0] for r in rows if r[2] == "x"]); ys = np.array([r[3][1] for r in rows if r[2] == "y"])
    med2 = np.median(two, axis=0)
    print(f"median 2D shift {np.round(med2, 1).tolist()}, median 1D x {np.median(xs):.1f}, 1D y {np.median(ys):.1f}")
    spread = np.abs(two - med2).max(axis=0)
    if spread.max() > max(8, 0.1 * np.abs(med2).max()):
        print(f"  ! 2D tracks disagree by up to {np.round(spread, 1).tolist()} px - some ROI locks onto a wrong peak "
              "(periodic detail, a moving character); prefer the ROIs that agree")
    for ax, one in ((0, np.median(xs)), (1, np.median(ys))):
        if axis_suspect(med2[ax], one):
            print(f"  ! 2D {'x' if ax == 0 else 'y'} ({med2[ax]:.1f}) != 1D profile ({one:.1f}): "
                  f"if the camera moves along one axis only, use \"axis\": \"{'y' if ax == 0 else 'x'}\"")
    if abs(med2[0]) < 5 and abs(med2[1]) > 30:
        print("  suggestion: vertical move only -> \"axis\": \"y\"")
    elif abs(med2[1]) < 5 and abs(med2[0]) > 30:
        print("  suggestion: horizontal move only -> \"axis\": \"x\"")
    return rows


# ---------------------------------------------------------------- probe
def probe(img, box, bg=None):
    """brightness statistics of glyphs vs background inside box; suggestions for detect thresholds and the
    zone bottom (glyph rows vs a line under them)"""
    x0, y0, x1, y1 = box
    lum = img.mean(axis=2)
    loc = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), 61).astype(np.float32)
    sub, lsub, mn, mx = lum[y0:y1, x0:x1], loc[y0:y1, x0:x1], img.min(axis=2)[y0:y1, x0:x1], img.max(axis=2)[y0:y1, x0:x1]
    light = (sub - lsub) > 15
    dark = (lsub - sub) > 15
    pol = "light" if light.sum() >= dark.sum() else "dark"
    g = light if pol == "light" else dark
    print(f"box {box}: {g.sum()} glyph-like px ({pol} text), {(~g).sum()} other px")
    if g.any():     # stroke colour: the most contrasting third of the glyph pixels (anti-aliased edges left out)
        con = np.abs(sub - lsub)[g]
        rgb = np.median(img[y0:y1, x0:x1][g][con >= np.percentile(con, 67)], axis=0)
        bgc = np.median(img[y0:y1, x0:x1][~g], axis=0)
        r, gg, b = [int(round(v)) for v in rgb]
        br, bgg, bb = [int(round(v)) for v in bgc]
        print(f"  glyph colour RGB ({r},{gg},{b}) = \\c&H{b:02X}{gg:02X}{r:02X}&; "
              f"background RGB ({br},{bgg},{bb}) = &H{bb:02X}{bgg:02X}{br:02X}&")
    q = lambda a, p: float(np.percentile(a, p)) if a.size else float("nan")
    if pol == "light":
        gm, bm = mn[g], mn[~g]
        print(f"  darkest channel: glyphs p20 {q(gm, 20):.0f} / p50 {q(gm, 50):.0f} / p80 {q(gm, 80):.0f};"
              f" background p50 {q(bm, 50):.0f} / p99 {q(bm, 99):.0f}")
        if bg:
            bx0, by0, bx1, by1 = bg
            bb = img.min(axis=2)[by0:by1, bx0:bx1]
            print(f"  bg box {bg}: darkest channel p99 {q(bb, 99):.0f}, max {bb.max():.0f}")
            bgp = max(q(bm, 99), q(bb, 99))
        else:
            bgp = q(bm, 99)
        if q(gm, 20) > bgp:
            minc = int((q(gm, 20) + bgp) / 2)
            print(f"  suggest: \"minc\": {minc}" + (f", \"absw\": {int(q(gm, 50)) - 5}" if q(gm, 50) > 200 else ""))
        else:
            print("  ! glyphs and background overlap in brightness: minc alone cannot separate them - raise thr, "
                  "shrink the zone, or use polarity/dil to taste")
    else:
        gm, bm = mx[g], mx[~g]
        print(f"  brightest channel: glyphs p50 {q(gm, 50):.0f} / p80 {q(gm, 80):.0f}; background p1 {q(bm, 1):.0f} / p50 {q(bm, 50):.0f}")
        print(f"  suggest: \"polarity\": \"dark\", \"maxc\": {int((q(gm, 80) + q(bm, 1)) / 2) if q(gm, 80) < q(bm, 1) else int(q(gm, 90))}")
    rows = g.sum(axis=1)
    runs = []
    for i in range(g.shape[0]):
        r = np.r_[0, g[i].astype(np.int8), 0]; d = np.diff(r)
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        runs.append(int((en - st).max()) if len(st) else 0)
    gl = [i for i in range(len(rows)) if rows[i] > 3 and runs[i] < 0.6 * (x1 - x0)]
    ln = [i for i in range(len(rows)) if runs[i] >= 0.6 * (x1 - x0)]
    if gl:
        print(f"  glyph rows y {y0 + gl[0]}..{y0 + gl[-1]}")
    if ln:
        print(f"  long horizontal line rows y {y0 + ln[0]}..{y0 + ln[-1]}"
              + (f"  -> zone bottom {y0 + ln[0] - 1}" if gl and ln[0] > gl[-1] - 3 else ""))
    frame_interior(img, box)


def _longest_runs(m):
    """longest run of True per row of m"""
    out = np.zeros(m.shape[0], int)
    for i in range(m.shape[0]):
        r = np.r_[0, m[i].astype(np.int8), 0]; d = np.diff(r)
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        out[i] = int((en - st).max()) if len(st) else 0
    return out


def _segs(idx):
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def frame_interior(img, box, frac=0.6):
    """Border lines of a plate / card / speech box inside `box` (long runs of pixels that differ from the plate
    colour: a dark frame on a yellow plate, an orange outline round a cream box) and the interior between the
    innermost lines around the box centre -> a zone that keeps the frame out of the mask."""
    x0, y0, x1, y1 = box
    lum = img.mean(axis=2)
    loc = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), 61).astype(np.float32)
    diff = np.abs(lum - loc)[y0:y1, x0:x1] > 30               # local contrast: gradients inside the plate do not count
    hr = _segs([i for i, r in enumerate(_longest_runs(diff)) if r >= frac * (x1 - x0)])
    vc = _segs([i for i, r in enumerate(_longest_runs(diff.T)) if r >= frac * (y1 - y0)])
    if not hr and not vc:
        return None
    cx, cy = (x1 - x0) / 2, (y1 - y0) / 2
    print("  frame lines: " + ", ".join([f"y {y0 + a}..{y0 + b}" for a, b in hr] + [f"x {x0 + a}..{x0 + b}" for a, b in vc]))
    top = max([b for a, b in hr if b < cy], default=None); bot = min([a for a, b in hr if a > cy], default=None)
    lef = max([b for a, b in vc if b < cx], default=None); rig = min([a for a, b in vc if a > cx], default=None)
    z = [x0 + lef + 2 if lef is not None else x0, y0 + top + 2 if top is not None else y0,
         x0 + rig - 1 if rig is not None else x1, y0 + bot - 1 if bot is not None else y1]
    print(f"  frame interior -> zone {z}" + ("" if None not in (top, bot, lef, rig) else "  (open on some side: box edge used)"))
    return z


# ---------------------------------------------------------------- set / split
def _parse_val(v):
    try:
        return json.loads(v)
    except Exception:
        return v


def apply_patch(c, patch):
    """patch file (UTF-8 JSON, written with an editor - no shell quoting, Cyrillic safe):
    {"@": {top-level keys}, "items": {"<id>": {keys}}}; a key set to null is removed, other values replace
    the old ones whole (dicts are not merged - give the full "text"/"detect"/...); an unknown id with a
    "frames" key is appended as a new item; "<id>": null removes the item (e.g. after joining two signs)"""
    for k, v in (patch.get("@") or {}).items():
        if v is None:
            c.pop(k, None)
        else:
            c[k] = v
    for iid, fields in (patch.get("items") or {}).items():
        if fields is None:
            n = len(c["items"])
            c["items"] = [i for i in c["items"] if i["id"] != iid]
            print(f"  - item {iid}" + ("" if len(c["items"]) < n else " (not found)"))
            continue
        node = next((i for i in c["items"] if i["id"] == iid), None)
        if node is None:
            if "frames" not in fields:
                raise SystemExit(f"no item '{iid}' (a new item needs \"frames\")")
            node = {"id": iid}; c["items"].append(node)
            print(f"  + new item {iid}")
        for k, v in fields.items():
            if v is None:
                node.pop(k, None)
            else:
                node[k] = v
        print(f"  {iid}: {', '.join(fields)}")


def set_values(cfg_path, assignments, patch_file=None):
    """'id.key.sub=value' (item) or '@key=value' (top level); value is JSON or a plain string; index lists with numbers"""
    c = json.load(open(cfg_path, encoding="utf-8"))
    if patch_file:
        apply_patch(c, json.load(open(patch_file, encoding="utf-8-sig")))
    for a in assignments:
        path, val = a.split("=", 1)
        keys = path.split(".")
        if keys[0].startswith("@"):
            node, keys = c, [keys[0][1:]] + keys[1:]
        else:
            node = next((i for i in c["items"] if i["id"] == keys[0]), None)
            if node is None:
                raise SystemExit(f"no item '{keys[0]}'")
            keys = keys[1:]
        for k in keys[:-1]:
            k = int(k) if isinstance(node, list) else k
            if not isinstance(node, list) and k not in node:
                node[k] = {}
            node = node[k]
        last = int(keys[-1]) if isinstance(node, list) else keys[-1]
        if val == "":
            if isinstance(node, list):
                node.pop(last)
            else:
                node.pop(last, None)
            print(f"  removed {path}")
        else:
            node[last] = _parse_val(val)
            print(f"  {path} = {json.dumps(node[last], ensure_ascii=False)}")
    tmp = cfg_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(c, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, cfg_path)


def split_item(cfg_path, iid):
    """one item per card (e.g. three cards with different background tracks); source lines follow their card"""
    c = json.load(open(cfg_path, encoding="utf-8"))
    idx = next((k for k, i in enumerate(c["items"]) if i["id"] == iid), None)
    if idx is None:
        raise SystemExit(f"no item '{iid}'")
    it = c["items"][idx]
    cards = it.get("cards") or []
    if len(cards) < 2:
        raise SystemExit(f"'{iid}' has {len(cards)} card(s), nothing to split")
    sl, sb = it.get("source_lines", []), it.get("source_boxes", [])
    new = []
    for k, card in enumerate(cards):
        n = json.loads(json.dumps(it))
        n["id"] = f"{iid}-{k + 1}"
        n["name"] = card.get("name", n["id"])
        n["cards"] = [card]
        if len(sl) == len(cards):
            n["source_lines"] = [sl[k]]
        if len(sb) == len(cards):
            n["source_boxes"] = [sb[k]]
        new.append(n)
    c["items"][idx:idx + 1] = new
    tmp = cfg_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(c, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, cfg_path)
    print(f"  {iid} -> {', '.join(n['id'] for n in new)} (give each its own motion.roi)")


# ---------------------------------------------------------------- font candidates
def try_fonts(video, frame, text, families, out, pos=None, em=60, colour="&H00303030&",
              outline="&H00FFFFFF&", bord=0.0, tags="", crop=None, workdir=None):
    """Render `text` in every candidate family over one frame (PlayRes = analysis px) and tile the crops with
    the family names: choosing a face by eye without a test script per sign.
    pos=(x, base) - centre x and baseline; em - em size in analysis px (libass sizing as in emit_text)."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from . import fonts
    from .assdoc import header, write_lines
    # own folder per output: parallel calls must not overwrite each other's try_N.ass
    workdir = workdir or os.path.join(os.path.dirname(os.path.abspath(out)),
                                      "_fonts_" + os.path.splitext(os.path.basename(out))[0])
    os.makedirs(workdir, exist_ok=True)
    x, b = pos
    jobs, skipped = [], []
    for fam in families:
        rec = fonts.lookup(fam, "\\b1" in tags)
        if rec is None:
            skipped.append(fam); continue
        met = fonts.metrics(rec)
        miss = fonts.missing_glyphs(rec, text.replace("\\N", ""))
        fs = em * met["ratio"]
        y = b + fs * met["desc"]
        st = (f"Style: T,{fam},{fs:.2f},{colour},&H000000FF&,{outline},&H00000000&,0,0,0,0,100,100,0,0,1,"
              f"{bord},0,2,0,0,0,1")
        ev = f"Dialogue: 0,0:00:00.00,9:00:00.00,T,,0,0,0,,{{\\an2\\pos({x:.1f},{y:.1f}){tags}}}{text}"
        p = os.path.join(workdir, f"try_{len(jobs)}.ass")
        write_lines(p, header("fonts", (coords.AW, coords.AH), video.ycbcr, [st]) + [ev])
        jobs.append((fam + (f"  (no {''.join(miss)})" if miss else ""), p, len(jobs)))
    if skipped:
        print(f"  not installed: {', '.join(skipped)}")

    def one(j):
        name, p, i = j
        return name, video.render(p, [frame], workdir, prefix=f"try{i}", crop=crop, jobs=1)[0]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    ims = []
    for name, p in res:
        im = Image.open(p).convert("RGB")
        if im.width > 960:
            im = im.resize((960, int(im.height * 960 / im.width)))
        ims.append(label(im, name))
        os.remove(p)
    return tile(ims, 2 if len(ims) > 1 else 1, out)


# ---------------------------------------------------------------- sheet
def contact_sheet(ctx, item_ass_fn, out, ids=None):
    """middle frame of every built item with its lines, one image"""
    ims = []
    for it in ctx.items(ids):
        if ctx.load_lines(it["id"]) is None:
            continue
        f0, f1 = it["frames"]
        mid = (f0 + f1) // 2
        p = ctx.video.render(item_ass_fn(ctx, it), [mid], ctx.path("check", "_tmp"), prefix=f"sheet_{it['id']}")[0]
        im = Image.open(p).convert("RGB")
        im = im.resize((640, int(im.height * 640 / im.width)))
        ims.append(label(im, f"{it['id']} {mid}"))
        os.remove(p)
    if not ims:
        raise SystemExit("nothing built yet")
    return tile(ims, 3, out)
