"""Font matching: rank the installed fonts and the online library by how close their strokes are to the
original (Japanese) text, drop the faces that cannot work, show the best ones over the frame.

Measured on the original's glyph mask and on each face's sample rendered at the same em (so anti-aliasing and
thresholding are comparable): stroke width / em (weight), thick-to-thin ratio (contrast: brush, serif display vs
monoline), edge roughness (brush / distressed vs clean outline). Script differences (kanji vs Cyrillic) cancel
out in these ratios well enough to rank; the final choice is by eye on the sheet - or the user's, when unsure.

Rejected outright: strokes thinner than `min_px` at the size the Russian text will really have (the
"barely visible pencil"), full-width Cyrillic (Japanese fonts often draw Russian letters one per em cell:
'Ц а р с т в о'), letters of the text missing."""
import os, json, hashlib
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor
from . import fonts, fontlib, coords

CATS = ("serif", "sans-serif", "display", "handwriting", "monospace")
HIST_BINS = np.linspace(np.log(0.008), np.log(0.25), 25)
FEAT_CACHE = os.path.join(fontlib.PROBE, "features.json")


def features(m, em):
    """stroke statistics of a binary glyph mask at em size `em` px; None if too little ink"""
    m = (m > 0).astype(np.uint8)
    area = int(m.sum())
    if area < 40:
        return None
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cs)
    if perim <= 0:
        return None
    w = 2.0 * area / perim                                   # mean stroke width of long strokes
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6) & (m > 0) & (dt > 0.6)
    rw = 2.0 * dt[ridge] if ridge.any() else np.array([w])
    contrast = float(np.percentile(rw, 90) / max(np.percentile(rw, 25), 0.75))
    # stroke-width distribution (width / em on log bins): weight and thick/thin contrast in one curve
    hist, _ = np.histogram(np.log(np.clip(rw / em, 1e-3, 1)), bins=HIST_BINS)
    hist = hist / max(1, hist.sum())
    sig = max(1.0, 0.015 * em)
    sm = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), sig) > 0.5).astype(np.uint8)
    cs2, _ = cv2.findContours(sm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim2 = sum(cv2.arcLength(c, True) for c in cs2) or perim
    return {"w": w / em, "contrast": contrast, "rough": perim / perim2, "hist": [round(float(h), 4) for h in hist]}


def original_native(video, frame, box, polarity=None, det=None):
    """original() on the source resolution (a 4K video: 2x the stroke detail); em returned in analysis px,
    features measured at the native em"""
    from . import coords as _c
    s = video.height / _c.AH
    if s < 1.3:
        img = video.frame(frame)
        m, em, pol, col = original(img, box, polarity, det)
        return m, em, em, pol, col
    W, H = video.width, video.height
    x0, y0, x1, y1 = [int(round(v * s)) for v in box]
    big = video.grab(frame, 1, w=W, h=H, crop=(max(0, x0 - 64), max(0, y0 - 64), min(W, x1 + 64), min(H, y1 + 64)))[0]
    full = np.zeros((H, W, 3), np.float32)
    full[max(0, y0 - 64):min(H, y1 + 64), max(0, x0 - 64):min(W, x1 + 64)] = big
    d = dict(det) if det else None
    if d:                                            # detect thresholds are levels: they hold; windows scale
        d["win"] = int(d.get("win", 61) * s) | 1
    m, em_n, pol, col = original(full, [x0, y0, x1, y1], polarity, d, win=int(61 * s) | 1)
    return m, em_n, em_n / s, pol, col


def original(img, box, polarity=None, det=None, win=61):
    """glyph mask of the original text inside box (light or dark text vs the local background) + its em.
    det: the item's `detect` settings (thr / minc / absw / polarity) - the same glyphs the mask erases."""
    x0, y0, x1, y1 = box
    if det:
        from .masks import glyph_mask
        d = dict(det)
        if polarity:
            d["polarity"] = polarity
        pol = d.get("polarity", "light")
        g = glyph_mask(img, [box], d)[y0:y1, x0:x1]
    else:
        lum = img.mean(axis=2)
        loc = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), win).astype(np.float32)
        sub, ls = lum[y0:y1, x0:x1], loc[y0:y1, x0:x1]
        light, dark = (sub - ls) > 18, (ls - sub) > 18
        pol = polarity or ("light" if light.sum() >= dark.sum() else "dark")
        g = light if pol == "light" else dark
    g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(g)
    keep = np.zeros_like(g)
    for i in range(1, n):                        # specks (snow, noise) out
        if st[i, cv2.CC_STAT_AREA] >= 12:
            keep[lab == i] = 1
    rows = keep.sum(axis=1)
    ys = np.where(rows >= max(2, 0.03 * rows.max()))[0] if rows.any() else []
    if len(ys) == 0:
        raise SystemExit(f"no {pol} glyphs found in {box}")
    em = (ys[-1] - ys[0] + 1) / 0.92                         # kanji fill ~92% of the em; one line in the box
    colour = np.median(img[y0:y1, x0:x1][keep > 0], axis=0)
    return keep, em, pol, colour


def _render(path, index, em, text=fontlib.SAMPLE):
    f = ImageFont.truetype(path, int(round(em)), index=index)
    l, t, r, b = f.getbbox(text)
    W, H = int(r - l + 8), int(b - t + 8)
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).text((4 - l, 4 - t), text, fill=255, font=f)
    a = np.asarray(im)
    adv = [f.getlength(ch) / em for ch in "абвгдежзиклмнопрстуфхцчшщыэюя"]
    return a > 127, float(np.median(adv))


def measure(path, index, em, mtime=None):
    key = f"v2|{path}|{index}|{int(round(em / 4)) * 4}|{mtime or os.path.getmtime(path):.0f}"
    cache = _load_cache()
    if key in cache:
        return cache[key]
    try:
        m, adv = _render(path, index, int(round(em / 4)) * 4 or 4)
        f = features(m, int(round(em / 4)) * 4 or 4)
        res = dict(f, adv=adv) if f else None
    except Exception:
        res = None
    cache[key] = res
    return res


_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            with open(FEAT_CACHE, encoding="utf-8") as fh:
                _cache = json.load(fh)
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save_cache():
    if _cache is not None:
        os.makedirs(os.path.dirname(FEAT_CACHE), exist_ok=True)
        tmp = FEAT_CACHE + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_cache, fh)
        os.replace(tmp, FEAT_CACHE)


def score(f, t):
    """lower = closer. Earth mover's distance between the stroke-width distributions (in log-width bins, x0.14
    per bin = the bin width in ln units), mean weight, edge roughness"""
    s = 3.0 * abs(np.log(f["w"] / t["w"])) + 2.0 * abs(np.log(f["rough"] / t["rough"]))
    if f.get("hist") and t.get("hist"):
        step = float(HIST_BINS[1] - HIST_BINS[0])
        s += 3.0 * float(np.abs(np.cumsum(f["hist"]) - np.cumsum(t["hist"])).sum()) * step
    else:
        s += abs(np.log(f["contrast"] / t["contrast"]))
    return s


def candidates(target, em, target_em, text, cats=None, online=True, italic=False, min_px=2.2, log=print):
    """ranked faces: [{name, bold, italic, source, family, weight, score, feat, path?}] + rejected counts"""
    pool = []
    seen_fam = set()
    for r in fonts.index():                       # installed (and already downloaded) faces
        if not r["cyr"] or not r["family"] or r["italic"] != italic:
            continue
        seen_fam.add((r["tfamily"] or r["family"]).lower())
        pool.append({"source": "installed", "path": r["path"], "index": r["index"], "family": r["tfamily"] or r["family"],
                     "weight": r["weight"], "cat": None, "rec": r})
    if online:
        try:
            cat = fontlib.catalog()
        except Exception as e:
            log(f"  online catalog unavailable ({e}) - installed fonts only")
            cat = []
        jobs = [(c["family"], w, c["category"]) for c in cat if c["family"].lower() not in seen_fam
                and (not cats or c["category"] in cats) and ("italic" in c["styles"] if italic else True)
                for w in c["weights"]]

        def fetch(j):
            fam, w, c = j
            try:
                return {"source": "Google Fonts", "path": fontlib.probe_file(fam, w, italic), "index": 0, "family": fam,
                        "weight": w, "cat": c}
            except Exception:
                return None
        with ThreadPoolExecutor(16) as ex:
            got = [g for g in ex.map(fetch, jobs) if g]
        log(f"  online: {len(got)}/{len(jobs)} faces measured" + (f" (categories {', '.join(cats)})" if cats else ""))
        pool += got
    out, rej = [], {"thin": 0, "full-width": 0, "missing letters": 0, "unreadable": 0}
    need = {c for c in text if not c.isspace()}
    for p in pool:
        f = measure(p["path"], p["index"], em)
        if not f:
            rej["unreadable"] += 1; continue
        if f["adv"] > 0.8:
            rej["full-width"] += 1; continue
        if f["w"] * target_em < min_px:
            rej["thin"] += 1; continue
        if p["source"] == "installed":
            if fonts.missing_glyphs(p["rec"], "".join(need)):
                rej["missing letters"] += 1; continue
        out.append(dict(p, feat=f, score=score(f, target)))
    _save_cache()
    out.sort(key=lambda c: c["score"])
    uniq, fams = [], set()
    for c in out:          # best weight of each family, and one family per root (Cormorant / Cormorant SC / ...)
        k = c["family"].lower().split()[0]
        if k in fams:
            continue
        fams.add(k); uniq.append(c)
    n_inst = sum(1 for c in out if c["source"] == "installed")
    log(f"  measured {len(out)} usable faces ({n_inst} installed, {len(out) - n_inst} online), {len(uniq)} families")
    return uniq, rej


def resolve(c, italic=False):
    """download (online) and name the face as a script will: fills c['name'], c['bold'], c['file']"""
    if c["source"] != "installed":
        c["file"] = fontlib.full_file(c["family"], c["weight"], italic)
        c["name"], c["bold"], c["italic"] = fontlib.ass_face(c["file"])
    else:
        c["file"] = c["path"]
        c["name"], c["bold"], c["italic"] = fontlib.ass_face(c["path"], c["index"])
        # a face installed as part of a family: name it by its family and let the bold flag pick it
        r = c["rec"]
        c["name"] = r["family"]
        c["bold"] = bool(r["bold"])
    return c


def sheet(video, frame, crop, cands, text, pos, em, colour, outline, bord, out, workdir, orig_label="original",
          under=None, under_styles=None):
    """original crop + every candidate rendered over the frame at the sign's place, labelled.
    under: script lines drawn first (the item's patch - candidates over the erased frame)"""
    from .assdoc import header, write_lines
    from .inspect import label, tile
    os.makedirs(workdir, exist_ok=True)
    x, b = pos
    ims = [label(Image.fromarray(video.grab(frame, 1, crop=crop)[0]), orig_label)]
    jobs = []
    for i, c in enumerate(cands):
        rec = fonts.lookup(c["name"], c["bold"], c.get("italic", False))
        if rec is None:
            continue
        met = fonts.metrics(rec)
        fs = em * met["ratio"]
        y = b + fs * met["desc"]
        st = (f"Style: T,{c['name']},{fs:.2f},{colour},&H000000FF&,{outline},&H00000000&,{-1 if c['bold'] else 0},"
              f"{-1 if c.get('italic') else 0},0,0,100,100,0,0,1,{bord},0,2,0,0,0,1")
        ev = f"Dialogue: 0,0:00:00.00,9:00:00.00,T,,0,0,0,,{{\\an2\\pos({x:.1f},{y:.1f})}}{text}"
        p = os.path.join(workdir, f"m_{i}.ass")
        if under:
            hdr = header("fonts", under_styles[0], video.ycbcr, under_styles[1] + [st])
            k = under_styles[0][0] / coords.AW            # the patch is in the script's PlayRes, the sample in px
            ev = (f"Dialogue: 30,0:00:00.00,9:00:00.00,T,,0,0,0,,{{\\an2\\pos({x * k:.1f},{y * k:.1f})"
                  f"\\fscx{100 * k:.3f}\\fscy{100 * k:.3f}}}{text}")
            write_lines(p, hdr + under + [ev])
        else:
            write_lines(p, header("fonts", (coords.AW, coords.AH), video.ycbcr, [st]) + [ev])
        lab = (f"{i + 1}. {c['name']}{' Bold' if c['bold'] else ''}  [{c['source']}"
               + (f", {c['cat']}" if c.get('cat') else "") + f"]  score {c['score']:.2f}")
        jobs.append((lab, p, i))

    def one(j):
        lab, p, i = j
        return lab, video.render(p, [frame], workdir, prefix=f"m{i}", crop=crop, jobs=1)[0]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    for lab, p in res:
        ims.append(label(Image.open(p).convert("RGB"), lab))
        os.remove(p)
    ims = [im.resize((960, int(im.height * 960 / im.width))) if im.width > 960 else im for im in ims]
    return tile(ims, 2, out)
