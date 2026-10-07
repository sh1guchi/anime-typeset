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
    v, h = _runs(m, em, axis=1), _runs(m, em, axis=0)
    return {"w": w / em, "contrast": contrast, "rough": perim / perim2, "hist": [round(float(x), 4) for x in hist],
            "v": v / em, "h": h / em, "hv": v / max(h, 0.5)}


def _runs(m, em, axis):
    """median length of the short ink runs along rows (axis=1: = thickness of the vertical strokes) or columns
    (axis=0: = thickness of the horizontal strokes); runs longer than 0.3 em are strokes along the scan, out"""
    a = m if axis == 1 else m.T
    p = np.pad(a.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(p, axis=1)
    starts = np.nonzero(d == 1); ends = np.nonzero(d == -1)
    L = ends[1] - starts[1]
    L = L[(L >= 1) & (L < 0.3 * em)]
    return float(np.median(L)) if len(L) else 1.0


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
    # stroke widths at half contrast, like the crisp candidate renders thresholded at 50%: the detection mask
    # above sits ~18 levels over the background and swallows the blur / glow skirts - thin horizontals come out
    # thick and the stress contrast low
    lum = img[y0:y1, x0:x1].mean(axis=2)
    bgm = ~cv2.dilate(keep, np.ones((7, 7), np.uint8)).astype(bool)
    if bgm.sum() > 50 and keep.sum() > 50:
        bg = float(np.median(lum[bgm]))
        peak = float(np.percentile(lum[keep > 0], 90 if pol == "light" else 10))
        half = (lum > (bg + peak) / 2) if pol == "light" else (lum < (bg + peak) / 2)
        region = cv2.dilate(keep, np.ones((5, 5), np.uint8)).astype(bool)
        hm = (half & region).astype(np.uint8)
        if hm.sum() > 0.3 * keep.sum():
            keep = hm
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
    key = f"v3|{path}|{index}|{int(round(em / 4)) * 4}|{mtime or os.path.getmtime(path):.0f}"
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


def _panose(path, index):
    """category from the font's own PANOSE (OS/2): family type 3 hand-written, 4 decorative; for text faces
    the serif style (11-15 sans / flared / rounded, 2-10 serif); None when unset"""
    from fontTools.ttLib import TTFont, TTCollection
    try:
        ft = TTCollection(path, lazy=True).fonts[index] if path.lower().endswith(".ttc") else TTFont(path, lazy=True)
        pn = ft["OS/2"].panose
    except Exception:
        return None
    if pn.bFamilyType == 3:
        return "handwriting"
    if pn.bFamilyType == 4:
        return "display"
    if pn.bFamilyType == 2:
        if pn.bProportion == 9:
            return "monospace"
        if pn.bSerifStyle in (11, 12, 13, 14, 15):
            return "sans-serif"
        if 2 <= pn.bSerifStyle <= 10:
            return "serif"
    return None


def classify(path, index=0):
    """category of a face: its PANOSE when set, else from the glyphs - serif (feet on the stems of 'п'),
    monospace (equal advances), handwriting (slanted 'l'), else sans-serif. Cached."""
    key = f"cls4|{path}|{index}|{os.path.getmtime(path):.0f}"
    cache = _load_cache()
    if key in cache:
        return cache[key]
    cat = _panose(path, index)
    if cat in (None, "serif", "sans-serif"):          # serif vs sans: PANOSE is often wrong (EuroStyle -> serif)
        cat = "sans-serif"
        try:
            f = ImageFont.truetype(path, 200, index=index)

            def mask(ch):
                l, t_, r, b_ = f.getbbox(ch)
                im = Image.new("L", (int(r - l + 8), int(b_ - t_ + 8)), 0)
                ImageDraw.Draw(im).text((4 - l, 4 - t_), ch, fill=255, font=f)
                a_ = np.asarray(im) > 127
                ys = np.nonzero(a_.any(axis=1))[0]
                return a_[ys[0]:ys[-1] + 1] if len(ys) else a_

            def runs(row):
                p_ = np.diff(np.r_[0, row.astype(np.int8), 0])
                return np.nonzero(p_ == -1)[0] - np.nonzero(p_ == 1)[0]
            pm = mask("п")
            n = len(pm)
            mid = [r_ for row in pm[n // 3: 2 * n // 3] for r_ in runs(row)]
            bot = [r_ for row in pm[-max(2, n // 12):] for r_ in runs(row)]
            stem = np.median(mid) if mid else 1.0
            foot = np.mean(bot) if bot else stem
            lm = mask("l")
            rows = [np.nonzero(r_)[0].mean() for r_ in lm if r_.any()]
            k = max(1, len(rows) // 5)
            slant = (np.mean(rows[:k]) - np.mean(rows[-k:])) / max(1, len(rows)) if len(rows) > 4 else 0.0
            adv = [f.getlength(c) for c in "ilmwЖ"]
            sm, _ = _render(path, index, 120)
            ft = features(sm, 120) or {"rough": 1.0}
            if np.std(adv) < 0.02 * np.mean(adv):
                cat = "monospace"
            elif ft["rough"] > 1.3:                     # brush / distressed edges (Edo, Kashima: ~1.5)
                cat = "display"
            elif abs(slant) > 0.12:
                cat = "handwriting"
            elif foot > 1.5 * stem:
                cat = "serif"
        except Exception:
            pass
    cache[key] = cat
    return cat


def score(f, t, k=1.0):
    """lower = closer: weight (thickness of the vertical stems / em; k = original em / Russian em, so the stems
    compare in pixels on screen - Russian set smaller than the kanji needs a relatively heavier face), stress contrast (vertical / horizontal
    stroke thickness: Mincho and antiqua ~2-3, gothic and grotesque ~1 - the cue that carries over from kanji to
    Cyrillic), edge roughness (brush / distressed vs clean)"""
    # the contrast is reliable on Latin / Cyrillic originals; on kana / kanji (short curved strokes) the run
    # lengths mix the two directions and read ~1.1-1.4 for a Mincho - a light weight there
    return (3.0 * abs(np.log(f["v"] / (t["v"] * k))) + 1.0 * abs(np.log(f["hv"] / t["hv"]))
            + 1.5 * abs(np.log(f["rough"] / t["rough"])))


def candidates(target, em, target_em, text, cats=None, online=True, italic=False, min_px=1.8, log=print,
               orig_em=None):
    """ranked faces: [{name, bold, italic, source, family, weight, score, feat, path?}] + rejected counts.
    em: the original's em in the measured image, target_em / orig_em: Russian / original em on screen"""
    k = orig_em / target_em if orig_em else 1.0
    pool = []
    seen_fam = set()
    try:
        catmap = {c["family"].lower(): c["category"] for c in fontlib.catalog()}
    except Exception:
        catmap = {}
    for r in fonts.index():                       # installed (and already downloaded) faces
        if not r["cyr"] or not r["family"] or r["italic"] != italic:
            continue
        seen_fam.add((r["tfamily"] or r["family"]).lower())
        fam = r["tfamily"] or r["family"]
        c_ = catmap.get(fam.lower()) or catmap.get(fam.lower().rsplit(" ", 1)[0]) or classify(r["path"], r["index"])
        if cats and c_ not in cats:
            continue
        pool.append({"source": "installed", "path": r["path"], "index": r["index"], "family": fam,
                     "weight": r["weight"], "cat": c_, "rec": r})
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
    need = {c for c in text.replace("\\N", " ") if not c.isspace()}     # \N is a line break, not letters
    for p in pool:
        f = measure(p["path"], p["index"], em)
        if not f:
            rej["unreadable"] += 1; continue
        if f["adv"] > 0.8:
            rej["full-width"] += 1; continue
        if f.get("v", f["w"]) * target_em < min_px:
            rej["thin"] += 1; continue
        if p["source"] == "installed":
            if fonts.missing_glyphs(p["rec"], "".join(need)):
                rej["missing letters"] += 1; continue
        out.append(dict(p, feat=f, score=score(f, target, k)))
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
    if "rec" not in c:                            # online: not in the library yet
        c["file"] = fontlib.full_file(c["family"], c["weight"], italic)
        c["name"], c["bold"], c["italic"] = fontlib.ass_face(c["file"])
    else:
        c["file"] = c["path"]
        c["name"], c["bold"], c["italic"] = fontlib.ass_face(c["path"], c["index"])
        # a face installed as part of a family: name it by its family and let the bold flag pick it
        r = c["rec"]
        c["name"] = r["family"]
        c["bold"] = bool(r["bold"])
    # the name libass really finds this file by (some .otf only by the PostScript name on Windows)
    ln = fontlib.libass_name(c["file"], c.get("index", 0), c["name"], c["bold"], italic)
    if ln is None:
        c["warn"] = c.get("warn", []) + ["libass picks another font"]
    else:
        c["name"], c["bold"] = ln
    return c


def specimen(text, out, grep=None, per_page=48, cols=4, dark=True):
    """the whole installed Cyrillic collection set in `text` (\\N = new line), one tile per family (its face
    nearest Regular), pages <out>_1.jpg, ...: for choosing by eye - character, which the strokes don't measure.
    Fonts without the letters or with full-width Cyrillic are left out. Returns (pages, skipped families)"""
    from .inspect import label
    lines = text.replace("\\N", "\n").split("\n")
    need = "".join({c for c in "".join(lines) if not c.isspace()})
    best = {}
    for r in fonts.index():
        if not r["cyr"] or not r["family"] or r["italic"]:
            continue
        fam = r["tfamily"] or r["family"]
        if grep and grep.lower() not in (fam + " " + r["path"]).lower():
            continue
        if fam not in best or abs(r["weight"] - 400) < abs(best[fam]["weight"] - 400):
            best[fam] = r
    W, H, S = 420, 130, 100
    bg, fg = ((32, 32, 32), 255) if dark else ((235, 235, 235), 0)
    tiles, skipped = [], []
    for fam in sorted(best, key=str.lower):
        r = best[fam]
        try:
            if fonts.missing_glyphs(r, need):
                raise ValueError("letters")
            f = ImageFont.truetype(r["path"], S, index=r["index"])
            if np.mean([f.getlength(c) for c in need]) > 0.8 * S:          # full-width Cyrillic
                raise ValueError("full-width")
            cw = int(max(f.getlength(l) for l in lines)) + 2 * S
            im = Image.new("L", (cw, int(S * 1.5 * len(lines)) + 2 * S), 0)
            d = ImageDraw.Draw(im)
            for i, l in enumerate(lines):
                w_ = f.getlength(l)
                d.text(((cw - w_) / 2, S * 1.5 * (i + 1)), l, fill=255, font=f, anchor="ls")
            bb = im.getbbox()
            if not bb:
                raise ValueError("empty")
            im = im.crop(bb)
            k = min((W - 16) / im.width, (H - 30) / im.height)
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
        except Exception:
            skipped.append(fam)
            continue
        t = Image.new("RGB", (W, H), bg)
        ink = Image.new("RGB", im.size, (fg,) * 3)
        t.paste(ink, ((W - im.width) // 2, 20 + (H - 24 - im.height) // 2), im)
        tiles.append(label(t, f"{fam}  ({os.path.basename(r['path'])})"))   # the file: some fonts claim a
        # system family name (an Old English 'old.ttf' calls itself Times New Roman)
    pages = []
    base, ext = os.path.splitext(out)
    for n in range(0, len(tiles), per_page):
        chunk = tiles[n:n + per_page]
        rows = (len(chunk) + cols - 1) // cols
        pg = Image.new("RGB", (W * cols, H * rows), (0, 0, 0))
        for i, t in enumerate(chunk):
            pg.paste(t, ((i % cols) * W, (i // cols) * H))
        pages.append(f"{base}_{n // per_page + 1}{ext or '.jpg'}")
        pg.save(pages[-1], quality=88)
    return pages, skipped


def picked(name, target, em, target_em, text, k=1.0, italic=False, min_px=1.8, bold=None, src="with"):
    """a face chosen by hand, measured like the ranked ones: name 'Family' / 'Family:700', or a style's
    (name, bold=flag) taken as it is. Installed family -> its weight closest to the original, else a Google
    family -> its best weight. None when neither has it; c['warn'] = what candidates() would have rejected"""
    fam, _, w = name.partition(":")
    fam, w = fam.strip(), int(w) if w.strip().isdigit() else None
    need = "".join({c for c in text.replace("\\N", " ") if not c.isspace()})
    if bold is not None:
        r = fonts.lookup(fam, bold, italic)
        recs = [r] if r else []
    else:
        recs = [r for r in fonts.index() if r["italic"] == italic and r["family"]
                and fam.lower() in (r["family"].lower(), (r["tfamily"] or "").lower())]
        if w is not None and recs:
            recs = [min(recs, key=lambda r: abs(r["weight"] - w))]
    opts = [{"source": src, "path": r["path"], "index": r["index"], "family": r["tfamily"] or r["family"],
             "weight": r["weight"], "rec": r} for r in recs]
    if not opts:
        c = next((c for c in fontlib.catalog() if c["family"].lower() == fam.lower()), None)
        if c is None:
            return None
        for w_ in ([w] if w else c["weights"]):
            try:
                opts.append({"source": src + " (Google Fonts)", "path": fontlib.probe_file(c["family"], w_, italic),
                             "index": 0, "family": c["family"], "weight": w_, "cat": c["category"]})
            except Exception:
                pass
    best = None
    for o in opts:
        f = measure(o["path"], o["index"], em)
        if f:
            o.update(feat=f, score=score(f, target, k))
            best = o if best is None or o["score"] < best["score"] else best
    if best is None:
        return None
    f = best["feat"]
    best["warn"] = ([f"thin ({f.get('v', f['w']) * target_em:.1f}px)"] if f.get("v", f["w"]) * target_em < min_px else [])         + (["full-width"] if f["adv"] > 0.8 else [])         + (["missing letters"] if "rec" in best and fonts.missing_glyphs(best["rec"], need) else [])
    _save_cache()
    return best


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
        rec = c.get("rec") or fonts.lookup(c["name"], c["bold"], c.get("italic", False))
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
               + (f", {c['cat']}" if c.get('cat') else "") + f"]  score {c['score']:.2f}"
               + (f"  ! {', '.join(c['warn'])}" if c.get("warn") else ""))
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
