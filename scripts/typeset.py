#!/usr/bin/env python
"""anime-typeset CLI.

  doctor                                   check ffmpeg/libass and python packages
  analyze  --video V --signs S --work DIR  cuts, motion, card detection, sheets, draft DIR/episode.json
  build    -c episode.json [ids..]         build items -> DIR/lines/<id>.json + DIR/check/<id>.jpg
  check    -c episode.json [ids..]         re-render check sheets (--frames a,b,c  --full)
  assemble -c episode.json                write the signs .ass (cfg "out")
  preview  -c episode.json [ids..]         mp4 of every sign with the assembled script (cfg "preview")
  merge    -c episode.json                 put the signs into the dialogue script (cfg "subs"), replacing old signs
  grid     --video V --frames 100:160:6 [--crop x0,y0,x1,y1] --out f.jpg     frames side by side
  ruler    --video V --frames a:b --box x0,y0,x1,y1 --out f.png [--std]      median crop with px rulers
  geom     -c episode.json --frames a:b --box x0,y0,x1,y1 [--no-title]       card geometry for a box
  fonts    [--cyr] [--grep text]                                            installed fonts
  fonts    [-c episode.json] --specimen --text T [--grep ..]                 every installed family set in T: pick by eye
  fonts    -c episode.json --match id --with "A,B:700" [--box ..] [--text ..]  check the picks at the sign (+ current face)
  fonts    -c episode.json --match id --top 8 [--category ..]               + ranking by the original's strokes
  fonts    --get "Family[:weight]"                                           download a Google font into the library
  fonts    --import file.zip|font.ttf|folder ...                            copy downloaded fonts into the library
  fonts    -c episode.json --try "A,B,C" --frame N --text T --pos x,base --em 60 [--colour --outline --bord --tags --crop] --out f.jpg
                                                                           candidate faces rendered over a frame
  place    -c episode.json id --box x0,y0,x1,y1 [--quad ..] [--text ..]     the original's plane + Russian set into it
  track    -c episode.json --frames a:b --roi x,y,w,h [--roi ...]          background track per ROI (2D / x / y), no build
  probe    -c episode.json --frames a:b --box x0,y0,x1,y1 [--bg box]       glyph vs background stats -> detect thresholds
  set      -c episode.json id.key=value [@top=value ...] [--from patch.json] edit the config (JSON values; empty = remove)
  split    -c episode.json id                                             one item per card
  timing   -c episode.json [ids..] [--apply] [--pad 2]                   frames where the original text is on screen
  lama-ab  -c episode.json id [..] [--zoom box]                          the plate with and without LaMa: sheet + seam
  add      -c episode.json --match <regex> [--preset song] [--prefix song] text items from translator lines (actor/style)
  sheet    -c episode.json [ids..] [--out f.jpg]                          middle frame of every built item on one image
  compact  file.ass [file.ass ...]                                        shrink built signs/subs in place, same picture
  (check also takes --zoom x0,y0,x1,y1, --compare: original | result | difference, and --with-others: the other
   built signs on screen at the same frames)
"""
import argparse, json, os, sys, time, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):   # never die on console encoding (cp1251/cp866 pipes)
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


def frames_arg(s):
    if not s:
        return None
    out = []
    for part in s.split(","):
        if ":" in part:
            p = [int(x) for x in part.split(":")]
            out += list(range(p[0], p[1] + 1, p[2] if len(p) > 2 else 1))
        else:
            out.append(int(part))
    return out


def box_arg(s):
    return [int(float(x)) for x in s.split(",")] if s else None


def cfg_path(ctx, key, default=None):
    p = ctx.cfg.get(key) or default
    if not p or p == "TODO":
        return None
    return p if os.path.isabs(p) else os.path.join(ctx.work, p)


def overlapping(ctx, it):
    """other built items on screen at the same time as `it` (their frames overlap)"""
    f0, f1 = it["frames"]
    return [o for o in ctx.items() if o["id"] != it["id"] and o["frames"][0] <= f1 and o["frames"][1] >= f0
            and ctx.load_lines(o["id"]) is not None]


def item_ass(ctx, it, others=False):
    from tslib.assdoc import header, write_lines, ev_style
    from tslib.text import parse_style
    data = ctx.load_lines(it["id"])
    lines = list(data["lines"])
    if others:      # what the viewer really sees: every sign active at these frames, not this one alone
        for o in overlapping(ctx, it):
            lines += ctx.load_lines(o["id"])["lines"]
    used = {ev_style(l) for l in lines}
    # all styles: lines kept from the translator's file use its styles (alignment, margins, fonts)
    styles = [l for l in ctx.all_style_lines if parse_style(l)["Name"] in used]
    p = ctx.path("check", f"_{it['id']}.ass")
    write_lines(p, header(it["id"], ctx.playres, ctx.video.ycbcr, styles) + lines)
    return p


def cmd_doctor(a):
    ok = True
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        has = " subtitles " in out
        print(f"ffmpeg: found, subtitles(libass) filter: {'yes' if has else 'NO'}"); ok &= has
    except FileNotFoundError:
        print("ffmpeg: NOT FOUND in PATH"); ok = False
    for mod, pip in (("numpy", "numpy"), ("cv2", "opencv-python"), ("PIL", "pillow"), ("fontTools", "fonttools")):
        try:
            m = __import__(mod); print(f"{mod}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod}: MISSING  ->  pip install {pip}"); ok = False
    print("OK" if ok else "fix the items above")


def cmd_analyze(a):
    from tslib import analyze
    os.makedirs(a.work, exist_ok=True)
    p = analyze.run(a.video, a.signs, a.work, force=a.force,
                    songs_re=analyze.SONGS_RE if a.songs is None else a.songs)
    print(f"draft config: {p}\nsheets: {os.path.join(a.work, 'analysis')}\nsummary: {os.path.join(a.work, 'analysis.md')}")


# relative build time per frame (measured on BC ep1, 4K source): ECC-tracked plates ~1.4 s/frame,
# moving cards ~0.35-0.7, static cards ~0.3 (per card), follow ~0.05
COST = {"affine": 4.5, "linear": 1.8, "path": 1.0, "static": 1.0, "follow": 0.2, "text": 0.05}


def _cost(it):
    """estimated build time, so the longest items start first (the slowest one sets the wall clock)"""
    m = it.get("motion", "static")
    m = m.get("mode", "static") if isinstance(m, dict) else m
    n = it["frames"][1] - it["frames"][0] + 1
    w = COST.get(it["type"], 1.0) if m == "static" else COST.get(m, 1.0)
    return n * w * max(1, len(it.get("cards", [1])))


def _build_one(job):
    """one item in a worker process; returns (id, log text)"""
    cfg, iid, no_check = job
    import io, contextlib, traceback
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            from tslib.ctx import Ctx
            from tslib.builders import build_item
            from tslib.inspect import check_sheet
            ctx = Ctx(cfg)
            it = next(i for i in ctx.items() if i["id"] == iid)
            t = time.time()
            print(f"[{iid}] {it['type']} frames {it['frames'][0]}-{it['frames'][1]}")
            data = build_item(ctx, it)
            data["frames"] = it["frames"]
            ctx.save_lines(iid, data)
            print(f"    {len(data['lines'])} lines, {sum(len(l) for l in data['lines']) // 1024} KB, {time.time() - t:.0f}s")
            if not no_check:
                print(f"    check: {check_sheet(ctx, it, item_ass(ctx, it), ctx.path('check', f'{iid}.jpg'))}")
        except SystemExit as e:
            print(f"[{iid}] ERROR: {e}")
        except Exception:
            print(f"[{iid}] ERROR:\n{traceback.format_exc()}")
    return iid, buf.getvalue()


def cmd_build(a):
    from tslib.ctx import Ctx
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = Ctx(a.config)
    its = sorted(ctx.items(a.ids), key=_cost, reverse=True)      # heavy items first: better packing
    jobs = [(ctx.cfg_path, it["id"], a.no_check) for it in its]
    n = max(1, min(a.jobs or 4, len(jobs)))
    t0 = time.time()
    if n == 1:
        for j in jobs:
            print(_build_one(j)[1], end="", flush=True)
    else:
        print(f"building {len(jobs)} items in {n} parallel workers (heaviest first)...", flush=True)
        with ProcessPoolExecutor(n) as ex:
            for fut in as_completed([ex.submit(_build_one, j) for j in jobs]):
                print(fut.result()[1], end="", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


def cmd_check(a):
    from tslib.ctx import Ctx
    from tslib.inspect import check_sheet
    ctx = Ctx(a.config)
    for it in ctx.items(a.ids):
        if ctx.load_lines(it["id"]) is None:
            print(f"[{it['id']}] not built"); continue
        suffix = ""
        if a.compare:
            suffix += "_cmp"
        if a.zoom:
            suffix += "_zoom"
        if a.with_others:
            suffix += "_all"
            ov = overlapping(ctx, it)
            print(f"[{it['id']}] together with: {', '.join(o['id'] for o in ov) or 'nothing else'}")
        out = check_sheet(ctx, it, item_ass(ctx, it, others=a.with_others), ctx.path("check", f"{it['id']}{suffix}.jpg"),
                          frames=frames_arg(a.frames), full=a.full, zoom=box_arg(a.zoom), compare=a.compare)
        print(out)


def cmd_assemble(a):
    from tslib.ctx import Ctx
    from tslib.assdoc import assemble
    ctx = Ctx(a.config)
    out = a.out or cfg_path(ctx, "out")
    if not out:
        raise SystemExit("set 'out' in episode.json or pass --out")
    n = assemble(ctx, out)
    print(f"{n} events -> {out} ({os.path.getsize(out) // 1024} KB)")
    # signs that share the screen: checks render each alone, so a clash (two lines on one spot) shows only here
    seen = set()
    for it in ctx.items():
        if ctx.load_lines(it["id"]) is None:
            continue
        A = ctx.load_lines(it["id"]).get("area")
        for o in overlapping(ctx, it):
            B = ctx.load_lines(o["id"]).get("area")
            key = tuple(sorted((it["id"], o["id"])))
            if key in seen or not A or not B:
                continue
            seen.add(key)
            if A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]:
                print(f"  on screen together and in the same area: {it['id']} + {o['id']} - look at "
                      f"`TS check -c ... {it['id']} --with-others`")


def cmd_preview(a):
    from tslib.ctx import Ctx
    ctx = Ctx(a.config)
    ass = a.ass or cfg_path(ctx, "out")
    out = a.out or cfg_path(ctx, "preview") or ctx.path("preview.mp4")
    pad = int(round(float(ctx.video.fps) * 0.4))
    rs = sorted((max(0, i["frames"][0] - pad), i["frames"][1] + pad) for i in ctx.items(a.ids))
    merged = []
    for r in rs:
        if merged and r[0] <= merged[-1][1] + int(float(ctx.video.fps) * 2):
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)
    ctx.video.preview(ass, merged, out, ctx.path("cache"))
    print(out)


def cmd_merge(a):
    from tslib.ctx import Ctx
    from tslib.assdoc import merge
    ctx = Ctx(a.config)
    subs = a.subs or cfg_path(ctx, "subs")
    signs = a.signs or cfg_path(ctx, "out")
    if not subs or not signs:
        raise SystemExit("need 'subs' and 'out' in episode.json (or --subs/--signs)")
    drop = a.drop_styles.split(",") if a.drop_styles else (ctx.cfg.get("merge_drop_styles") or [])
    # only the translator lines of the signs we actually replaced; the rest stay in the subs untouched
    built = [i for i in ctx.items() if ctx.load_lines(i["id"]) is not None]
    if any("source_lines" in i for i in ctx.cfg["items"]):
        src = [l for i in built for l in i.get("source_lines", [])]
    else:   # older configs without per-item source lines: the whole source file
        src = ctx.cfg.get("source_events") or cfg_path(ctx, "source_signs")
    r = merge(signs, subs, ctx.video.ycbcr, source_signs=src, drop_styles=drop)
    print(f"removed {r['removed']} old sign lines (same as in the signs file / previous merge"
          + (f" / styles {drop}" if drop else "") + f"); inserted {r['inserted']} lines in {r['blocks']} blocks"
          + (f"; unused styles removed {r['styles_removed']}" if r["styles_removed"] else "")
          + (f"; dialogue layers +{r['dialogue_bump']}" if r["dialogue_bump"] else "") + (f"; backup {r['backup']}" if r["backup"] else ""))
    ok = r["dialogue_before"] == r["dialogue_after"] and r["dialogue_same"]
    print(f"dialogue lines: {r['dialogue_before']} -> {r['dialogue_after']}" + (
        "  OK: text, times, style, actor and order unchanged" + (f" (only Layer +{r['dialogue_bump']})" if r["dialogue_bump"] else "")
        if ok else f"  <-- CHANGED (first difference at dialogue line {r['dialogue_diff']}), check the subs!"))


def _video(a):
    """inspection commands need only the video: no preset / styles (works on a fresh analyze draft)"""
    from tslib.video import Video
    if a.video:
        return Video(a.video)
    if not a.config:
        raise SystemExit("give --video or -c episode.json")
    cfg = json.load(open(a.config, encoding="utf-8"))
    p = cfg["video"]
    return Video(p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(a.config)), p))


def cmd_grid(a):
    from tslib.inspect import grid
    print(grid(_video(a), frames_arg(a.frames), a.out, crop=box_arg(a.crop), cols=a.cols, width=a.width))


def cmd_ruler(a):
    import numpy as np
    from tslib.inspect import ruler
    v = _video(a)
    fr = frames_arg(a.frames)
    G = v.grab(fr[0], fr[-1] - fr[0] + 1, step=max(1, (fr[-1] - fr[0] + 1) // 40)).astype(np.float32)
    med = np.median(G, axis=0)
    std = G.std(axis=0).mean(axis=2) if a.std else None
    print(ruler(med, box_arg(a.box), a.out, scale=a.scale, std=std))


def cmd_geom(a):
    from tslib.ctx import Ctx
    from tslib.cards import card_geom, zones_from_geom
    ctx = Ctx(a.config)
    fr = frames_arg(a.frames)
    try:
        g = card_geom(ctx.median(fr[0], fr[-1]), box_arg(a.box), not a.no_title)
    except Exception as e:
        raise SystemExit(f"card geometry not found in {a.box} ({type(e).__name__}): light background or the box misses "
                         "the card. Measure with `ruler` and give geom_only + geom {cx, line, underline, L, R, fl_in, fr_in}"
                         " + zones + glow_box by hand.")
    print(json.dumps({k: round(v, 1) for k, v in g.items()}, ensure_ascii=False))
    print("zones:", zones_from_geom(g))


def _item_text(it):
    """the Russian text of an item (first text spec / card name), for font samples"""
    for k in ("text", "texts", "plain"):
        v = it.get(k)
        for s in (v if isinstance(v, list) else [v]):
            if isinstance(s, dict) and s.get("text"):
                return s["text"]                     # \N kept: the sheet shows the lines as the sign has them
    for c in it.get("cards", []):
        return c.get("name", "")
    return it.get("name", "")


def _item_fonts(ctx, it):
    """(family, bold) of the styles the item's text is set in now (its specs' style / preset layers)"""
    specs = []
    for k in ("text", "texts", "plain"):
        v = it.get(k)
        specs += [s for s in (v if isinstance(v, list) else [v]) if isinstance(s, dict) and s.get("text")]
    names = []
    for s in specs:
        lay = s.get("layers") or (ctx.preset.get(s.get("preset", "sign")) or {}).get("layers") or []
        names.append(s.get("style") or (lay[-1]["style"] if lay else None))
    if it.get("cards"):
        lay = (ctx.preset.get("card") or {}).get("layers") or []
        names.append(lay[-1]["style"] if lay else None)
    out = []
    for n in names:
        if n and n in ctx.styles.lines:
            d = ctx.styles.get(n)
            fb = (d["Fontname"], d["Bold"] not in ("0", ""))
            if fb not in out:
                out.append(fb)
    return out


def cmd_fonts_match(a):
    """rank the installed + online fonts by how close their strokes are to the original text, sheet of the best"""
    import json as _json
    import numpy as np
    from tslib.ctx import Ctx
    from tslib import fontmatch
    ctx = Ctx(a.config)
    v = ctx.video
    it = next((i for i in ctx.cfg["items"] if i["id"] == a.match), None) if a.match else None
    if a.match and it is None:
        raise SystemExit(f"no item {a.match}")
    box = box_arg(a.box) if a.box else None
    if box is None and it is not None:
        z = it.get("zones", [it.get("zone")]) if (it.get("zones") or it.get("zone")) else []
        z = z[0] if z else None
        box = [int(c) for c in (z["box"] if isinstance(z, dict) else z)] if z else None
    if box is None:
        raise SystemExit("give --box x0,y0,x1,y1 around ONE line of the original text")
    frame = a.frame if a.frame is not None else ((it["frames"][0] + it["frames"][1]) // 2 if it else None)
    if frame is None:
        raise SystemExit("give --frame")
    det = it.get("detect") if it and it.get("type") == "plate" and not a.box else None
    mask, em_n, em0, pol, col = fontmatch.original_native(v, frame, box, a.polarity, det)
    target = fontmatch.features(mask, em_n)
    text = a.text or (_item_text(it) if it else "") or fontmatch.fontlib.SAMPLE
    cats = [c.strip() for c in a.category.split(",")] if a.category else None
    # the Russian size: cap height ~0.78 of the kanji (as on the cards / place signs), and no wider than the
    # original line by more than 10% (measured with a typical face; the sheet shows each face at this em)
    from tslib import fonts as _f
    ys, xs = np.nonzero(mask)
    ink_w = (xs.max() - xs.min() + 1) * em0 / em_n if len(xs) else (box[2] - box[0])
    ref = _f.lookup("Arial") or (_f.index()[0] if _f.index() else None)
    em = a.em or 0.78 * em0
    if not a.em and ref is not None:
        tw = max(_f.text_width(ref, part, em) for part in text.split("\\N"))
        if tw > 1.1 * ink_w:
            em *= 1.1 * ink_w / tw
    print(f"original: {pol} text, em {em0:.0f} px, stems {target['v'] * em0:.1f} px (the Russian stems are "
          f"matched to it in px), stress contrast "
          f"(vertical / horizontal) {target['hv']:.2f}, roughness {target['rough']:.3f}; Russian at em {em:.0f}")
    # with picks by eye the sheet checks them, no ranked faces (on BC 2-01 none of those fitted): --top N adds them
    n_top = a.top if a.top is not None else (0 if a.with_ else 8)
    cands = []
    if n_top:                       # the ranking measures the whole catalog (the first run ~10 min, then cached)
        cands, rej = fontmatch.candidates(target, em_n, em, text, cats=cats, online=not a.offline,
                                          italic=a.italic, min_px=a.min_px, orig_em=em0)
        print("  rejected: " + ", ".join(f"{k} {n}" for k, n in rej.items() if n))
    # the faces the item uses now (preset styles) and the ones named with --with (by eye, by genre - references/
    # fonts.md) go first on the sheet, whatever they score: the strokes don't measure character (BC title: Moyenage)
    cur = [(fam, b, "current") for fam, b in (_item_fonts(ctx, it) if it else [])]
    cur += [(f.strip(), None, "with") for f in (a.with_ or "").split(",") if f.strip()]
    pinned = []
    for fam, b, src in cur:
        c = fontmatch.picked(fam, target, em_n, em, text, em0 / em, a.italic, a.min_px, bold=b, src=src)
        if c is None:
            print(f"  {src} {fam}: neither installed nor in the Google catalog")
            continue
        print(f"  {src} {fam}: stems {c['feat']['v'] * em:.1f}px (original {target['v'] * em0:.1f}px), score "
              f"{c['score']:.2f}" + (f" - would rank #{1 + sum(x['score'] < c['score'] for x in cands)} of "
                                     f"{len(cands)}" if cands else "")
              + (f" ! {', '.join(c['warn'])}" if c["warn"] else ""))
        pinned.append(c)
    roots = {p["family"].lower().split()[0] for p in pinned}
    top = pinned + [c for c in cands if c["family"].lower().split()[0] not in roots][:n_top]
    for c in top:
        fontmatch.resolve(c, a.italic)
    for i, c in enumerate(top, 1):
        f = c["feat"]
        print(f"  {i}. {c['name']}{' (bold)' if c['bold'] else ''}  [{c['source']}{', ' + c['cat'] if c.get('cat') else ''}]"
              f"  score {c['score']:.2f}  stems {f['v'] * em:.1f}px contrast {f['hv']:.2f} rough {f['rough']:.3f}"
              + (f"  ! {', '.join(c['warn'])}" if c.get("warn") else ""))
    ranked = [c for c in top if c not in pinned]
    if pinned or (len(ranked) > 1 and ranked[1]["score"] - ranked[0]["score"] < 0.15):
        print("  " + ("the current face and the picks are on the sheet: choose by eye (stems much thinner than the "
                      "original - a heavier weight or \\bord); several fit" if pinned else "the best ones are close")
              + " - show the sheet to the user and ask (SKILL.md, font choice)")
    colour = a.colour or "&H00{:02X}{:02X}{:02X}&".format(*[int(round(x)) for x in col[::-1]])
    x0, y0, x1, y1 = box
    if len(xs):                                       # baseline: kanji sit ~0.12 em below it
        ink_bot = box[1] + ys.max() * em0 / em_n
        ink_cx = box[0] + (xs.min() + xs.max()) / 2 * em0 / em_n
    else:
        ink_bot, ink_cx = y1, (x0 + x1) / 2
    pos = [float(p) for p in a.pos.split(",")] if a.pos else (ink_cx, ink_bot - 0.12 * em0)
    pad_x, pad_y = max(60, (x1 - x0) // 3), max(40, (y1 - y0))
    crop = box_arg(a.crop) if a.crop else [max(0, x0 - pad_x), max(0, y0 - pad_y), min(coords_aw(), x1 + pad_x), min(coords_ah(), y1 + pad_y)]
    out = a.out or ctx.path("check", f"fonts_{a.match or 'box'}.jpg")
    under = under_st = None
    if it is not None and ctx.load_lines(it["id"]) is not None:       # built: show the faces over the erased frame
        from tslib.assdoc import ev_style
        from tslib.text import parse_style
        under = [l for l in ctx.load_lines(it["id"])["lines"] if ev_style(l) == "Маска"]
        under_st = (ctx.playres, [l for l in ctx.all_style_lines if parse_style(l)["Name"] == "Маска"])
    fontmatch.sheet(v, frame, crop, top, text, pos, em, colour, a.outline, a.bord, out,
                    ctx.path("check", "_fonts_" + os.path.splitext(os.path.basename(out))[0]),   # own per sheet
                    orig_label=f"original ({pol}, em {em0:.0f})",
                    under=under, under_styles=under_st)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            _json.dump([{k: c[k] for k in ("name", "bold", "italic", "source", "family", "weight", "score", "file")
                         if k in c} for c in top], fh, ensure_ascii=False, indent=1)
    print(out)


def cmd_place(a):
    """the original text's plane (quad) + the Russian text set into it, over the frame: for signs at an angle"""
    import json as _json
    import numpy as np
    from PIL import Image, ImageDraw
    from tslib.ctx import Ctx
    from tslib import fontmatch, persp, coords
    from tslib.assdoc import header, write_lines, ev_style
    from tslib.text import emit_text, parse_style
    from tslib.inspect import label, tile, fit_area
    ctx = Ctx(a.config)
    v = ctx.video
    it = next((i for i in ctx.cfg["items"] if i["id"] == a.item), None)
    if it is None:
        raise SystemExit(f"no item {a.item}")
    frame = a.frame if a.frame is not None else (it["frames"][0] + it["frames"][1]) // 2
    if a.quad:
        q = [float(x) for x in a.quad.split(",")]
        quad = [[q[0], q[1]], [q[2], q[3]], [q[4], q[5]], [q[6], q[7]]]
        info = {}
    else:
        if not a.box:
            raise SystemExit("give --box around ONE line of the original text (or --quad)")
        box = box_arg(a.box)
        det = it.get("detect") if it.get("type") == "plate" else None
        mask, em_n, em0, pol, col = fontmatch.original_native(v, frame, box, a.polarity, det)
        s = em0 / em_n                                   # native px -> analysis px
        quad, info = persp.auto_quad(mask, perspective=a.perspective)
        quad = [[box[0] + x * s, box[1] + y * s] for x, y in quad]
    print("quad (analysis px, TL TR BR BL): " + _json.dumps([[round(x, 1), round(y, 1)] for x, y in quad]))
    if info:
        print(f"  text direction {info['angle']:+.2f} deg" + (f", top / bottom lines converge: slopes "
              f"{info['top_slope']:+.4f} / {info['bottom_slope']:+.4f} (mean residual {info['top_err']:.1f} / "
              f"{info['bottom_err']:.1f} px) - check the quad on the sheet" if info["perspective"] else
              " (rotated rectangle; --perspective for converging lines, or --quad from the sign's own edges)"))
    spec = {"text": a.text or _item_text(it), "quad": quad, "preset": a.preset}
    for k in ("cap", "base_frac", "maxw_frac"):
        if getattr(a, k) is not None:
            spec[k] = getattr(a, k)
    lines, _ = emit_text(ctx, spec, frame, frame)
    under = []
    if ctx.load_lines(it["id"]) is not None:
        under = [l for l in ctx.load_lines(it["id"])["lines"] if ev_style(l) == "Маска"]
    used = {ev_style(l) for l in lines + under}
    styles = [l for l in ctx.all_style_lines if parse_style(l)["Name"] in used]
    p = ctx.path("check", f"_place_{it['id']}.ass")
    write_lines(p, header("place", ctx.playres, v.ycbcr, styles) + under + lines)
    xs = [x for x, _ in quad]; ys = [y for _, y in quad]
    crop = fit_area([min(xs) - 80, min(ys) - 60, max(xs) + 80, max(ys) + 60])
    r = Image.open(v.render(p, [frame], ctx.path("check", "_tmp"), prefix="place", crop=crop)[0]).convert("RGB")
    o = Image.fromarray(v.grab(frame, 1, crop=crop)[0])
    d = ImageDraw.Draw(o)
    d.line([(x - crop[0], y - crop[1]) for x, y in quad + [quad[0]]], fill=(0, 255, 0), width=2)
    out = a.out or ctx.path("check", f"place_{it['id']}.jpg")
    tile([label(o, "original + quad"), label(r, "Russian in the quad")], 1, out)
    print(f"  config: \"text\": {{..., \"quad\": {_json.dumps([[round(x, 1), round(y, 1)] for x, y in quad])}}}")
    print(out)


def coords_aw():
    from tslib import coords
    return coords.AW


def coords_ah():
    from tslib import coords
    return coords.AH


def cmd_fonts(a):
    from tslib import fonts
    if a.match or a.box:
        return cmd_fonts_match(a)
    if a.specimen:
        from tslib import fontmatch
        if not a.text:
            raise SystemExit("--specimen needs --text (the sign's Russian text)")
        out = a.out or (os.path.join(os.path.dirname(os.path.abspath(a.config)), "check", "fonts_specimen.jpg")
                        if a.config else "fonts_specimen.jpg")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        pages, skipped = fontmatch.specimen(a.text, out, a.grep, dark=not a.light)
        print(f"{len(skipped)} families left out (no letters / full-width / unreadable)")
        print("\n".join(pages))
        return
    if a.import_:
        from tslib import fontlib
        got = fontlib.import_fonts(a.import_)
        for f, recs in got:
            for r in recs:
                n, b = fontlib.libass_name(r["path"], r["index"], r["family"], bool(r["bold"]), bool(r["italic"])) or \
                    (r["family"] + "  ! libass picks another font", r["bold"])
                print(f"{os.path.basename(f)}: '{n}'{' Bold' if b else ''}  ({'cyr' if r['cyr'] else 'NO Cyrillic'})"
                      f"  -> --with \"{r['tfamily'] or r['family']}\"")
        print(f"{len(got)} font files in {fontlib.library_dir()}")
        return
    if a.get:
        from tslib import fontlib
        fam, _, w = a.get.partition(":")
        p = fontlib.full_file(fam.strip(), int(w or 400), a.italic)
        name, bold, ital = fontlib.ass_face(p)
        print(f"{p}\n  in a script: Fontname '{name}', Bold {-1 if bold else 0}, Italic {-1 if ital else 0}")
        return
    if a.try_:
        from tslib.tools import try_fonts
        if not (a.frame is not None and a.text and a.pos and a.out):
            raise SystemExit("--try needs --frame, --text, --pos x,base and --out")
        x, b = [float(v) for v in a.pos.split(",")]
        print(try_fonts(_video(a), a.frame, a.text, [f.strip() for f in a.try_.split(",") if f.strip()], a.out,
                        pos=(x, b), em=a.em or 60, colour=a.colour or "&H00303030&", outline=a.outline, bord=a.bord,
                        tags=a.tags or "",
                        crop=box_arg(a.crop)))
        return
    seen = set()
    for r in sorted(fonts.index(), key=lambda r: (r["family"].lower(), r["sub"])):
        if a.cyr and not r["cyr"]:
            continue
        key = (r["family"], r["sub"])
        if key in seen or (a.grep and a.grep.lower() not in (r["family"] + r["sub"] + r["path"]).lower()):
            continue
        seen.add(key)
        print(f"{r['family']} | {r['sub']} | {'cyr' if r['cyr'] else '   '} | {os.path.basename(r['path'])}")


def cmd_track(a):
    from tslib.tools import track_report
    v = _video(a)
    fr = frames_arg(a.frames)
    track_report(v, fr[0], fr[-1], [box_arg(r) for r in a.roi], scales=tuple(a.scale or [0.5, 1.0]))


def cmd_probe(a):
    import numpy as np
    from tslib.tools import probe
    v = _video(a)
    fr = frames_arg(a.frames)
    G = v.grab(fr[0], fr[-1] - fr[0] + 1, step=max(1, (fr[-1] - fr[0] + 1) // 20)).astype(np.float32)
    probe(np.median(G, axis=0), box_arg(a.box), box_arg(a.bg))


def cmd_set(a):
    from tslib.tools import set_values
    if not a.assign and not a.from_:
        raise SystemExit("nothing to set: give id.key=value pairs and/or --from patch.json")
    set_values(a.config, a.assign, a.from_)


def cmd_split(a):
    from tslib.tools import split_item
    split_item(a.config, a.id)


def cmd_add(a):
    """items from the translator's lines by actor/style (songs, credits...): one `type: text` item per line"""
    import json
    from tslib.ctx import Ctx
    from tslib import analyze
    from tslib.assdoc import ev_fields
    ctx = Ctx(a.config)
    c = ctx.cfg
    have = {l for i in c["items"] for l in i.get("source_lines", [])}
    evs = [l for l in (c.get("source_events") or []) if l not in have and analyze.is_song(ev_fields(l)[1], a.match)]
    if not evs:
        raise SystemExit(f"no translator line matches {a.match!r} (actor or style) that is not in an item already")
    used = {i["id"] for i in c["items"]}
    new = analyze.song_items(ctx.video, evs, used, bool(a.preset) and a.preset in ctx.preset, set_name=a.preset or "song",
                             prefix=a.prefix, label=a.label)
    if a.preset and a.preset not in ctx.preset:
        print(f"  preset has no set {a.preset!r}: the lines are kept as the translator's ({{\"source\": true}})")
    c["items"] += new
    c["items"].sort(key=lambda i: i["frames"][0])
    with open(a.config, "w", encoding="utf-8") as fh:
        json.dump(c, fh, ensure_ascii=False, indent=1)
    print(f"  + {len(new)} item(s): {new[0]['id']} .. {new[-1]['id']}")


def cmd_timing(a):
    """when the original text is really on screen, per item; --apply writes `frames` (and the `_timing` hint)"""
    import json
    from tslib.ctx import Ctx
    from tslib import timing
    ctx = Ctx(a.config)
    v = ctx.video
    pad = int(round(float(v.fps) * a.pad))
    found = {}
    for it in ctx.items(a.ids):
        tm = timing.detect(v, it, pad)
        if tm is None:
            why = "locked" if isinstance(it.get("timing"), dict) and it["timing"].get("lock") else                   "moving / no fixed region - give \"timing\": {\"box\": [x0,y0,x1,y1]} to judge it"
            print(f"  {it['id']:30s} {it['frames'][0]}-{it['frames'][1]}  not judged ({why})")
            continue
        h, fr = tm["hint"], tm["frames"]
        fi, fo = timing.fade_ms(v, h)
        src = it.get("source_frames")
        line = (f"  {it['id']:30s} {it['frames'][0]}-{it['frames'][1]}" + (f" (translator {src[0]}-{src[1]})" if src else "")
                + (f"  ->  on screen {h['on_screen'][0]}-{h['on_screen'][1]}" if "on_screen" in h else "")
                + (f", fades in {h['fade_in'][0]}-{h['fade_in'][1]} ({fi} ms)" if h.get("fade_in") else "")
                + (f", fades out {h['fade_out'][0]}-{h['fade_out'][1]} ({fo} ms)" if h.get("fade_out") else "")
                + (f"  [{tm['note']}]" if tm["note"] else ""))
        if fr != it["frames"]:
            line += f"  =>  frames {fr[0]}-{fr[1]}" + ("" if a.apply else " (use --apply)")
        print(line)
        found[it["id"]] = (fr, h)
    if a.apply and found:
        with open(a.config, encoding="utf-8") as fh:
            c = json.load(fh)
        n = 0
        for it in c["items"]:
            if it["id"] in found:
                fr, h = found[it["id"]]
                n += it["frames"] != fr
                it["frames"] = fr
                it["_timing"] = h
        with open(a.config, "w", encoding="utf-8") as fh:
            json.dump(c, fh, ensure_ascii=False, indent=1)
        print(f"  frames changed: {n} item(s); rebuild them")


def cmd_lama_ab(a):
    """the same item with and without clean.fill "lama": original | without | with LaMa on the same frames (zoomed to
    the sign), and the seam score of each patch - to keep LaMa only where it is better"""
    import copy
    import numpy as np
    from PIL import Image
    from tslib.ctx import Ctx
    from tslib.builders import build_item
    from tslib.assdoc import header, write_lines, ev_style
    from tslib.text import parse_style
    from tslib.inspect import check_frames, fit_area, seam_score, label, tile
    ctx = Ctx(a.config)
    v = ctx.video
    for it in ctx.items(a.ids):
        if it.get("type") != "plate":
            print(f"[{it['id']}] not a plate - skipped"); continue
        f0, f1 = it["frames"]
        moving = isinstance(it.get("motion"), dict) and it["motion"].get("mode", "static") != "static"
        frames = check_frames(f0, f1, moving)
        res = {}
        for name, fill in (("без LaMa", None), ("LaMa", "lama")):
            v_it = copy.deepcopy(it)
            v_it["clean"] = {**v_it.get("clean", {}), "fill": fill} if fill else {k: x for k, x in v_it.get("clean", {}).items() if k != "fill"}
            print(f"[{it['id']}] building {name}...", flush=True)
            data = build_item(ctx, v_it)
            lines = data["lines"]
            used = {ev_style(l) for l in lines}
            styles = [l for l in ctx.all_style_lines if parse_style(l)["Name"] in used]
            p = ctx.path("check", f"_ab_{it['id']}_{'lama' if fill else 'base'}.ass")
            write_lines(p, header(it["id"], ctx.playres, v.ycbcr, styles) + lines)
            pm = p.replace(".ass", "_mask.ass")
            write_lines(pm, header(it["id"], ctx.playres, v.ycbcr, styles) + [l for l in lines if ev_style(l) == "Маска"])
            res[name] = (p, seam_score(v, pm, frames), data.get("area"))
        crop = fit_area(a.zoom and [int(c) for c in a.zoom.split(",")] or res["LaMa"][2] or [0, 0, 1920, 1080])
        rows = []
        for f in frames:
            row = [Image.fromarray(v.grab(f, 1, crop=crop)[0])]
            for name in ("без LaMa", "LaMa"):
                p = v.render(res[name][0], [f], ctx.path("check", "_tmp"), prefix=f"ab_{name == 'LaMa'}", crop=crop)[0]
                row.append(Image.open(p).convert("RGB"))
            w1 = 640
            row = [x.resize((w1, int(x.height * w1 / x.width)), Image.LANCZOS) for x in row]
            c = Image.new("RGB", (w1 * 3 + 8, row[0].height))
            for k, x in enumerate(row):
                c.paste(x, (k * (w1 + 4), 0))
            rows.append(label(c, f"{it['id']} {f}: original | without LaMa | LaMa"))
        out = ctx.path("check", f"{it['id']}_lama_ab.jpg")
        tile(rows, 1, out)
        sb, sl = np.nanmean(res["без LaMa"][1]), np.nanmean(res["LaMa"][1])
        print(f"[{it['id']}] seam (lower = less visible): without LaMa {sb:.2f}, with LaMa {sl:.2f} -> "
              + ("LaMa better" if sl < sb - 0.2 else "LaMa worse" if sl > sb + 0.2 else "about the same")
              + f"; look at {out} - keep LaMa only if it also looks better")


def cmd_sheet(a):
    from tslib.ctx import Ctx
    from tslib.tools import contact_sheet
    ctx = Ctx(a.config)
    print(contact_sheet(ctx, item_ass, a.out or ctx.path("check", "sheet.jpg"), a.ids or None))


def cmd_compact(a):
    """shrink an already built signs/subs .ass in place (the original is kept once as исходники/<name> (до сжатия))"""
    import shutil
    from tslib import coords
    from tslib.ctx import read_playres
    from tslib.compact import compact_lines, short_names
    from tslib.assdoc import backup_path
    for path in a.files:
        coords.set_playres(read_playres(path)[0])
        with open(path, encoding="utf-8-sig") as fh:
            src = fh.read()
        nl = "\r\n" if "\r\n" in src else "\n"
        lines = src.split(nl)
        out = short_names(compact_lines(lines))
        before = os.path.getsize(path)
        b = backup_path(path, " (до сжатия)")
        if not os.path.exists(b):
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.copy2(path, b)
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(nl.join(out))
        print(f"{path}: {before / 1024:.0f} KB -> {os.path.getsize(path) / 1024:.0f} KB, "
              f"{len(lines) - len(out)} empty lines dropped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor").set_defaults(fn=cmd_doctor)
    p = sp.add_parser("analyze"); p.add_argument("--video", required=True); p.add_argument("--signs", required=True)
    p.add_argument("--work", required=True); p.add_argument("--force", action="store_true")
    p.add_argument("--songs", default=None, help="regex on actor/style of song lines (not analysed, one text item each); "
                   "default: Песня/song/lyric/karaoke/OP/ED/opening/ending/insert; \"\" = analyse everything")
    p.set_defaults(fn=cmd_analyze)
    for name, fn in (("build", cmd_build), ("check", cmd_check), ("preview", cmd_preview)):
        p = sp.add_parser(name); p.add_argument("-c", "--config", required=True); p.add_argument("ids", nargs="*")
        if name == "build":
            p.add_argument("--no-check", action="store_true")
            p.add_argument("-j", "--jobs", type=int, default=4, help="parallel items (1 = sequential)")
        if name == "check":
            p.add_argument("--frames"); p.add_argument("--full", action="store_true")
            p.add_argument("--zoom", help="x0,y0,x1,y1 crop to enlarge"); p.add_argument("--compare", action="store_true")
            p.add_argument("--with-others", action="store_true", help="render the other signs on screen at the same frames too")
        if name == "preview":
            p.add_argument("--ass"); p.add_argument("--out")
        p.set_defaults(fn=fn)
    p = sp.add_parser("assemble"); p.add_argument("-c", "--config", required=True)
    p.add_argument("--out"); p.set_defaults(fn=cmd_assemble)
    p = sp.add_parser("merge"); p.add_argument("-c", "--config", required=True); p.add_argument("--subs"); p.add_argument("--signs")
    p.add_argument("--drop-styles"); p.set_defaults(fn=cmd_merge)
    for name, fn in (("grid", cmd_grid), ("ruler", cmd_ruler)):
        p = sp.add_parser(name); p.add_argument("--video"); p.add_argument("-c", "--config")
        p.add_argument("--frames", required=True); p.add_argument("--out", required=True)
        if name == "grid":
            p.add_argument("--crop"); p.add_argument("--cols", type=int, default=4); p.add_argument("--width", type=int)
        else:
            p.add_argument("--box", required=True); p.add_argument("--scale", type=float, default=1.5); p.add_argument("--std", action="store_true")
        p.set_defaults(fn=fn)
    p = sp.add_parser("geom"); p.add_argument("-c", "--config", required=True); p.add_argument("--frames", required=True)
    p.add_argument("--box", required=True); p.add_argument("--no-title", action="store_true"); p.set_defaults(fn=cmd_geom)
    p = sp.add_parser("fonts"); p.add_argument("--cyr", action="store_true"); p.add_argument("--grep")
    p.add_argument("--try", dest="try_", help="comma-separated families to render over --frame")
    p.add_argument("--video"); p.add_argument("-c", "--config"); p.add_argument("--frame", type=int); p.add_argument("--text")
    p.add_argument("--pos", help="x,baseline (analysis px)"); p.add_argument("--em", type=float)
    p.add_argument("--colour"); p.add_argument("--outline", default="&H00FFFFFF&")
    p.add_argument("--bord", type=float, default=0.0); p.add_argument("--tags"); p.add_argument("--crop"); p.add_argument("--out")
    p.add_argument("--match", help="item id: rank fonts by the strokes of its original text")
    p.add_argument("--box", help="x0,y0,x1,y1 around one line of the original text (default: the item's first zone)")
    p.add_argument("--polarity", choices=("light", "dark")); p.add_argument("--category", help="serif,sans-serif,display,handwriting")
    p.add_argument("--top", type=int, help="ranked faces on the sheet (default 8; none next to --with picks)"); p.add_argument("--offline", action="store_true", help="installed fonts only")
    p.add_argument("--italic", action="store_true"); p.add_argument("--min-px", type=float, default=1.8,
                                                                    help="thinnest acceptable stroke at the real size (1080p px)")
    p.add_argument("--with", dest="with_", help="'Family[:weight],...' picked by eye (installed or Google): shown first "
                   "on the sheet after the item's current face, scored and flagged but never filtered out")
    p.add_argument("--specimen", action="store_true", help="every installed Cyrillic family set in --text, pages of "
                   "48 tiles (--grep narrows): look through them and pick by eye")
    p.add_argument("--light", action="store_true", help="--specimen: dark text on light tiles")
    p.add_argument("--json", help="write the ranked candidates here")
    p.add_argument("--get", help="'Family[:weight]' - download a Google font into the library")
    p.add_argument("--import", dest="import_", nargs="+", help="font files / .zip / folders (a download from "
                   "fonts-online.ru) copied into the font library; prints the names for --with and the style")
    p.set_defaults(fn=cmd_fonts)
    p = sp.add_parser("place"); p.add_argument("-c", "--config", required=True); p.add_argument("item")
    p.add_argument("--box", help="x0,y0,x1,y1 around one line of the original text: its quad is measured")
    p.add_argument("--quad", help="x,y,x,y,x,y,x,y (TL TR BR BL) instead of measuring")
    p.add_argument("--frame", type=int); p.add_argument("--text"); p.add_argument("--preset", default="sign")
    p.add_argument("--polarity", choices=("light", "dark")); p.add_argument("--perspective", action="store_true")
    p.add_argument("--cap", type=float); p.add_argument("--base-frac", dest="base_frac", type=float)
    p.add_argument("--maxw-frac", dest="maxw_frac", type=float); p.add_argument("--out"); p.set_defaults(fn=cmd_place)
    p = sp.add_parser("track"); p.add_argument("--video"); p.add_argument("-c", "--config"); p.add_argument("--frames", required=True)
    p.add_argument("--roi", action="append", required=True, help="x,y,w,h (repeat to compare ROIs)")
    p.add_argument("--scale", type=float, action="append"); p.set_defaults(fn=cmd_track)
    p = sp.add_parser("probe"); p.add_argument("--video"); p.add_argument("-c", "--config"); p.add_argument("--frames", required=True)
    p.add_argument("--box", required=True); p.add_argument("--bg"); p.set_defaults(fn=cmd_probe)
    p = sp.add_parser("set"); p.add_argument("-c", "--config", required=True); p.add_argument("assign", nargs="*")
    p.add_argument("--from", dest="from_", help="UTF-8 JSON patch {\"@\": {...}, \"items\": {id: {...}}}"); p.set_defaults(fn=cmd_set)
    p = sp.add_parser("split"); p.add_argument("-c", "--config", required=True); p.add_argument("id"); p.set_defaults(fn=cmd_split)
    p = sp.add_parser("timing"); p.add_argument("-c", "--config", required=True); p.add_argument("ids", nargs="*")
    p.add_argument("--apply", action="store_true", help="write the frames where the original is on screen")
    p.add_argument("--pad", type=float, default=2.0, help="seconds searched before / after the current frames")
    p.set_defaults(fn=cmd_timing)
    p = sp.add_parser("lama-ab"); p.add_argument("-c", "--config", required=True); p.add_argument("ids", nargs="+")
    p.add_argument("--zoom", help="x0,y0,x1,y1 crop (default: the sign's area)"); p.set_defaults(fn=cmd_lama_ab)
    p = sp.add_parser("add"); p.add_argument("-c", "--config", required=True)
    p.add_argument("--match", required=True, help="regex on the actor or style of the translator's lines")
    p.add_argument("--preset", default=None, help="layer set of the preset to restyle with (e.g. song)")
    p.add_argument("--prefix", default="song"); p.add_argument("--label", default="Песня: "); p.set_defaults(fn=cmd_add)
    p = sp.add_parser("sheet"); p.add_argument("-c", "--config", required=True); p.add_argument("ids", nargs="*")
    p.add_argument("--out"); p.set_defaults(fn=cmd_sheet)
    p = sp.add_parser("compact"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_compact)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
