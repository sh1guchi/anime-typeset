"""Shorter ASS for the same picture. Applied to every built item before it is saved (ctx.save_lines).

Drawings: the patches are rectangles in block units that overlap their neighbours by 1 analysis px, so
their coordinates are fractions like 15.33. Multiplying every coordinate by m (3 for 3-px blocks) and
dividing \\fscx/\\fscy (also inside \\t) by m gives integer coordinates and the same geometry, ~20% shorter.
A multiplier is used only when every coordinate lands within `tol_px` (default 0.05 analysis px) of the
original point; otherwise the line stays as it is.
"""
import re
from . import coords

CMDS = set("mnlbspc")
SCALE = re.compile(r"\\fsc([xy])(-?\d+(?:\.\d+)?)")
MULTS = (2, 3, 4, 6, 8, 12, 16, 24, 1 / 2, 1 / 3, 1 / 4)   # < 1: integer coords that are all multiples


def _text_start(line):
    """index of the Text field (after the 9th comma) of a Dialogue/Comment line"""
    i = -1
    for _ in range(9):
        i = line.find(",", i + 1)
        if i < 0:
            return -1
    return i + 1


def _fmt(v):
    if abs(v - round(v)) < 5e-4:      # 33.3333 * 3 is 100, not 99.9999
        v = round(v)
    return f"{v:.4f}".rstrip("0").rstrip(".")


EFFECTS = re.compile(r"\\(bord|xbord|ybord|shad|xshad|yshad|blur|be)\d")


def _is_patch(line, i, tags):
    """only plain patch drawings (actor 'mask', mask style without outline/shadow): libass scales outline,
    shadow and blur with \\fscx/\\fscy, so rescaling a styled drawing (card lines, ornaments) changes it"""
    f = line[:i].split(",")
    actor = f[4].strip().lower()
    return (actor in ("mask", "маска") or f[3].strip() == MASK_OUT) and not EFFECTS.search(tags)


def compact_drawing(line, tol_px=0.05):
    i = _text_start(line)
    if i < 0:
        return line
    text = line[i:]
    if not text.startswith("{"):
        return line
    j = text.find("}") + 1
    tags, body = text[:j], text[j:]
    if "\\p1" not in tags or "{" in body or not _is_patch(line, i, tags):
        return line
    toks = body.split()
    try:
        nums = [float(t) for t in toks if t not in CMDS]
    except ValueError:
        return line
    if not nums:
        return line
    scales = [float(v) for _, v in SCALE.findall(tags)]
    if len(scales) < 2:      # no explicit \fscx\fscy: the style's scale applies, leave the line alone
        return line
    unit_px = max(abs(s) for s in scales) / 100.0 / coords.K       # one drawing unit in analysis px
    tol = tol_px / max(unit_px, 1e-9)
    best = line
    for m in MULTS:
        if any(abs(v * m - round(v * m)) > tol * m for v in nums):
            continue
        nb = " ".join(t if t in CMDS else str(int(round(float(t) * m))) for t in toks)
        nt = SCALE.sub(lambda mo: f"\\fsc{mo.group(1)}{_fmt(float(mo.group(2)) / m)}", tags)
        cand = line[:i] + nt + nb
        if len(cand) < len(best):
            best = cand
    return best


CLIP = re.compile(r"\\(i?)clip\((m [^)]*)\)")
POS = re.compile(r"\\(pos|move)\(([^)]*)\)")
TSCALE = re.compile(r"\\t\([^)]*?\\fscx(-?[\d.]+)\\fscy(-?[\d.]+)")


def _subpaths(d):
    """'m x y l x y ... m ...' -> [[(x, y), ...], ...] (only m/l polygons; None if anything else)"""
    out, cur = [], None
    toks = d.split()
    k = 0
    while k < len(toks):
        t = toks[k]
        if t == "m":
            cur = []; out.append(cur); k += 1
        elif t == "l":
            k += 1
        elif t in CMDS:
            return None
        else:
            cur.append((float(t), float(toks[k + 1]))); k += 2
    return out


def _clip_poly(pts, x0, y0, x1, y1):
    """Sutherland-Hodgman: polygon (any shape) against an axis-aligned rectangle"""
    def cut(p, inside, inter):
        res = []
        for i in range(len(p)):
            a, b = p[i - 1], p[i]
            ia, ib = inside(a), inside(b)
            if ib:
                if not ia:
                    res.append(inter(a, b))
                res.append(b)
            elif ia:
                res.append(inter(a, b))
        return res

    def ix(xc):
        return lambda a, b: (xc, a[1] + (b[1] - a[1]) * (xc - a[0]) / (b[0] - a[0]))

    def iy(yc):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yc - a[1]) / (b[1] - a[1]), yc)
    for inside, inter in ((lambda q: q[0] >= x0, ix(x0)), (lambda q: q[0] <= x1, ix(x1)),
                          (lambda q: q[1] >= y0, iy(y0)), (lambda q: q[1] <= y1, iy(y1))):
        if not pts:
            break
        pts = cut(pts, inside, inter)
    return pts


def _area(p):
    return 0.5 * sum(p[i - 1][0] * p[i][1] - p[i][0] * p[i - 1][1] for i in range(len(p)))


def _num(v):
    """clip vertices: the original ones have 0.1 precision; points cut on the box edge get 0.01, otherwise the
    edge that runs to them turns a little and shifts the antialiased mask border"""
    v = round(v, 2)
    return str(int(v)) if v == int(v) else f"{v:.2f}".rstrip("0")


def localize_clip(line, margin=1.0):
    """A drawing line only paints inside the box its shape sweeps (from \\pos/\\move and \\fscx/\\fscy, also
    their \\t end values), so its vector clip can be cut down to that box: the same pixels with a much shorter
    clip. A box wholly inside the clip drops the clip; a box outside it drops the line ("")."""
    i = _text_start(line)
    if i < 0 or "\\p1" not in line[i:]:
        return line
    text = line[i:]
    j = text.find("}") + 1
    tags, body = text[:j], text[j:]
    mc, mp = CLIP.search(tags), POS.search(tags)
    sc = [float(v) for _, v in SCALE.findall(tags.split("\\t(")[0])]
    if not mc or mc.group(1) or not mp or len(sc) < 2 or not _is_patch(line, i, tags):
        return line
    try:
        nums = [float(t) for t in body.split() if t not in CMDS]
        polys = _subpaths(mc.group(2))
    except (ValueError, IndexError):
        return line
    if not nums or not polys:
        return line
    us, vs = nums[0::2], nums[1::2]
    pv = [float(v) for v in mp.group(2).split(",")]
    starts = [(pv[0], pv[1])] + ([(pv[2], pv[3])] if mp.group(1) == "move" else [])
    scales = [(sc[0], sc[1])]
    mt = TSCALE.search(tags)
    if mt:
        scales.append((float(mt.group(1)), float(mt.group(2))))
    xs, ys = [], []
    for (px, py) in starts:          # position and scale both move linearly in time: the ends bound it
        for (sx, sy) in scales:
            xs += [px + min(us) * sx / 100, px + max(us) * sx / 100]
            ys += [py + min(vs) * sy / 100, py + max(vs) * sy / 100]
    x0, y0, x1, y1 = min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin
    cut = [p for p in (_clip_poly(q, x0, y0, x1, y1) for q in polys) if len(p) >= 3]
    area = sum(_area(p) for p in cut)
    if abs(area) < 1e-3:
        return ""                                            # never inside the clip: paints nothing
    if abs(abs(area) - (x1 - x0) * (y1 - y0)) < 0.5:
        new = ""                                             # wholly inside the clip
    else:
        new = "\\clip(" + " ".join("m " + f"{_num(p[0][0])} {_num(p[0][1])} l " +
                                   " ".join(f"{_num(x)} {_num(y)}" for x, y in p[1:]) for p in cut) + ")"
        if len(new) >= len(mc.group(0)):
            return line
    return line[:i] + tags[:mc.start()] + new + tags[mc.end():] + body


MASK_STYLE = "Маска"     # every preset defines it with \an7, 100 % scale, no outline/shadow
MASK_OUT = "M"           # its name in the finished files: thousands of patch lines carry the style name (and an
                         # actor) - "Маска" + actor "маска" cost 19 bytes a line, "M" + no actor 1 (BC 2-01: -65 KB,
                         # the same render; a shorter clip or \fsc precision is not lossless - checked, edges move)
MASK_STYLES = (MASK_STYLE, MASK_OUT)


def short_names(lines):
    """finished file: the patch style 'Маска' -> 'M' (style line and events), no actor on patch lines.
    Only for output files - built items (lines/*.json) keep 'Маска'."""
    out = []
    for l in lines:
        if l.startswith("Style: " + MASK_STYLE + ","):
            l = "Style: " + MASK_OUT + l[len("Style: " + MASK_STYLE):]
        elif l.startswith(("Dialogue:", "Comment:")):
            i = _text_start(l)
            if i > 0:
                p = l[:i].split(",")
                if p[3].strip() == MASK_STYLE:
                    p[3] = MASK_OUT
                    if p[4].strip().lower() in ("mask", "маска"):
                        p[4] = ""
                    l = ",".join(p) + l[i:]
        out.append(l)
    return out


def prune_defaults(line):
    """drop tags that only repeat the mask style: \\an7 and \\fscx100\\fscy100 (scale not animated)"""
    i = _text_start(line)
    if i < 0 or line[:i].split(",")[3].strip() not in MASK_STYLES or not line[i:].startswith("{"):
        return line
    j = line.find("}", i) + 1
    tags = line[i:j]
    if "\\p1" not in tags:
        return line
    new = re.sub(r"\\an7(?=[\\}])", "", tags, count=1)
    animated = "\\t(" in new and "\\fsc" in new.split("\\t(", 1)[1]
    if not animated and "\\clip" not in new:       # (a clipped line keeps its scale for localize_clip)
        new = re.sub(r"\\fscx100\\fscy100(?=[\\}])", "", new, count=1)
    return line[:i] + new + line[j:]


def compact_lines(lines):
    out = []
    for l in lines:
        if l.startswith(("Dialogue:", "Comment:")):
            l = localize_clip(compact_drawing(l))
            if not l:
                continue
            l = prune_defaults(l)
        out.append(l)
    return out
