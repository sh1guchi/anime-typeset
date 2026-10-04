"""ASS styles and text lines placed from analysis-space measurements."""
import re
from . import coords, fonts
from .coords import f2

FIELDS = ["Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour", "BackColour", "Bold",
          "Italic", "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle", "Outline",
          "Shadow", "Alignment", "MarginL", "MarginR", "MarginV", "Encoding"]
SCALED = ["Fontsize", "Spacing", "Outline", "Shadow", "MarginL", "MarginR", "MarginV"]


def parse_style(line):
    body = line.split(":", 1)[1].strip()
    vals = [v.strip() for v in body.split(",")]
    vals = vals[:len(FIELDS) - 1] + [",".join(vals[len(FIELDS) - 1:])] if len(vals) > len(FIELDS) else vals
    return dict(zip(FIELDS, vals))


def style_line(d):
    return "Style: " + ",".join(str(d[f]) for f in FIELDS)


def scale_style(line, k):
    """rescale size-like fields (preset authored for one PlayResY, script uses another)"""
    if abs(k - 1) < 1e-9:
        return line
    d = parse_style(line)
    for f in SCALED:
        d[f] = f2(float(d[f]) * k)
    return style_line(d)


class Styles:
    def __init__(self, lines):
        self.lines = {parse_style(l)["Name"]: l for l in lines}
        self._font = {}

    def get(self, name):
        return parse_style(self.lines[name])

    def font(self, name):
        """(font record, metrics) of a style's face"""
        if name not in self._font:
            d = self.get(name)
            rec = fonts.lookup(d["Fontname"], d["Bold"] not in ("0", ""), d["Italic"] not in ("0", ""))
            if rec is None:
                raise SystemExit(f"font '{d['Fontname']}' of style '{name}' is not installed / not in the font dirs")
            self._font[name] = (rec, fonts.metrics(rec))
        return self._font[name]


def fade_tag(video, spec):
    if "fade_frames" in spec:
        a, b = spec["fade_frames"]
        return f"\\fad({round(video.tms(b) - video.tms(a))},0)"
    if "fade" in spec:
        i, o = spec["fade"]
        return f"\\fad({int(i)},{int(o)})"
    return ""


def emit_source(ctx, spec, f0, f1, actor="Надпись"):
    """The translator's own (already typeset, e.g. Crunchyroll) lines, kept as they are.
    spec: {"source": true, "lines": [raw Dialogue lines] (filled from the item's source_lines),
           "retime": true (default: put them on the item's frames), "layer": 20 (added to their layers),
           "tags": extra tags prepended to each line,
           "only": [indices of the lines to keep], "drop_drawings": true (CR's own \\p1 plates/boxes: our mask
           replaces them, their text stays)}"""
    out = []
    lines = spec.get("lines") or []
    if spec.get("only") is not None:
        lines = [lines[i] for i in spec["only"]]
    if spec.get("drop_drawings"):
        lines = [l for l in lines if not re.search(r"\\p[1-9]", l.split(",", 9)[-1])]
    for raw in lines:
        _, rest = raw.split(":", 1)
        p = rest.strip().split(",", 9)
        p[0] = str(int(spec.get("layer", 20)) + int(p[0]))
        if spec.get("retime", True):
            p[1], p[2] = ctx.video.atime(f0), ctx.video.atime(f1 + 1)
        p[3] = ctx.src_style_map.get(p[3], p[3])
        if getattr(ctx, "src_wrap", 0) != 2 and "\\q" not in p[9]:
            p[9] = "{\\q%d}" % ctx.src_wrap + p[9]
        p[4] = actor
        if spec.get("tags"):
            p[9] = "{" + spec["tags"] + "}" + p[9]
        out.append("Dialogue: " + ",".join(p))
    return out, (None, 100)


def emit_text(ctx, spec, f0, f1, layers=None, actor="Надпись"):
    """Text lines for one sign.
    spec: text (\\N allowed), layers [{style, layer, tags, offset:[dx,dy]}] (default: preset[spec.preset or 'sign']),
      em (em size, analysis px) or fs (raw \\fs), x + base (centre x / baseline y, analysis px -> \\an2) or pos [x,y],
      maxw + fit ('shrink'|'squeeze'), spacing (analysis px), fade [in,out] ms or fade_frames [a,b], tags, an.
      {"source": true} instead: keep the translator's typeset lines (emit_source)."""
    if spec.get("source"):
        return emit_source(ctx, spec, f0, f1, actor)
    K = coords.K
    layers = layers or spec.get("layers") or ctx.preset[spec.get("preset", "sign")]["layers"]
    main = spec.get("style") or layers[-1]["style"]
    text = spec["text"]
    st, en = ctx.video.atime(f0), ctx.video.atime(f1 + 1)
    tags = fade_tag(ctx.video, spec)
    fs = None; em = spec.get("em"); sx = 100
    if em is not None or "base" in spec or "maxw" in spec:
        rec, met = ctx.styles.font(main)
        miss = fonts.missing_glyphs(rec, text.replace("\\N", ""))
        if miss:
            print(f"    warning: style '{main}' font lacks glyphs {''.join(miss)}", flush=True)
        if em is None:
            em = float(spec.get("fs", ctx.styles.get(main)["Fontsize"])) / K / met["ratio"]
        if "maxw" in spec:
            w = max(fonts.text_width(rec, part, em) for part in text.split("\\N"))
            if w > spec["maxw"]:
                if spec.get("fit", "squeeze") == "shrink":
                    em *= spec["maxw"] / w
                else:
                    sx = 100 * spec["maxw"] / w
        fs = em * met["ratio"] * K
    elif "fs" in spec:
        fs = float(spec["fs"])
    size = (f"\\fs{f2(fs)}" if fs is not None else "") + (f"\\fscx{f2(sx)}" if sx != 100 else "")
    if "spacing" in spec:
        size += f"\\fsp{f2(spec['spacing'] * K)}"
    if "base" in spec:
        rec, met = ctx.styles.font(main)
        x, y = spec["x"] * K, spec["base"] * K + fs * met["desc"]
        # x is the centre, or with align left/right the line's left/right end (lists, notes)
        an = {"left": "\\an1", "right": "\\an3"}.get(spec.get("align"), "\\an2")
    else:
        x, y = spec["pos"][0] * K, spec["pos"][1] * K
        an = f"\\an{spec['an']}" if "an" in spec else ""
    out = []
    mb = spec.get("move_by")       # [dx, dy] analysis px over the item (a slow pan the text should follow)
    for L in layers:
        ox, oy = L.get("offset", [0, 0])
        px_, py_ = x + ox * K, y + oy * K
        pos = (f"\\move({f2(px_)},{f2(py_)},{f2(px_ + mb[0] * K)},{f2(py_ + mb[1] * K)})" if mb
               else f"\\pos({f2(px_)},{f2(py_)})")
        out.append(f"Dialogue: {L['layer']},{st},{en},{L['style']},{actor},0,0,0,,"
                   f"{{{tags}{an}{L.get('tags', '')}{pos}{size}{spec.get('tags', '')}}}{text}")
    return out, (em, sx)
