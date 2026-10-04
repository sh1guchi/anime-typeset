"""Font index (family/bold -> file), metrics as libass uses them, text measurement."""
import os, json, hashlib, logging
from fontTools.ttLib import TTFont, TTCollection

logging.getLogger("fontTools").setLevel(logging.ERROR)
from PIL import ImageFont

SYSTEM_DIRS = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
               os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts"),
               os.path.expanduser("~/.fonts"), "/usr/share/fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "anime-typeset", "fonts.json")
_index = None
_extra = []


def add_dirs(dirs):
    """project font dirs (e.g. the release's fonts folder) take part in lookups"""
    global _index
    for d in dirs:
        if d and os.path.isdir(d) and d not in _extra:
            _extra.append(d); _index = None


def _files(dirs):
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, names in os.walk(d):
            for f in names:
                if f.lower().endswith((".ttf", ".otf", ".ttc")):
                    out.append(os.path.join(root, f))
    return sorted(out)


def _scan_file(p):
    recs = []
    try:
        fonts = TTCollection(p, lazy=True).fonts if p.lower().endswith(".ttc") else [TTFont(p, lazy=True)]
    except Exception:
        return recs
    for i, ft in enumerate(fonts):
        try:
            n = ft["name"]; head = ft["head"]
            os2 = ft["OS/2"] if "OS/2" in ft else None
            hhea = ft["hhea"]
            cmap = ft.getBestCmap() or {}
            winA, winD = (os2.usWinAscent, os2.usWinDescent) if os2 and (os2.usWinAscent + os2.usWinDescent) else (hhea.ascent, -hhea.descent)
            recs.append(dict(path=p, index=i, family=n.getDebugName(1) or "", sub=n.getDebugName(2) or "",
                             tfamily=n.getDebugName(16) or "", full=n.getDebugName(4) or "",
                             bold=bool(head.macStyle & 1) or bool(os2 and os2.usWeightClass >= 600),
                             italic=bool(head.macStyle & 2), weight=os2.usWeightClass if os2 else 400,
                             cyr=(0x0416 in cmap and 0x0451 in cmap), upm=head.unitsPerEm, winA=winA, winD=winD,
                             cap=int(getattr(os2, "sCapHeight", 0) or 0) if os2 else 0))
        except Exception:
            continue
    return recs


def index():
    global _index
    if _index is not None:
        return _index
    files = _files(SYSTEM_DIRS + _extra)
    key = hashlib.md5("|".join(f"{p}:{os.path.getmtime(p):.0f}" for p in files).encode("utf-8", "ignore")).hexdigest()
    try:
        c = json.load(open(CACHE, encoding="utf-8"))
        if c.get("key") == key:
            _index = c["fonts"]; return _index
    except Exception:
        pass
    recs = []
    for p in files:
        recs += _scan_file(p)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = f"{CACHE}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"key": key, "fonts": recs}, fh, ensure_ascii=False)
    os.replace(tmp, CACHE)
    _index = recs
    return recs


def lookup(family, bold=False, italic=False):
    """closest face of `family` (name ID 1 or 16), preferring the requested bold/italic"""
    fam = family.strip().lower()
    cands = [r for r in index() if r["family"].lower() == fam or r["tfamily"].lower() == fam or r["full"].lower() == fam]
    if not cands:
        return None
    def score(r):
        return (r["bold"] == bool(bold)) * 2 + (r["italic"] == bool(italic)) + (r["family"].lower() == fam) * 0.5
    return max(cands, key=score)


def metrics(rec):
    """libass sizing: font size = usWinAscent + usWinDescent (VSFilter compatible)"""
    tot = rec["winA"] + rec["winD"]
    cap = rec["cap"] or _measure_cap(rec)
    return dict(ratio=tot / rec["upm"], desc=rec["winD"] / tot, cap=cap / rec["upm"])


def _measure_cap(rec):
    f = ImageFont.truetype(rec["path"], rec["upm"], index=rec["index"])
    b = f.getbbox("H", anchor="ls")
    return -b[1]


_pil = {}


def text_width(rec, text, em):
    """advance width (px) of `text` at em size `em` px (no letter spacing)"""
    key = (rec["path"], rec["index"])
    if key not in _pil:
        _pil[key] = ImageFont.truetype(rec["path"], 400, index=rec["index"])
    return _pil[key].getlength(text) * em / 400


def missing_glyphs(rec, text):
    try:
        ft = TTCollection(rec["path"], lazy=True).fonts[rec["index"]] if rec["path"].lower().endswith(".ttc") else TTFont(rec["path"], lazy=True)
        cmap = ft.getBestCmap() or {}
        return sorted({c for c in text if not c.isspace() and ord(c) not in cmap})
    except Exception:
        return []
