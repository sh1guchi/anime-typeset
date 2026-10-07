"""Font library: the installed fonts plus the free (OFL) Cyrillic families of Google Fonts.

Catalog - the Fontsource API (family, category, weights, styles; no key). Files - the Google Fonts CSS API:
with `text=` it returns a few-KB subset holding just those letters (enough to measure a face), without it the
whole static font of one weight. Full fonts are kept in the library folder - flat, because libass loads that
folder (`fontsdir`) on every render, so a downloaded face renders like an installed one; measuring subsets go
to the cache. The library folder: ~/.anime-typeset.json "font_library" (else ~/.cache/anime-typeset/fontlib)."""
import os, json, time, hashlib, urllib.request, urllib.parse, re
from . import fonts, video

CFG = os.path.join(os.path.expanduser("~"), ".anime-typeset.json")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "anime-typeset")
PROBE = os.path.join(CACHE, "fontprobe")
CATALOG_URL = "https://api.fontsource.org/v1/fonts?subsets=cyrillic"
CSS_URL = "https://fonts.googleapis.com/css2?family={fam}:{axis}@{val}{text}"
UA = "curl/8.0"          # with an old-style agent the CSS API answers with TrueType (libass / PIL read it)
# every typical Cyrillic letter shape, lower and upper case: the measuring sample
SAMPLE = "Съешь же ещё этих мягких французских булок, да выпей чаю. ЦАРСТВО КЛЕВЕРА"


def _cfg():
    try:
        with open(CFG, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def library_dir(create=True):
    d = _cfg().get("font_library") or os.path.join(CACHE, "fontlib")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _get(url, timeout=40, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:          # network hiccup: retry, then give up on this one file
            last = e
            time.sleep(1 + 2 * k)
    raise RuntimeError(f"{url}: {last}")


def catalog(max_age_days=14):
    """[{family, category, weights, styles}] of the Google families with Cyrillic, cached"""
    os.makedirs(PROBE, exist_ok=True)
    p = os.path.join(PROBE, "catalog.json")
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < max_age_days * 86400:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    data = json.loads(_get(CATALOG_URL))
    cat = [{"family": f["family"], "category": f.get("category", ""), "weights": f.get("weights") or [400],
            "styles": f.get("styles") or ["normal"]} for f in data if f.get("type") == "google"]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(cat, fh, ensure_ascii=False)
    return cat


def _css_url(family, weight, italic, text=None):
    fam = urllib.parse.quote_plus(family)
    axis, val = ("ital,wght", f"1,{weight}") if italic else ("wght", str(weight))
    t = "&text=" + urllib.parse.quote(text) if text else ""
    css = _get(CSS_URL.format(fam=fam, axis=axis, val=val, text=t)).decode("utf-8", "replace")
    m = re.search(r"url\((https://[^)]+)\)", css)
    if not m:
        raise RuntimeError(f"no font file for {family} {weight}{' italic' if italic else ''}")
    return m.group(1)


def _slug(family, weight, italic):
    return f"{re.sub(r'[^A-Za-z0-9]', '', family)}-{weight}{'i' if italic else ''}"


def probe_file(family, weight, italic=False):
    """few-KB subset of the face with the measuring sample (cached)"""
    os.makedirs(PROBE, exist_ok=True)
    key = hashlib.md5(SAMPLE.encode("utf-8")).hexdigest()[:8]
    p = os.path.join(PROBE, f"{_slug(family, weight, italic)}-{key}.ttf")
    if not os.path.exists(p):
        data = _get(_css_url(family, weight, italic, SAMPLE))
        with open(p + ".tmp", "wb") as fh:
            fh.write(data)
        os.replace(p + ".tmp", p)
    return p


def full_file(family, weight, italic=False):
    """the whole static font of one weight, in the library (libass and the index see it at once)"""
    lib = library_dir()
    p = os.path.join(lib, _slug(family, weight, italic) + ".ttf")
    if not os.path.exists(p):
        data = _get(_css_url(family, weight, italic), timeout=120)
        with open(p + ".tmp", "wb") as fh:
            fh.write(data)
        os.replace(p + ".tmp", p)
        fonts.add_dirs([lib], force=True)
    return p


def ass_face(path, index=0):
    """how a script names this file: (Fontname, bold flag, italic flag). Static Google instances are named
    'Family' + Regular/Bold/Italic/Bold Italic, or 'Family SemiBold' + Regular for the other weights."""
    from fontTools.ttLib import TTFont, TTCollection
    ft = TTCollection(path, lazy=True).fonts[index] if path.lower().endswith(".ttc") else TTFont(path, lazy=True)
    n = ft["name"]
    fam = n.getDebugName(1) or n.getDebugName(16) or ""
    sub = (n.getDebugName(2) or "").lower()
    return fam, "bold" in sub, "italic" in sub or "oblique" in sub


def _init():
    lib = _cfg().get("font_library") or os.path.join(CACHE, "fontlib")
    if os.path.isdir(lib):
        video.FONTSDIR = lib
        fonts.add_dirs([lib])


_init()
