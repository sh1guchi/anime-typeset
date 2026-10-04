"""Episode context: config, video, preset styles, caches."""
import os, json, re
import numpy as np
from . import coords, fonts
from .video import Video
from .text import Styles, scale_style, parse_style, style_line

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRESETS = os.path.join(SKILL_DIR, "presets")


def load_preset(name_or_path):
    """preset by name or path; unknown / 'TODO' -> 'default' with a warning (inspection commands must work
    on a fresh analyze draft)"""
    name_or_path = name_or_path or "default"
    p = name_or_path if os.path.isfile(name_or_path) else os.path.join(PRESETS, f"{name_or_path}.json")
    if not os.path.isfile(p):
        print(f"  preset '{name_or_path}' not found - using 'default' (set \"preset\" in episode.json)", flush=True)
        p = os.path.join(PRESETS, "default.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def guess_preset(*names):
    """preset whose "match" words occur in the video / signs file names (e.g. 'black clover')"""
    hay = " ".join(n.lower().replace("_", " ").replace(".", " ") for n in names if n)
    for f in sorted(os.listdir(PRESETS)):
        if not f.endswith(".json") or f == "default.json":
            continue
        try:
            words = json.load(open(os.path.join(PRESETS, f), encoding="utf-8")).get("match", [])
        except Exception:
            continue
        if any(w.lower() in hay for w in words):
            return f[:-5]
    return "default"


def read_playres(ass_path):
    px = py = None
    with open(ass_path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.startswith("PlayResX:"):
                px = int(line.split(":")[1])
            elif line.startswith("PlayResY:"):
                py = int(line.split(":")[1])
            elif line.startswith("[Events]"):
                break
    return px, py


def read_wrapstyle(ass_path):
    """WrapStyle of a script ([Script Info]); 0 (smart wrap) when absent, as libass assumes"""
    with open(ass_path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.startswith("WrapStyle:"):
                try:
                    return int(line.split(":")[1])
                except ValueError:
                    return 0
            if line.startswith("[Events]"):
                break
    return 0


def read_styles(ass_path):
    with open(ass_path, encoding="utf-8-sig") as fh:
        return [l.rstrip("\r\n") for l in fh if l.startswith("Style:")]


def read_events(ass_path):
    with open(ass_path, encoding="utf-8-sig") as fh:
        return [l.rstrip("\r\n") for l in fh if l.startswith("Dialogue:")]


class Ctx:
    def __init__(self, cfg_path):
        self.cfg_path = os.path.abspath(cfg_path)
        self.work = os.path.dirname(self.cfg_path)
        with open(self.cfg_path, encoding="utf-8") as fh:
            self.cfg = json.load(fh)
        c = self.cfg
        self.video = Video(self.abs(c["video"]))
        self.preset = load_preset(c.get("preset", "default"))
        src = self.abs(c["source_signs"]) if c.get("source_signs") else None
        if c.get("playres"):
            px, py = c["playres"]
        elif src and os.path.exists(src):
            px, py = read_playres(src)
        else:
            px, py = None, None
        self.playres = (px or coords.AW, py or coords.AH)
        coords.set_playres(self.playres[0])
        k = self.playres[1] / float(self.preset.get("playres_y", 360))
        self.style_lines = [scale_style(l, k) for l in self.preset["styles"]]
        # styles of the translator's (e.g. Crunchyroll) signs file, for lines kept "as in the source".
        # analyze stores them in the config, so the config keeps working after the source file is overwritten.
        src_styles = c.get("source_styles")
        if src_styles is None and src and os.path.exists(src):
            src_styles = read_styles(src)
        names = {parse_style(l)["Name"] for l in self.style_lines}
        # the translator's WrapStyle: lines kept as in the source must wrap as they did there (our scripts are
        # WrapStyle 2, a long CR disclaimer line would run off the screen)
        self.src_wrap = c.get("source_wrapstyle")
        if self.src_wrap is None:
            self.src_wrap = read_wrapstyle(src) if src and os.path.exists(src) else 0
        self.src_style_map, extra = {}, []
        for l in src_styles or []:
            d = parse_style(l)
            new = d["Name"] if d["Name"] not in names else d["Name"] + " CR"
            self.src_style_map[d["Name"]] = new
            d["Name"] = new
            extra.append(style_line(d))
        self.all_style_lines = self.style_lines + extra
        self.styles = Styles(self.all_style_lines)
        fonts.add_dirs([d if os.path.isabs(d) else os.path.join(self.work, d) for d in c.get("font_dirs", [])])
        for d in ("cache", "lines", "check"):
            os.makedirs(os.path.join(self.work, d), exist_ok=True)

    def path(self, *p):
        return os.path.join(self.work, *p)

    def abs(self, p):
        """config paths may be relative to the config's folder"""
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.work, p))

    def cache_path(self, name):
        return os.path.join(self.work, "cache", name)

    def items(self, ids=None):
        its = self.cfg["items"]
        if ids:
            missing = set(ids) - {i["id"] for i in its}
            if missing:
                raise SystemExit(f"unknown item ids: {', '.join(sorted(missing))}")
            its = [i for i in its if i["id"] in ids]
        return [i for i in its if not i.get("skip")]

    def median(self, a, b, max_frames=60):
        """per-pixel median of frames a..b (analysis-size RGB float32), cached"""
        a, b = int(a), int(b)
        step = max(1, -(-(b - a + 1) // max_frames))
        p = self.cache_path(f"med_{a}_{b}_{step}.npy")
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
        fr = self.video.grab(a, b - a + 1, step=step)
        med = np.median(fr, axis=0).astype(np.float32)
        tmp = f"{p}.{os.getpid()}.tmp.npy"            # atomic: parallel builds may share a cache entry
        np.save(tmp, med.astype(np.float16))
        os.replace(tmp, p)
        return med

    def lines_path(self, item_id):
        return self.path("lines", f"{item_id}.json")

    def load_lines(self, item_id):
        p = self.lines_path(item_id)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def save_lines(self, item_id, data):
        if self.cfg.get("compact", True):
            from .compact import compact_lines
            data = dict(data, lines=compact_lines(data.get("lines", [])))
        with open(self.lines_path(item_id), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
