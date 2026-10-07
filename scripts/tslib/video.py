"""Video access in the analysis space (width 1920, height by aspect), and rendering through libass (ffmpeg `subtitles`)."""
import subprocess, json, os, math, shutil
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from . import coords

# fonts libass loads on every render besides the installed ones: the font library (downloaded fonts) - set by
# tslib.fontlib on import from ~/.anime-typeset.json "font_library" (see references/technique.md, "Шрифты")
FONTSDIR = None


class Video:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate,start_time,color_space,color_range:format=duration",
                            "-of", "json", self.path], capture_output=True, check=True)
        info = json.loads(r.stdout)
        s = info["streams"][0]
        self.width, self.height = int(s["width"]), int(s["height"])
        self.fps = Fraction(s["r_frame_rate"])
        self.start = float(s.get("start_time") or 0.0)
        self.duration = float(info.get("format", {}).get("duration") or 0)
        cs = s.get("color_space") or "unknown"
        self.matrix = "601" if (cs in ("smpte170m", "bt470bg") or (cs == "unknown" and self.height < 720)) else "709"
        self.range = "pc" if s.get("color_range") == "pc" else "tv"
        ff = "bt709" if self.matrix == "709" else "smpte170m"
        # untagged sources: tag explicitly so swscale and the subtitles filter agree on the matrix
        self.tag = f"setparams=colorspace={ff}:range={self.range}:color_primaries={ff}:color_trc={ff}"
        self.in_cm = "bt709" if self.matrix == "709" else "bt601"
        self.ycbcr = f"{'TV' if self.range == 'tv' else 'PC'}.{self.matrix}"   # ASS "YCbCr Matrix"
        self.nframes = int(self.duration * float(self.fps))
        coords.set_frame(self.width, self.height)
        self._hw = None

    def hw(self):
        """GPU decoding (NVDEC/D3D11) when it works: frames come back to system memory, so the pixels are
        bit-identical to CPU decoding (scaling/colour conversion stay on the CPU), but 4K HEVC decoding
        costs ~half the CPU time. TS_HWACCEL=none|cuda|d3d11va|... overrides the auto choice."""
        if self._hw is None:
            want = os.environ.get("TS_HWACCEL", "auto").lower()
            cands = [] if want == "none" else ([want] if want != "auto" else ["cuda", "d3d11va"])
            self._hw = []
            for hw in cands:
                r = subprocess.run(["ffmpeg", "-v", "error", "-hwaccel", hw,
                                    "-ss", "1", "-i", self.path, "-frames:v", "1", "-f", "null", "-"],
                                   capture_output=True, text=True, errors="replace")
                if r.returncode == 0 and "Failed" not in r.stderr and "rror" not in r.stderr:
                    self._hw = ["-hwaccel", hw]
                    break
        return self._hw

    def h264(self):
        """preview encoder: NVENC (fast, visually lossless at these settings) or x264"""
        if not hasattr(self, "_enc"):
            r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=256x144:d=0.2", "-c:v", "h264_nvenc",
                                "-f", "null", "-"], capture_output=True)
            self._enc = (["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "17", "-b:v", "0"]
                         if r.returncode == 0 else ["-c:v", "libx264", "-crf", "17", "-preset", "medium"])
        return self._enc

    # --- time <-> frames (CFR)
    def ft(self, n):
        """start time of frame n, seconds"""
        return float(Fraction(n) / self.fps)

    def frame_at(self, t):
        """first frame displayed at or after time t (seconds)"""
        return int(math.ceil(t * float(self.fps) - 1e-6))

    def atime(self, n):
        """ASS timestamp (centiseconds, truncated) at which frame n is already visible"""
        cs = math.floor(Fraction(n) * 100 / self.fps)
        return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"

    def tms(self, n):
        return float(Fraction(n) * 1000 / self.fps)

    # --- decoding
    def grab(self, n, count=1, crop=None, w=None, h=None, gray=False, step=1):
        """frames n .. n+count-1 (every `step`-th) as uint8 (N,h,w,3) RGB full range, or (N,h,w) gray.
        crop=(x0,y0,x1,y1) in the scaled (w x h) space."""
        w, h = w or coords.AW, h or coords.AH
        x0, y0, x1, y1 = crop or (0, 0, w, h)
        vf = (f"{self.tag},scale={w}:{h}:flags=lanczos:in_color_matrix={self.in_cm}:"
              f"in_range={self.range}:out_range=pc,format=rgb24")
        if crop:
            vf += f",crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
        if gray:
            vf += ",format=gray"
        nout = count
        if step > 1:
            vf = f"select='not(mod(n\\,{step}))'," + vf
            nout = (count + step - 1) // step
        cmd = ["ffmpeg", "-v", "error"] + self.hw() + ["-ss", f"{max(0.0, self.ft(n) - 0.002):.4f}", "-i", self.path,
               "-frames:v", str(nout), "-vf", vf, "-fps_mode", "passthrough",
               "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        shape = (-1, y1 - y0, x1 - x0) if gray else (-1, y1 - y0, x1 - x0, 3)
        return np.frombuffer(raw, np.uint8).reshape(shape)

    def frame(self, n, **kw):
        return self.grab(n, 1, **kw)[0].astype(np.float32)

    # --- rendering with subtitles
    def _sub_vf(self, assname, fontsdir=None, w=None, h=None):
        """fontsdir: absolute path (any drive) - libass loads every font in it besides the installed ones"""
        w, h = w or coords.AW, h or coords.AH
        sub = f"subtitles={assname}"
        fontsdir = fontsdir or FONTSDIR
        if fontsdir and os.path.isdir(fontsdir):
            fd = os.path.abspath(fontsdir).replace("\\", "/").replace(":", "\\:").replace("'", "")
            sub += f":fontsdir='{fd}'"
        return f"{self.tag},scale={w}:{h}:flags=lanczos,{sub}"

    def render(self, ass_path, frames, outdir, prefix="r", crop=None, w=None, h=None, fontsdir=None, jobs=4):
        """PNG per frame with the script burned in (RGB, colour-correct). Returns paths."""
        outdir = os.path.abspath(outdir)     # ffmpeg runs with cwd=outdir: a relative output path would double up
        os.makedirs(outdir, exist_ok=True)
        final = outdir
        if len(os.path.abspath(outdir)) > 180:
            # ffmpeg runs with cwd=outdir (a drive letter in the subtitles= path breaks the filter parser);
            # Windows refuses a cwd near MAX_PATH (WinError 267) - render in a short temp folder instead
            import tempfile
            outdir = tempfile.mkdtemp(prefix="tsr_")
        tmp = f"_{prefix}.ass"
        shutil.copyfile(ass_path, os.path.join(outdir, tmp))
        fd = fontsdir

        def one(n):
            vf = (self._sub_vf(tmp, fd, w, h) +
                  f",scale=in_color_matrix={self.in_cm}:in_range={self.range}:out_range=pc,format=rgb24")
            if crop:
                x0, y0, x1, y1 = crop
                vf += f",crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
            out = os.path.join(outdir, f"{prefix}_{n}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-y"] + self.hw() + ["-copyts", "-ss", f"{max(0.0, self.ft(n) - 0.002):.4f}",
                            "-i", self.path, "-frames:v", "1", "-vf", vf, out], check=True, cwd=outdir)
            return out

        with ThreadPoolExecutor(jobs) as ex:
            paths = list(ex.map(one, frames))
        if outdir != final:
            moved = []
            for p in paths:
                q = os.path.join(final, os.path.basename(p))
                shutil.move(p, q)
                moved.append(q)
            shutil.rmtree(outdir, ignore_errors=True)
            paths = moved
        return paths

    def audio_map(self):
        """-map arguments for the preview: the Japanese track when the file has one (releases carry several
        dubs and ffmpeg would take the first), else the first audio stream"""
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=index:stream_tags=language", "-of", "json", self.path], capture_output=True)
        try:
            streams = json.loads(r.stdout).get("streams", [])
        except ValueError:
            streams = []
        if not streams:
            return ["-map", "0:v:0"]
        jp = [s_ for s_ in streams if (s_.get("tags") or {}).get("language", "").lower() in ("jpn", "ja", "jp")]
        return ["-map", "0:v:0", "-map", f"0:{(jp or streams)[0]['index']}"]

    def preview(self, ass_path, ranges, out_mp4, workdir, fontsdir=None):
        """H.264 clips of the given (frame_a, frame_b) ranges with the script burned in, concatenated."""
        os.makedirs(workdir, exist_ok=True)
        amap = self.audio_map()
        shutil.copyfile(ass_path, os.path.join(workdir, "_preview.ass"))
        fd = fontsdir
        cm = "bt709" if self.matrix == "709" else "bt601"
        parts = []
        for i, (a, b) in enumerate(ranges):
            p = os.path.join(workdir, f"_prev_{i}.mp4")
            vf = self._sub_vf("_preview.ass", fd) + f",scale=out_color_matrix={cm}:out_range=tv,format=yuv420p,setpts=PTS-STARTPTS"
            ta, tb = self.ft(a), self.ft(b + 1)
            subprocess.run(["ffmpeg", "-v", "error", "-y"] + self.hw() + ["-copyts", "-ss", f"{ta:.3f}", "-t", f"{tb - ta:.3f}",
                            "-i", self.path] + amap + ["-vf", vf, "-af", "asetpts=PTS-STARTPTS"] + self.h264() + ["-colorspace", cm, "-color_primaries", "bt709" if cm == "bt709" else "smpte170m",
                            "-color_trc", "bt709" if cm == "bt709" else "smpte170m", "-c:a", "aac", "-b:a", "160k", p],
                           check=True, cwd=workdir)
            parts.append(p)
        lst = os.path.join(workdir, "_prev_list.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("".join(f"file '{os.path.basename(p)}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "_prev_list.txt",
                        "-c", "copy", os.path.abspath(out_mp4)], check=True, cwd=workdir)
        for p in parts + [lst]:
            os.remove(p)
        return out_mp4
