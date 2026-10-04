"""Fast regression check of analyze's draft hints (card motion, plate zones) against a known-good config.

    python tests/check_analyze.py --draft examples/black-clover-2-01.draft.json
                                  --truth examples/black-clover-2-01.episode.json --video "<the episode video>"
                                  [--cache DIR] [--only cards|plates]

--draft  the episode.json `TS analyze` wrote (frames / cards / source_boxes per item; draft TODOs are fine).
--video  the episode video (default: the draft's "video", relative to the draft's folder).
--truth  a finished config of the same episode; items are matched by id (a draft item split later, like
         salli -> salli/baltos/rades, is judged by the truth item with the same id).
The first run grabs what card_motion and glyph_zone look at (960 px gray frames per card item, a 12-frame
median per plate) into the cache (default %TEMP%/anime-typeset-check/<video name>; ~200 MB for a 4K episode,
1-2 min). Later runs take seconds, so the heuristics in tslib/analyze.py can be tuned without re-running
analyze (3+ min). Delete the cache folder when done.

Card: OK when static vs moving matches the truth (linear vs path is shown, not counted - both work for a
near-uniform pan). Plate: worst zone edge error vs the truth zone, OK at <= 20 px.
"""
import argparse, json, os, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from tslib import coords, analyze          # noqa: E402
from tslib.video import Video              # noqa: E402

ZONE_TOL = 20


class CachedVideo:
    """stands in for Video.grab inside card_motion: serves the cached frames"""
    def __init__(self, G, s0):
        self.G, self.s0 = G, s0

    def grab(self, n, count=1, **kw):
        return self.G[n - self.s0:n - self.s0 + count]


def abs_path(cfg_path, p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.path.dirname(cfg_path), p))


def mode_of(it):
    m = it.get("motion", "static")
    return m.get("mode", "static") if isinstance(m, dict) else m


def fill_cache(v, items, cache):
    os.makedirs(cache, exist_ok=True)
    for it in items:
        out = os.path.join(cache, it["id"] + ".npz")
        if os.path.exists(out):
            continue
        s0, s1 = it["frames"]
        t = time.time()
        if it.get("type") == "card":
            sw = 960; sh = int(round(sw * coords.AH / coords.AW / 2) * 2)      # as card_motion grabs
            np.savez(out, G=v.grab(s0, s1 - s0 + 1, w=sw, h=sh, gray=True))
        else:                                                                    # as analyze.run's median
            mid = (s0 + s1) // 2
            a = max(s0, mid - 6)
            np.savez(out, med=np.median(v.grab(a, min(12, s1 - a + 1)), axis=0).astype(np.float32))
        print(f"  cached {it['id']} ({time.time() - t:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--video")
    ap.add_argument("--cache")
    ap.add_argument("--only", choices=("cards", "plates"))
    a = ap.parse_args()
    draft = json.load(open(a.draft, encoding="utf-8"))
    truth = {it["id"]: it for it in json.load(open(a.truth, encoding="utf-8"))["items"]}
    video = a.video or abs_path(a.draft, draft["video"])
    v = Video(video)                                   # sets the analysis frame size
    coords.set_playres((draft.get("playres") or [640, 360])[0])
    cache = a.cache or os.path.join(tempfile.gettempdir(), "anime-typeset-check",
                                    os.path.splitext(os.path.basename(video))[0])
    items = [it for it in draft["items"] if it["id"] in truth and it.get("frames")]
    fill_cache(v, items, cache)

    cards_ok = cards_n = plates_ok = plates_n = 0
    for it in items:
        d = np.load(os.path.join(cache, it["id"] + ".npz"))
        tr = truth[it["id"]]
        if it.get("type") == "card" and tr.get("type") == "card" and a.only != "plates":
            s0, s1 = it["frames"]
            t = time.time()
            m, res = analyze.card_motion(CachedVideo(d["G"], s0), s0, s1, [c["box"] for c in it["cards"]])
            got, want = (m if isinstance(m, str) else m["mode"]), mode_of(tr)
            good = (got == "static") == (want == "static")
            cards_n += 1; cards_ok += good
            note = "" if isinstance(m, str) else m.get("_analyze", "")
            print(f"{'OK ' if good else 'BAD'} card  {it['id']:22s} {got:7s} truth {want:7s} {note} "
                  f"[{time.time() - t:.1f}s]")
        elif it.get("type") != "card" and "med" in d and a.only != "cards":
            tz = tr.get("zone") or (tr.get("zones") or [None])[0]
            tz = tz["box"] if isinstance(tz, dict) else tz
            if not tz:
                continue
            for b in it.get("source_boxes") or []:
                if not b:
                    continue
                gz = analyze.glyph_zone(d["med"], b)
                err = max(abs(p - q) for p, q in zip(gz[0], tz)) if gz else None
                good = err is not None and err <= ZONE_TOL
                plates_n += 1; plates_ok += good
                print(f"{'OK ' if good else 'BAD'} plate {it['id']:22s} -> {gz[0] if gz else None} "
                      f"{gz[1] if gz else ''} truth {tz} worst edge {err}")
    print(f"cards {cards_ok}/{cards_n}, plates {plates_ok}/{plates_n} (cache: {cache})")
    return 0 if cards_ok == cards_n and plates_ok == plates_n else 1


if __name__ == "__main__":
    sys.exit(main())
