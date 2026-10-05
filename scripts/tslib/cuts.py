"""Cuts (shot changes) in a frame range - shared by analyze (timing snap, sheets) and timing (a fade never
crosses a cut, so a cut bounds the sign)."""
import numpy as np


def detect_cuts(video, a, b):
    a = max(0, a)
    G = video.grab(a, b - a + 1, w=256, h=144, gray=True).astype(np.float32)
    G = 255.0 * np.sqrt(G / 255.0)      # lift shadows: cuts between dark shots count too
    if len(G) < 2:
        return []
    diffs = np.abs(np.diff(G, axis=0)).mean(axis=(1, 2))
    thr = max(18.0, 5 * float(np.median(diffs)))
    return drop_flashes(G, a, [a + i + 1 for i, dv in enumerate(diffs) if dv > thr], thr)


def drop_flashes(G, a, cuts, thr, gap=6):
    """A flash (white/black frames, an explosion) fires a run of 'cuts' and the shot comes back the same: drop
    such runs (the picture after the run matches the one before it). Runs that do lead elsewhere (a flash
    transition, an animated logo) stay whole - snapping the sign timing needs every real jump."""
    runs, out = [], []
    for c in cuts:
        if runs and c - runs[-1][-1] <= gap:
            runs[-1].append(c)
        else:
            runs.append([c])
    for r in runs:
        if len(r) > 1:
            before, after = G[r[0] - 1 - a], G[min(r[-1], a + len(G) - 1) - a]
            if float(np.abs(after - before).mean()) < thr:
                continue                                   # the same shot again: a flash, not a cut
        out += r
    return out


def cut_runs(cuts, gap=6):
    """for reading: 20326..20338 (13) instead of thirteen numbers"""
    runs = []
    for c in cuts:
        if runs and c - runs[-1][-1] <= gap:
            runs[-1].append(c)
        else:
            runs.append([c])
    return ", ".join(str(r[0]) if len(r) == 1 else f"{r[0]}..{r[-1]} ({len(r)})" for r in runs)
