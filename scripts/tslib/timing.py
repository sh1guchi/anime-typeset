"""Sign timing from the picture: the frames where the original (Japanese) text is really on screen.

The translator's timing (CR) is often off the original: it starts on the cut before a title fades in, ends
a few frames early, or runs past the cut where the sign is gone. Here the original's own pixels decide.
In the sign's region every frame's local contrast (detail) minus the bare background's detail is projected
onto the detail the text adds: 1 = the text fully there, 0 = gone, in between = fading (the projection is
linear in the text's opacity). The sign runs from the first to the last frame where the text shows at all,
so fades are inside it; the frames where it is fully there are reported for `fade_frames` / `fade`.

Works for text that stays put on screen (cards, titles, plates that are overlays, even over a moving
background); text that moves with the scene (attached plates, `track`, `follow`) is not judged."""
import numpy as np, cv2
from . import coords
from .cuts import detect_cuts

ON, FULL, ABS = 0.3, 0.85, 0.06   # clearly there / fully there / clearly gone


def regions(it):
    """screen boxes of the item's original text, or None when a fixed region can't judge it"""
    t = it.get("timing")
    if t is False or (isinstance(t, dict) and t.get("lock")):
        return None
    if isinstance(t, dict) and (t.get("box") or t.get("boxes")):
        return [[int(v) for v in b] for b in (t.get("boxes") or [t["box"]])]
    typ = it.get("type")
    out = []
    if typ == "card":
        for c in it.get("cards", []):
            b = c.get("box") or c.get("glow_box")
            if not b and c.get("zones"):
                zs = c["zones"]
                b = [min(z[0] for z in zs), min(z[1] for z in zs), max(z[2] for z in zs), max(z[3] for z in zs)]
            if b:
                out.append([int(v) for v in b])
    elif typ == "plate" and not it.get("attached"):
        zs = it["zones"] if "zones" in it else ([it["zone"]] if "zone" in it else [])
        for z in zs if isinstance(zs, list) else []:
            z = z.get("box") if isinstance(z, dict) else z
            if isinstance(z, (list, tuple)) and len(z) == 4 and all(isinstance(v, (int, float)) for v in z):
                out.append([int(v) for v in z])
    return out or None


def _detail(G):
    return np.stack([g - cv2.GaussianBlur(g, (0, 0), 2.5) for g in G])


def _side(L, ref, mask, idx_out, centre, step, n):
    """walk from `centre` towards one end (step -1 / +1). Background = median of the frames just beyond the
    current boundary (same shot as the boundary, unless the boundary is a cut), refined twice.
    Returns (scores, last index of the run where the text is clearly there, note)."""
    bg_idx = list(idx_out) or ([0, 1, 2] if step < 0 else [n - 3, n - 2, n - 1])
    k, s, TT = centre, None, 0.0
    for _ in range(3):
        B = np.median(L[bg_idx], axis=0) * mask
        T = (ref - B) * mask
        TT = float((T * T).sum())
        if TT < 1e-3:
            return None, None, "no text detail in the region"
        s = np.array([float(((L[j] - B) * T).sum()) / TT for j in range(n)])
        k = centre
        while 0 <= k + step < n and s[k + step] > ON:
            k += step
        # nearest frames beyond the run where the text is clearly gone (not the faint start of a fade)
        beyond = [j for j in range(k + step, -1 if step < 0 else n, step) if s[j] < ABS][:8]
        if len(beyond) < 3 or beyond == bg_idx:
            break
        bg_idx = beyond
    if k in (0, n - 1):
        return s, k, "the text is still there at the window edge"
    if np.sqrt(TT / max(1.0, float(mask.sum()))) < 1.0:
        return s, k, "weak text detail"
    return s, k, ""


def detect(video, it, pad=48):
    """{"frames": [a, b] proposed, "sure": [start_ok, end_ok], "hint": {...} for the config, "note": str} or None"""
    boxes = regions(it)
    if not boxes:
        return None
    f0, f1 = it["frames"]
    a = max(0, f0 - pad)
    b = min(video.nframes - 1, f1 + pad) if getattr(video, "nframes", 0) else f1 + pad
    x0 = max(0, min(q[0] for q in boxes)); y0 = max(0, min(q[1] for q in boxes))
    x1 = min(coords.AW, max(q[2] for q in boxes)); y1 = min(coords.AH, max(q[3] for q in boxes))
    x0 -= x0 % 2; y0 -= y0 % 2; x1 -= x1 % 2; y1 -= y1 % 2
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    G = video.grab(a, b - a + 1, crop=(x0, y0, x1, y1), gray=True).astype(np.float32)
    n = len(G)
    if n < 6:
        return None
    mask = np.zeros(G.shape[1:], np.float32)
    for q in boxes:
        mask[max(0, q[1] - y0):max(0, q[3] - y0), max(0, q[0] - x0):max(0, q[2] - x0)] = 1
    L = _detail(G) * mask
    i0, i1 = f0 - a, min(f1 - a, n - 1)
    lo, hi = i0 + (i1 - i0) * 3 // 10, i0 + (i1 - i0) * 7 // 10
    ref = np.median(L[lo:hi + 1], axis=0)
    centre = (i0 + i1) // 2
    ss, ks, ns = _side(L, ref, mask, range(0, i0), centre, -1, n)
    se, ke, ne = _side(L, ref, mask, range(i1 + 1, n), centre, +1, n)
    if ss is None or se is None:
        note = ns or ne
        return {"frames": [f0, f1], "sure": [False, False], "note": note, "hint": {"note": note}}
    # conservative: extend only over frames where the text is clearly there (plus the fade tail before /
    # after them), cut only frames where it is clearly gone - a missed faint fade is worse than a long line
    def bound(s, k_on, cur, step):
        """step -1: the start side, +1: the end side (outward). A sharp boundary (the text pops in / out within
        2 frames) is taken exactly. A fade: its faint end is the least certain, so the translator's frame stands
        when it is within a second of it, and an extension takes half the measured fade more"""
        extend = (k_on - cur) * step > 0           # clearly on screen beyond the current frame
        k = k_on
        # a fade never crosses a cut: past a cut only a text that is clearly there counts
        blocked = lambda k: max(k, k + step) in cut_idx and s[k + step] < ON
        if extend:
            while 0 <= k + step < n and s[k + step] > 2 * ABS and not blocked(k):
                k += step
        else:                                       # outermost frame still showing the text, not past cur
            while (k - cur) * step < 0 and s[k + step] > ABS and not blocked(k):
                k += step
        full = k                                    # the fade: from k inward to the first frame fully there
        while full != centre and 0 <= full - step < n and s[full] < FULL:
            full -= step
        ramp = abs(full - k)
        if extend:
            return min(n - 1, max(0, k + step * ((ramp + 1) // 2))) if ramp > 2 else k
        across = any(min(k, cur) < j <= max(k, cur) for j in cut_idx)
        if ramp > 2 and abs(k - cur) <= fps1 and not across:
            return cur
        return k
    fps1 = int(round(float(video.fps)))
    cut_idx = {c - a for c in detect_cuts(video, a, b)}
    sure_s, sure_e = not ns, not ne
    ks2 = bound(ss, ks, i0, -1) if sure_s else i0
    ke2 = bound(se, ke, i1, +1) if sure_e else i1
    start, end = a + ks2, a + ke2
    full_in = next((a + k for k in range(ks2, ke2 + 1) if ss[k] >= FULL), None)
    full_out = next((a + k for k in range(ke2, ks2 - 1, -1) if se[k] >= FULL), None)
    hint = {"on_screen": [start, end], "fade_in": [start, full_in] if full_in and full_in - start >= 3 else None,
            "fade_out": [full_out, end] if full_out and end - full_out >= 3 else None,
            "translator": it.get("source_frames"), "sure": [sure_s, sure_e]}
    note = "; ".join(x for x in (("start: " + ns) if ns else "", ("end: " + ne) if ne else "") if x)
    if note:
        hint["note"] = note
    return {"frames": [start, end], "sure": [sure_s, sure_e], "hint": hint, "note": note}


def fade_ms(video, hint):
    """fade in / out of the original in ms (for a text spec's "fade"), 0 when it pops"""
    fi = hint.get("fade_in"); fo = hint.get("fade_out")
    ms = lambda p: int(round((video.ft(p[1]) - video.ft(p[0])) * 1000)) if p else 0
    return ms(fi), ms(fo)
