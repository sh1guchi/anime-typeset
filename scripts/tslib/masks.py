"""Glyph masks and smooth (harmonic) background fills."""
import numpy as np, cv2
from . import coords


def white_mask(img, zones, thr=28, minc=140, win=31, absw=None):
    """Bright glyph pixels inside zones [(x0,y0,x1,y1)].
    A pixel counts when it is `thr` brighter than the local median (window `win`) and its darkest channel
    exceeds `minc`. absw: also accept near-neutral pixels whose darkest channel exceeds this value
    (dense glyph clusters where the local median itself is bright)."""
    H, W, _ = img.shape
    lum = img.mean(axis=2).astype(np.float32)
    bg = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), win).astype(np.float32)
    white = ((lum - bg) > thr) & (img.min(axis=2) > minc)
    if absw is not None:
        white |= (img.min(axis=2) > absw) & ((img.max(axis=2) - img.min(axis=2)) < 30)
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in zones:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        m[y0:int(y1), x0:int(x1)] = white[y0:int(y1), x0:int(x1)]
    return m


def dark_mask(img, zones, thr=28, maxc=140, win=31):
    """Dark glyph pixels (dark text on a light background): darker than the local median by `thr`
    and the brightest channel below `maxc`."""
    H, W, _ = img.shape
    lum = img.mean(axis=2).astype(np.float32)
    bg = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), win).astype(np.float32)
    dark = ((bg - lum) > thr) & (img.max(axis=2) < maxc)
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in zones:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        m[y0:int(y1), x0:int(x1)] = dark[y0:int(y1), x0:int(x1)]
    return m


def glyph_mask(img, zones, det, thr=20, minc=120, win=61):
    """light or dark glyphs depending on det['polarity'] ('light' default / 'dark')"""
    if det.get("polarity", "light") == "dark":
        return dark_mask(img, zones, thr=det.get("thr", thr), maxc=det.get("maxc", 140), win=det.get("win", win))
    return white_mask(img, zones, thr=det.get("thr", thr), minc=det.get("minc", minc), win=det.get("win", win),
                      absw=det.get("absw"))


def dilate(m, r):
    if r <= 0:
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def fill_holes(m):
    """m with every enclosed hole filled (pixels not reachable from the image border through ~m)"""
    inv = np.pad(~m.astype(bool), 1, constant_values=True).astype(np.uint8)
    ff = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 2)
    return m.astype(bool) | (inv[1:-1, 1:-1] == 1)


def close(m, kx, ky=1):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((ky, kx), np.uint8)).astype(bool)


def rect_mask(rects, shape=None):
    m = np.zeros(shape or coords.shape(), bool)
    for x0, y0, x1, y1 in rects:
        m[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True
    return m


def laplace(f, U, iters=None, tol=1e-5, maxit=5000):
    """Solve Laplace on unknown pixels U with Dirichlet data f elsewhere: conjugate gradients on the
    5-point stencil (the matrix 4I - adjacency over U is SPD), all channels at once. Converges in a few
    hundred steps where plain Jacobi needed tens of thousands (1500 sweeps left zone-sized fills ~5 levels
    off). `iters` is accepted for compatibility and ignored. U must not touch the array border."""
    f = f.astype(np.float32).copy()
    if not U.any():
        return f
    ys, xs = np.where(U)
    y0, y1 = max(0, ys.min() - 1), min(U.shape[0], ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(U.shape[1], xs.max() + 2)
    F = f[y0:y1, x0:x1].astype(np.float64)
    Ub = U[y0:y1, x0:x1]
    if F.ndim == 2:
        F = F[..., None]
    Um = Ub[..., None].astype(np.float64)

    def nsum(X):
        S = np.zeros_like(X)
        S[1:] += X[:-1]; S[:-1] += X[1:]; S[:, 1:] += X[:, :-1]; S[:, :-1] += X[:, 1:]
        return S

    K = F * (1 - Um)                       # known values (zero on unknowns)
    b = nsum(K) * Um                       # known neighbours feed the right-hand side
    w = (1 - Um[..., 0]).astype(np.float32)
    H, W = Ub.shape
    if Ub.sum() > 12000 and min(H, W) > 48:
        # start from the solution at half resolution (known-weighted downsample): CG then only has to fix
        # the fine detail
        h2, w2 = (H + 1) // 2, (W + 1) // 2
        num = cv2.resize((K * (1 - Um)).astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA).reshape(h2, w2, -1)
        den = cv2.resize(w, (w2, h2), interpolation=cv2.INTER_AREA)
        Uc = den <= 0.5
        Uc[0, :] = Uc[-1, :] = Uc[:, 0] = Uc[:, -1] = False
        fc = num / np.maximum(den, 1e-6)[..., None]
        sc = laplace(fc, Uc, tol=tol * 10, maxit=maxit).reshape(h2, w2, -1)
        est = cv2.resize(sc, (W, H), interpolation=cv2.INTER_LINEAR).reshape(F.shape)
    else:                                  # a known-weighted blur (close for small smooth fills)
        est = cv2.GaussianBlur(K.astype(np.float32), (0, 0), 8).reshape(K.shape) / \
            np.maximum(cv2.GaussianBlur(w, (0, 0), 8), 1e-6)[..., None]
    x = est.astype(np.float64) * Um
    A = lambda X: (4 * X - nsum(X)) * Um
    r = b - A(x)
    p = r.copy()
    rs = (r * r).sum(axis=(0, 1))
    bn = np.maximum((b * b).sum(axis=(0, 1)), 1e-12)
    for _ in range(maxit):
        if (rs / bn).max() < tol ** 2:
            break
        Ap = A(p)
        al = rs / np.maximum((p * Ap).sum(axis=(0, 1)), 1e-30)
        x += al * p
        r -= al * Ap
        rs2 = (r * r).sum(axis=(0, 1))
        p = r + (rs2 / np.maximum(rs, 1e-30)) * p
        rs = rs2
    out = np.where(Um > 0, x, F)
    f[y0:y1, x0:x1] = (out if f.ndim == 3 else out[..., 0]).astype(np.float32)
    return f


def laplace_jacobi(f, U, iters):
    """Old solver: coarse-to-fine Jacobi (kept for comparison)."""
    f = f.astype(np.float32).copy()
    H, W = U.shape
    if U.sum() == 0:
        return f
    if min(H, W) > 24 and U.sum() > 400:
        h2, w2 = (H + 1) // 2, (W + 1) // 2
        fc = cv2.resize(f, (w2, h2), interpolation=cv2.INTER_AREA)
        Uc = cv2.resize(U.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST).astype(bool)
        # coarse known pixels must not average unknown garbage: known-weighted downsample
        wk = (~U).astype(np.float32)
        if f.ndim == 3:
            num = cv2.resize(f * wk[..., None], (w2, h2), interpolation=cv2.INTER_AREA)
            den = cv2.resize(wk, (w2, h2), interpolation=cv2.INTER_AREA)[..., None]
            fc = np.where(den > 0.5, num / np.maximum(den, 1e-6), fc)
            Uc = Uc | (den[..., 0] <= 0.5)
        else:
            num = cv2.resize(f * wk, (w2, h2), interpolation=cv2.INTER_AREA)
            den = cv2.resize(wk, (w2, h2), interpolation=cv2.INTER_AREA)
            fc = np.where(den > 0.5, num / np.maximum(den, 1e-6), fc)
            Uc = Uc | (den <= 0.5)
        Uc[0, :] = Uc[-1, :] = Uc[:, 0] = Uc[:, -1] = False
        sc = laplace_jacobi(fc, Uc, max(200, iters // 2))
        up = cv2.resize(sc, (W, H), interpolation=cv2.INTER_LINEAR)
        f[U] = up[U]
    else:
        w = (~U).astype(np.float32)
        if f.ndim == 3:
            est = cv2.GaussianBlur(f * w[..., None], (0, 0), 8) / np.maximum(cv2.GaussianBlur(w, (0, 0), 8), 1e-6)[..., None]
        else:
            est = cv2.GaussianBlur(f * w, (0, 0), 8) / np.maximum(cv2.GaussianBlur(w, (0, 0), 8), 1e-6)
        f[U] = est[U]
    for _ in range(iters):
        nb = 0.25 * (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1))
        f[U] = nb[U]
    return f


def harmonic_fill(img, unknown, box, iters=1500, presmooth=5):
    """Smooth fill of `unknown` pixels inside box=(x0,y0,x1,y1) from the known (median-presmoothed) border.
    Good for skies, glows, gradients; textures (bricks) come out blurred."""
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    sub = img[y0:y1, x0:x1].astype(np.float32)
    if presmooth:
        sub = cv2.medianBlur(np.clip(sub, 0, 255).astype(np.uint8), presmooth).astype(np.float32)
    U = unknown[y0:y1, x0:x1].copy()
    U[0, :] = U[-1, :] = U[:, 0] = U[:, -1] = False
    f = laplace(sub, U, iters)
    out = img.astype(np.float32).copy()
    o = out[y0:y1, x0:x1]
    o[U] = f[U]
    return out


def text_mask_zone(med, zones, dil=28, thr=28, minc=140, close_k=(25, 5), win=31, absw=None):
    """Region to replace: glyphs + their glow (dilated), limited to the zones."""
    w = white_mask(med, zones, thr=thr, minc=minc, win=win, absw=absw)
    m = close(dilate(w, dil), close_k[0], close_k[1])
    return m & rect_mask(zones, m.shape)
