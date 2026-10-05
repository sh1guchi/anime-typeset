"""Neural inpainting (LaMa) for the clean plate: where no frame of the video shows the background (the text is
there from the first frame of the shot) and the background has texture (merlons, brick, wood grain, paper),
a smooth harmonic fill looks like a smudge; LaMa continues the texture.

It only replaces where the plate's colours come from: the result is one image like any other clean plate,
then quantised into the same mosaic and moved by the same track - the file stays as small as before.

The model runs in its own PyTorch environment (torch is big and pins its own numpy): set it up once and
point the tool at it - environment variables TS_LAMA_PYTHON / TS_LAMA_MODEL, or ~/.anime-typeset.json
{"lama_python": "...\\Scripts\\python.exe", "lama_model": "...\\big-lama.pt"}."""
import os, json, subprocess, tempfile
import numpy as np

WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lama_worker.py")


def setup():
    cfg = {}
    p = os.path.join(os.path.expanduser("~"), ".anime-typeset.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
    py = os.environ.get("TS_LAMA_PYTHON") or cfg.get("lama_python")
    model = os.environ.get("TS_LAMA_MODEL") or cfg.get("lama_model")
    if not py or not model or not os.path.isfile(py) or not os.path.isfile(model):
        raise SystemExit("clean.fill \"lama\": the LaMa environment is not set up - see references/technique.md "
                         "(TS_LAMA_PYTHON / TS_LAMA_MODEL or ~/.anime-typeset.json)")
    return py, model


def fill(P, mask, pad=96):
    """P (H,W,3) float/uint8 analysis-resolution plate; mask (H,W) bool = pixels to replace.
    Each connected area of the mask is cropped with `pad` px of context and filled; only masked pixels change."""
    if not mask.any():
        return P
    py, model = setup()
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask.astype(np.uint8), np.ones((2 * pad + 1,) * 2, np.uint8)))
    H, W = mask.shape
    jobs, boxes = {}, []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        k = f"{len(boxes):03d}"
        jobs["img_" + k] = np.clip(P[y0:y1, x0:x1], 0, 255).astype(np.uint8)
        jobs["mask_" + k] = mask[y0:y1, x0:x1].astype(np.uint8)
        boxes.append((k, x0, y0, x1, y1))
    with tempfile.TemporaryDirectory(prefix="tslama_") as d:
        jp, op = os.path.join(d, "jobs.npz"), os.path.join(d, "out.npz")
        np.savez(jp, **jobs)
        r = subprocess.run([py, WORKER, model, jp, op], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("LaMa worker failed:\n" + (r.stderr or r.stdout)[-2000:])
        print("    " + r.stdout.strip().splitlines()[-1], flush=True)
        with np.load(op) as z:              # read it all and close: Windows won't delete an open file
            out = {k: z[k] for k in z.files}
        Q = P.copy()
        for k, x0, y0, x1, y1 in boxes:
            sub = mask[y0:y1, x0:x1]
            Q[y0:y1, x0:x1][sub] = out["out_" + k][sub].astype(Q.dtype)
    return Q
