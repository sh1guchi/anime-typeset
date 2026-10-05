"""LaMa inpainting worker - runs in the separate PyTorch environment (see references/technique.md, "LaMa").

    <ai python> lama_worker.py <model.pt> <jobs.npz> <out.npz>

jobs.npz: img_<k> uint8 (H,W,3) RGB, mask_<k> uint8 (H,W) (non-zero = fill). out.npz: out_<k> uint8 (H,W,3).
The main tool never imports torch: it writes the crops, runs this script and reads the result back."""
import sys, time
import numpy as np
import torch


def pad8(a, mode):
    h, w = a.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    pads = ((0, ph), (0, pw)) + (((0, 0),) if a.ndim == 3 else ())
    return np.pad(a, pads, mode=mode), h, w


def main():
    model_path, jobs_path, out_path = sys.argv[1:4]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = time.time()
    model = torch.jit.load(model_path, map_location=dev)
    model.eval()
    jobs = np.load(jobs_path)
    keys = sorted(k[4:] for k in jobs.files if k.startswith("img_"))
    out = {}
    with torch.inference_mode():
        for k in keys:
            img, h, w = pad8(jobs["img_" + k], "symmetric")
            m, _, _ = pad8(jobs["mask_" + k], "constant")
            it = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(dev)
            mt = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(dev)
            res = model(it, mt)[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
            out["out_" + k] = res[:h, :w]
    np.savez(out_path, **out)
    print(f"lama: {len(keys)} crop(s) on {dev.type} in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
