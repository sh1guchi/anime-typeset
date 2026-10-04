"""Images for looking at the video: frame grids, rulers, check sheets of rendered signs."""
import os
import numpy as np
from . import coords
from PIL import Image, ImageDraw


def label(im, text, color=(255, 255, 0)):
    d = ImageDraw.Draw(im)
    w = int(d.textlength(text)) + 8
    d.rectangle((0, 0, w, 14), fill=(0, 0, 0))
    d.text((4, 2), text, fill=color)
    return im


def tile(images, cols, out, quality=90):
    w = max(i.width for i in images); h = max(i.height for i in images)
    rows = (len(images) + cols - 1) // cols
    c = Image.new("RGB", (w * cols, h * rows))
    for i, im in enumerate(images):
        c.paste(im, ((i % cols) * w, (i // cols) * h))
    c.save(out, quality=quality)
    return out


def grid(video, frames, out, crop=None, cols=4, width=None):
    """frames (list of numbers) cropped to crop=(x0,y0,x1,y1) analysis px, tiled with frame labels"""
    frames = sorted(frames)
    ims = []
    i = 0
    while i < len(frames):   # decode contiguous runs at once
        j = i
        while j + 1 < len(frames) and frames[j + 1] - frames[i] < 48:
            j += 1
        a, b = frames[i], frames[j]
        fr = video.grab(a, b - a + 1, crop=crop)
        for f in frames[i:j + 1]:
            im = Image.fromarray(fr[f - a])
            if width:
                im = im.resize((width, int(im.height * width / im.width)))
            ims.append(label(im, str(f)))
        i = j + 1
    return tile(ims, cols, out)


def ruler(img, box, out, scale=1.5, std=None):
    """crop of img (analysis px) with coordinate ticks every 10px (labels every 50/100) - for reading positions"""
    x0, y0, x1, y1 = box
    panels = [np.clip(img[y0:y1, x0:x1], 0, 255).astype(np.uint8)]
    if std is not None:
        s = np.clip(std[y0:y1, x0:x1] * 8, 0, 255).astype(np.uint8)
        panels.append(np.repeat(s[..., None], 3, 2))
    res = []
    for p in panels:
        im = Image.fromarray(p).resize((int((x1 - x0) * scale), int((y1 - y0) * scale)), Image.NEAREST)
        d = ImageDraw.Draw(im)
        for x in range((x0 // 10 + 1) * 10, x1, 10):
            X = (x - x0) * scale; big = x % 50 == 0
            d.line((X, 0, X, 14 if big else 5), fill=(255, 255, 0) if big else (0, 255, 0))
            if x % 100 == 0:
                d.text((X + 2, 14), str(x), fill=(255, 255, 0))
        for y in range((y0 // 10 + 1) * 10, y1, 10):
            Y = (y - y0) * scale; big = y % 50 == 0
            d.line((0, Y, 14 if big else 5, Y), fill=(255, 255, 0) if big else (0, 255, 0))
            if y % 50 == 0:
                d.text((16, Y - 5), str(y), fill=(255, 255, 0))
        res.append(im)
    c = Image.new("RGB", (res[0].width, sum(r.height for r in res)))
    y = 0
    for r in res:
        c.paste(r, (0, y)); y += r.height
    c.save(out)
    return out


def check_frames(f0, f1, moving):
    pts = [f0, (f0 + f1) // 2, f1] if not moving else [f0, f0 + (f1 - f0) // 4, (f0 + f1) // 2, f0 + 3 * (f1 - f0) // 4, f1]
    return sorted(set(pts))


def fit_area(area, min_w=760, min_h=300):
    x0, y0, x1, y1 = area
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, min_w), max(y1 - y0, min_h)
    AW, AH = coords.AW, coords.AH
    w, h = min(w, AW), min(h, AH)
    x0 = int(max(0, min(AW - w, cx - w / 2))); y0 = int(max(0, min(AH - h, cy - h / 2)))
    return [x0, y0, int(min(AW, x0 + w)), int(min(AH, y0 + h))]


def check_sheet(ctx, item, ass_path, out, frames=None, full=False, zoom=None, compare=False):
    """render the item's lines over its frames (cropped to its area, or full frame) into one image.
    zoom=(x0,y0,x1,y1): crop to this box and enlarge; compare: original | result | difference (x4) per frame."""
    f0, f1 = item["frames"]
    data = ctx.load_lines(item["id"]) or {}
    moving = item.get("type") in ("follow",) or (isinstance(item.get("motion"), dict) and item["motion"].get("mode", "static") != "static") \
        or item.get("motion") in ("linear", "path", "affine")
    frames = frames or check_frames(f0, f1, moving)
    if zoom:
        crop = list(zoom)
    else:
        crop = None if full or not data.get("area") else fit_area(data["area"])
    paths = ctx.video.render(ass_path, frames, ctx.path("check", "_tmp"), prefix=item["id"], crop=crop)
    ims = []
    width = 960 if not zoom else min(1400, max(960, 2 * (crop[2] - crop[0])))
    for f, p in zip(frames, paths):
        im = Image.open(p).convert("RGB")
        if compare:
            o = ctx.video.grab(f, 1, crop=crop)[0] if crop else ctx.video.grab(f, 1)[0]
            r = np.asarray(im, np.int16)
            d = np.clip(np.abs(r - o.astype(np.int16)).max(axis=2) * 4, 0, 255).astype(np.uint8)
            row = [Image.fromarray(o), im, Image.fromarray(np.stack([d] * 3, 2))]
            w1 = width // 2 if zoom else 640
            row = [x.resize((w1, int(x.height * w1 / x.width)), Image.LANCZOS) for x in row]
            c = Image.new("RGB", (w1 * 3 + 8, row[0].height))
            for k, x in enumerate(row):
                c.paste(x, (k * (w1 + 4), 0))
            ims.append(label(c, f"{item['id']} {f}: original | result | difference x4"))
        else:
            if full:
                im = im.resize((960, int(960 * coords.AH / coords.AW)))
            elif im.width != width:
                im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
            ims.append(label(im, f"{item['id']} {f}"))
        os.remove(p)
    return tile(ims, 1 if not full or compare else 2, out)
