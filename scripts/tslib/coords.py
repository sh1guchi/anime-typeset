"""Coordinate spaces.

All analysis happens on frames scaled to a fixed width of 1920 ("analysis px"); the height follows the
video's aspect ratio (1080 for 16:9, 816 for 2.35:1 3840x1632, ...). The output script uses its own
PlayRes (often 640x360 / 640x272 for fansub scripts); K converts analysis px -> PlayRes units.
"""
AW, AH = 1920, 1080
K = 1 / 3


def set_frame(width, height):
    """analysis frame size for a video of width x height (even height, same aspect)"""
    global AH
    AH = int(round(AW * height / width / 2) * 2)


def set_playres(playres_x):
    global K
    K = playres_x / float(AW)


def shape():
    return (AH, AW)


def f2(v):
    """number -> short string with at most 2 decimals"""
    v = round(float(v), 2)
    return str(int(v)) if v.is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")


def P(v):
    """analysis px -> PlayRes number string"""
    return f2(v * K)


def scl(block=1.0):
    """\\fscx/\\fscy value that makes drawing units equal `block` analysis px"""
    return f"{100 * block * K:.4f}".rstrip("0").rstrip(".")
