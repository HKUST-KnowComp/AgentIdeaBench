r"""Minimal SVG-to-matplotlib converter for flat icon files.

The paper figures need the referee icon (reports/assets/referee-svgrepo-com.svg)
drawn inside a matplotlib schematic. No SVG rasteriser is installed in this
environment (cairosvg / rsvg-convert / inkscape are all absent), and rasterising
would put a bitmap into an otherwise vector PDF, so this module parses the
subset of SVG that flat icons use and returns matplotlib Paths.

Supported: <path d="..."> with M/m L/l H/h V/v C/c S/s Z/z, and
<polygon points="...">, each with an inline style="fill:#rrggbb". Elements are
returned in document order so painter's-algorithm layering is preserved.
Arcs (A/a) and quadratics (Q/q/T/t) are not supported and raise, rather than
being silently dropped.

Nothing runs on import.
"""
import re

from matplotlib.path import Path

_CMD = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])")
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_ELEM = re.compile(r"<(path|polygon)\b([^>]*)>", re.S)
_FILL = re.compile(r"fill:\s*(#[0-9a-fA-F]{6}|none)")
_ATTR = re.compile(r'(d|points|style)\s*=\s*"([^"]*)"', re.S)
_VIEWBOX = re.compile(r'viewBox\s*=\s*"([^"]+)"')


def _pairs(nums):
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _parse_d(d):
    """Return (vertices, codes) for one path data string."""
    verts, codes = [], []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    last_ctrl = None
    last_cmd = ""
    tok = _CMD.split(d)
    i = 1
    while i < len(tok):
        cmd = tok[i]
        nums = [float(x) for x in _NUM.findall(tok[i + 1])] if i + 1 < len(tok) else []
        i += 2
        rel = cmd.islower()
        c = cmd.upper()

        if c in "QTA":
            raise ValueError(f"unsupported SVG path command {cmd!r}")

        if c == "Z":
            verts.append(start)
            codes.append(Path.CLOSEPOLY)
            cur = start
            last_ctrl = None
            last_cmd = c
            continue

        if c == "M":
            pts = _pairs(nums)
            for k, (x, y) in enumerate(pts):
                p = (cur[0] + x, cur[1] + y) if rel else (x, y)
                verts.append(p)
                codes.append(Path.MOVETO if k == 0 else Path.LINETO)
                cur = p
                if k == 0:
                    start = p
            last_ctrl = None

        elif c == "L":
            for x, y in _pairs(nums):
                p = (cur[0] + x, cur[1] + y) if rel else (x, y)
                verts.append(p)
                codes.append(Path.LINETO)
                cur = p
            last_ctrl = None

        elif c in "HV":
            for v in nums:
                if c == "H":
                    p = (cur[0] + v, cur[1]) if rel else (v, cur[1])
                else:
                    p = (cur[0], cur[1] + v) if rel else (cur[0], v)
                verts.append(p)
                codes.append(Path.LINETO)
                cur = p
            last_ctrl = None

        elif c == "C":
            pts = _pairs(nums)
            for k in range(0, len(pts) - 2, 3):
                trio = pts[k:k + 3]
                abs_trio = [(cur[0] + x, cur[1] + y) if rel else (x, y)
                            for x, y in trio]
                verts.extend(abs_trio)
                codes.extend([Path.CURVE4] * 3)
                last_ctrl = abs_trio[1]
                cur = abs_trio[2]

        elif c == "S":
            pts = _pairs(nums)
            for k in range(0, len(pts) - 1, 2):
                duo = pts[k:k + 2]
                abs_duo = [(cur[0] + x, cur[1] + y) if rel else (x, y)
                           for x, y in duo]
                if last_cmd in ("C", "S") and last_ctrl is not None:
                    c1 = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1])
                else:
                    c1 = cur
                verts.extend([c1, abs_duo[0], abs_duo[1]])
                codes.extend([Path.CURVE4] * 3)
                last_ctrl = abs_duo[0]
                cur = abs_duo[1]

        last_cmd = c

    return verts, codes


def load_icon(svg_path):
    """Parse an icon file.

    Returns (shapes, viewbox) where shapes is a document-ordered list of
    (matplotlib.path.Path, fill_colour) and viewbox is (minx, miny, w, h).
    """
    src = open(svg_path).read()
    vb = _VIEWBOX.search(src)
    viewbox = tuple(float(v) for v in vb.group(1).split()) if vb else (0, 0, 1, 1)

    shapes = []
    for m in _ELEM.finditer(src):
        tag, attrs = m.group(1), m.group(2)
        a = dict((k, v) for k, v in _ATTR.findall(attrs))
        fill = _FILL.search(a.get("style", ""))
        colour = fill.group(1) if fill else "#000000"
        if colour == "none":
            continue
        if tag == "path":
            verts, codes = _parse_d(a.get("d", ""))
        else:
            nums = [float(x) for x in _NUM.findall(a.get("points", ""))]
            pts = _pairs(nums)
            if not pts:
                continue
            verts = pts + [pts[0]]
            codes = ([Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
                     + [Path.CLOSEPOLY])
        if verts:
            shapes.append((Path(verts, codes), colour))
    return shapes, viewbox


def icon_patches(svg_path, cx, cy, height, aspect, zorder=5, lw=0.0):
    """Build PathPatches for the icon, centred on (cx, cy) in axes coords.

    height is the icon height in y-axes-units; aspect is (fig_width_in /
    fig_height_in), used so the icon keeps its true proportions on a
    non-square canvas.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D

    shapes, (vx, vy, vw, vh) = load_icon(svg_path)
    sy = height / vh
    sx = sy / aspect                      # equal physical size on both axes
    tr = (Affine2D()
          .translate(-vx - vw / 2.0, -vy - vh / 2.0)
          .scale(sx, -sy)                 # SVG y grows downward
          .translate(cx, cy))
    return [PathPatch(tr.transform_path(p), facecolor=col, edgecolor="none",
                      lw=lw, zorder=zorder) for p, col in shapes]
