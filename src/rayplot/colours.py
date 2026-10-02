"""Turn whatever colours a caller has into an (N, 3) sRGB array in [0, 1]."""
import colorsys
import re

import numpy as np

_RGB = re.compile(r'rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)')
_HSL = re.compile(r'hsla?\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%')


def _parse(c):
    s = str(c).strip()
    if s.startswith('#') and len(s) in (4, 7, 9):
        h = s[1:] if len(s) != 4 else ''.join(ch * 2 for ch in s[1:])
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    m = _RGB.match(s)
    if m:
        return [float(x) / 255 for x in m.groups()]
    m = _HSL.match(s)
    if m:
        h, sat, lig = (float(x) for x in m.groups())
        return list(colorsys.hls_to_rgb(h / 360 % 1, lig / 100, sat / 100))
    from matplotlib.colors import to_rgb        # names like 'tab:blue', 'C3', 'red'
    return list(to_rgb(s))


def _unit(rgb, raw):
    """RGB numbers as 0..1: uint8, or anything above 1.5, is on the 0..255 scale. The margin
    keeps 0..1 floats that rounding pushed slightly over 1 (1.02, say) from being divided by
    255, and 0/1 integers ([[1, 0, 0], ...]) are 0..1 like any other number up to 1."""
    rgb = np.asarray(rgb, dtype=np.float64)[..., :3]
    if np.asarray(raw).dtype == np.uint8 or np.nanmax(rgb, initial=0) > 1.5:
        rgb = rgb / 255
    return np.clip(rgb, 0, 1)


def to_rgb(colours, n, cmap='viridis', vmin=None, vmax=None):
    """(n, 3) sRGB floats in [0, 1] from any of:

    - one colour for every point: a name / hex / 'rgb(...)' / 'hsl(...)' string, or a tuple of
      3 (or 4) numbers;
    - n colour strings (hex, 'rgb(r, g, b)', 'hsl(h, s%, l%)', or any matplotlib colour name);
    - an (n, 3) or (n, 4) array (alpha is ignored);
    - n numbers, mapped through `cmap` between `vmin` and `vmax` (default: their range).

    RGB numbers are 0..1, unless the array is uint8 or a value exceeds 1.5, when they are
    0..255; either way they are clipped to range. A list or array of 3 (or 4) numbers is one
    colour, except when there are exactly that many points: then it is n values for `cmap`.
    A tuple is always one colour.
    """
    if isinstance(colours, str):
        return np.tile(_parse(colours), (n, 1))
    if isinstance(colours, tuple) and len(colours) in (3, 4) and not _has_str(colours):
        return np.tile(_unit(colours, colours), (n, 1))
    arr = np.asarray(colours, dtype=object if _has_str(colours) else None)
    if arr.dtype == object:
        if arr.shape != (n,):
            raise ValueError(f'expected {n} colour strings, got shape {arr.shape}')
        return np.array([_parse(c) for c in arr], dtype=np.float64)
    if arr.shape in ((3,), (4,)) and n != arr.shape[0]:
        return np.tile(_unit(arr, arr), (n, 1))
    if arr.ndim == 1:
        if len(arr) != n:
            raise ValueError(f'expected {n} values, got {len(arr)}')
        import matplotlib
        arr = arr.astype(np.float64)
        lo = np.nanmin(arr) if vmin is None else vmin
        hi = np.nanmax(arr) if vmax is None else vmax
        t = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        return np.asarray(matplotlib.colormaps[cmap](np.clip(t, 0, 1)))[:, :3]
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] not in (3, 4):
        raise ValueError(f'colours must be (n, 3) or (n, 4) with n = {n}, got {arr.shape}')
    return _unit(arr, arr)


def _has_str(x):
    if isinstance(x, np.ndarray):
        return x.dtype.kind in 'USO'
    try:
        return len(x) > 0 and isinstance(x[0], str)
    except TypeError:
        return False
