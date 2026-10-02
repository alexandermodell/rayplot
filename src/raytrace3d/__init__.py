"""raytrace3d: path-traced 3-D scatter plots (beads over a white floor that catches their shadows).

    import raytrace3d as rt

    # a still image (PIL.Image; shows inline in a notebook)
    img = rt.render(points, colours, out='figure.png')

    # interactive, in a notebook cell
    rt.notebook(points, colours)

    # interactive, in a web browser (http://127.0.0.1:17100/ by default)
    v = rt.browser(points, colours, port=17100)
    ...
    v.stop()

`points` is (n, 3); the third coordinate is up. Scale and offset do not matter (the cloud is
centred and fitted to the frame). `colours` can be one colour, n colour strings (hex,
'rgb(...)', 'hsl(...)', matplotlib names), an (n, 3)/(n, 4) array (floats 0..1 or ints
0..255), or n numbers mapped through `cmap`.

Interactive viewers: drag the image to turn the points freely; the xy / yz / xz sliders turn
them within one plane only; a tickbox under them turns each data axis over (`flip=` sets them to
begin with); other sliders set camera height, zoom, bead size and the floor gap.
"Save PNG" (or `.save()`) renders the current view at print quality into `save_dir`, and
`.view` gives the current settings to reproduce it with `rt.render(points, colours, **v.view)`.

`rt.points_from_notebook(path, 'layer 28')` reads the points and colours of a 3-D plotly figure
saved in a notebook's outputs.

Needs an NVIDIA GPU (Mitsuba 3, `cuda_ad_rgb` variant, set on import) and torch with CUDA.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .colours import pure_hue, to_rgb
from .core import FloorScene, Style, plane_rotation
from .plotly_io import notebook_traces, points_from_notebook

__all__ = ['render', 'notebook', 'browser', 'Style', 'plane_rotation', 'to_rgb', 'pure_hue',
           'points_from_notebook', 'notebook_traces']


def _prepare(points, colours, cmap, vmin, vmax, hue_only):
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'points must be (n, 3), got {xyz.shape}')
    ok = np.isfinite(xyz).all(1)
    rgb = to_rgb(colours, len(xyz), cmap=cmap, vmin=vmin, vmax=vmax)
    if hue_only:
        rgb = pure_hue(rgb)
    if not ok.all():
        print(f'raytrace3d: dropping {int((~ok).sum())} points with non-finite coordinates')
    return xyz[ok], rgb[ok]


def render(points, colours, out=None, *, res=2400, spp=1024, elevation=30.0, zoom='fit',
           size=0.25, gap=1.5, rotation=None, flip=None, cmap='viridis', vmin=None, vmax=None,
           pure_hues=False, crop=True, style=None, **_ignored):
    """Render a still image; returns a PIL.Image (and writes it to `out` if given).

    res, spp     output size (square, before cropping) and samples per pixel
    elevation    camera height, degrees above the floor
    zoom         'fit' (points and floor fill the frame) or a number (1.35 = viewer default)
    size         bead radius, relative to the mean spacing of the points
    gap          floor distance below the lowest bead, in bead radii
    rotation     3x3 matrix turning the points (see `plane_rotation`, or a viewer's `.view`)
    flip         three booleans, turning each data axis over before `rotation`
    pure_hues    recolour each point to its HSL hue at full saturation, lightness 0.5
    crop         trim the white page to the drawing
    style        a `Style` (lighting, floor, materials, camera azimuth and field of view)
    """
    xyz, rgb = _prepare(points, colours, cmap, vmin, vmax, pure_hues)
    fit = isinstance(zoom, str)
    scene = FloorScene(xyz, rgb, res, rotation=rotation, flip=flip, elevation=elevation,
                       zoom=1.35 if fit else zoom, size=size, gap=gap, style=style)
    if fit:
        scene.fit()
    img = Image.fromarray(scene.still(spp, crop=crop))
    del scene
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
    return img


def _session(points, colours, cmap, vmin, vmax, pure_hues, res, target_spp, save_dir, name,
             **view):
    from .session import ViewerSession
    xyz, rgb = _prepare(points, colours, cmap, vmin, vmax, pure_hues)
    return ViewerSession(xyz, rgb, res=res, target_spp=target_spp, save_dir=save_dir, name=name,
                         **view).wait_ready()


def notebook(points, colours, *, res=720, target_spp=256, elevation=30.0, zoom=1.35,
             size=0.25, gap=1.5, rotation=None, flip=None, cmap='viridis', vmin=None, vmax=None,
             pure_hues=False, style=None, save_dir='raytrace_saves', save_res=2400,
             save_spp=1024, name='view', max_fps=20):
    """Interactive viewer in a notebook cell (returns the widget; display it or leave it last).

    res, target_spp     the viewer's image. There is one resolution: after every change the
                        picture is rebuilt at `res` and refined towards `target_spp` samples, so
                        `res` sets both what you drag and what you settle on, and `target_spp`
                        only how long it keeps improving once you stop.
    save_res, save_spp  what "Save PNG" / `.save()` writes, which is rendered separately.
    max_fps             frames pushed to the notebook per second while you drag.

    `widget.save()` renders the current view at print quality; `widget.view` holds its
    settings; `widget.close()` frees the GPU."""
    from .widget import NotebookViewer
    s = _session(points, colours, cmap, vmin, vmax, pure_hues, res, target_spp, save_dir, name,
                 elevation=elevation, zoom=zoom, size=size, gap=gap, rotation=rotation,
                 flip=flip, style=style, save_res=save_res, save_spp=save_spp)
    return NotebookViewer(s, max_fps=max_fps)


def browser(points, colours, *, host='127.0.0.1', port=17100, block=False, res=960,
            target_spp=512, elevation=30.0, zoom=1.35, size=0.25, gap=1.5, rotation=None,
            flip=None, cmap='viridis', vmin=None, vmax=None, pure_hues=False, style=None,
            save_dir='raytrace_saves', save_res=2400, save_spp=1024, name='view',
            title='Raytraced scatter'):
    """Interactive viewer in a web browser at http://host:port/.

    `res` / `target_spp` size the viewer's image and `save_res` / `save_spp` the saved PNG,
    as in `notebook` above.

    block=False (default) serves from a background thread and returns the viewer: `.url`,
    `.save()`, `.view`, `.stop()`. block=True serves until interrupted (for scripts).
    On a remote machine, forward the port (e.g. VS Code's Ports panel, or ssh -L)."""
    from .server import BrowserViewer
    s = _session(points, colours, cmap, vmin, vmax, pure_hues, res, target_spp, save_dir, name,
                 elevation=elevation, zoom=zoom, size=size, gap=gap, rotation=rotation,
                 flip=flip, style=style, save_res=save_res, save_spp=save_spp)
    return BrowserViewer(s, host=host, port=port, title=title).start(block=block)
