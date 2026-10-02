# raytraced_plots

Path-traced 3-D scatter plots: glossy beads floating over a white square floor that catches their shadows. Rendering runs on the GPU (Mitsuba 3 with OptiX).

## `raytrace3d` (the module)

It is installed in editable mode into `/venv/main`, so `import raytrace3d as rt` works from any directory. To install it elsewhere, run `uv pip install -e /workspace/raytraced_plots`.

```python
import raytrace3d as rt

img = rt.render(points, colours, out='fig.png')   # still image (PIL.Image)
w = rt.notebook(points, colours)                  # interactive, in a notebook cell
v = rt.browser(points, colours, port=17100)       # interactive, at http://127.0.0.1:17100/
```

- **`points`:** an `(n, 3)` array. The third coordinate points up. Scale and offset don't matter.
- **`colours`:** any of the following:
  - one colour for every point;
  - `n` hex / `rgb(...)` / `hsl(...)` / matplotlib-name strings;
  - an `(n, 3)` array of floats in 0–1 or integers in 0–255;
  - `n` numbers mapped through `cmap=`.
- **`rt.points_from_notebook(path, 'layer 28')`:** reads the points and colours of a 3-D plotly figure saved in a notebook's outputs.
- **`flip`:** three booleans, turning each data axis over before `rotation`. A component's sign is
  arbitrary, so this is how two plots are brought into the same orientation.
- **Interactive viewers:**
  - Drag the image to turn the points freely.
  - The xy / yz / xz sliders turn them within one plane only.
  - The **flip tickboxes** under those sliders turn each axis over; `.flip` gives them back, and
    they also ride inside `.view['rotation']`, so `rt.render(points, colours, **v.view)` still
    reproduces exactly what is on screen.
  - Other sliders set camera height, zoom, bead size and the gap to the floor.
  - **Save PNG** (or `.save()`) writes a separate, slower render to `save_dir`, sized by
    `save_res` / `save_spp` (2400 px, 1024 spp by default).
  - **Resolution:** a viewer has one, `res`, refined towards `target_spp` samples after every
    change — so `res` is both what you drag and what you settle on, and `target_spp` only how
    long it keeps improving once you stop. There is no separate preview size: the first frame
    after a change costs 29-65 ms over 480-1440 px (36k points), so dropping the resolution
    while dragging would buy almost nothing. `max_fps` (notebook) caps what is pushed to the
    front end.
  - `.view` returns the current settings, so `rt.render(points, colours, **w.view)` reproduces the view.
  - `.close()` (notebook) or `.stop()` (browser) frees the GPU.
- **`rt.Style(...)`:** lighting, floor, materials and camera. `rt.plane_rotation(xy=, yz=, xz=)` builds a rotation matrix for `rotation=`.
- **`rt.Style(anchor='floor')`:** pins the camera to the floor rather than to the cloud. By default
  the camera follows the points and the floor is re-fitted under them, so the floor is drawn at a
  different size, in a different place, in every figure — fine for one picture, useless for
  comparing two. With `anchor='floor'` the floor's projection depends on `zoom` alone: its corners
  land in the same pixels whatever the cloud, and only the points move. Over 108 clouds the floor's
  corners moved 167 px across an 800-px frame by default, and 0.00 px anchored.

See `raytrace3d_demo.ipynb`.

## Other files

- **`render_3d_raytraced.py`, `render_3d_floor.py`, `render_3d_floor_physical.py`:** the original stand-alone scripts that made the `scores_3d_*.png` images.
- **`interactive/server.py`:** the `raytrace_viewer` service, which shows layer 28 in the browser viewer.
  - Start it with `supervisorctl start raytrace_viewer`; it doesn't start at boot.
  - Through the instance's proxy it is on port 10100, which needs the instance token. On the machine itself it is at `127.0.0.1:17100`.
  - Saves go to `interactive_saves/`.
- **`scratch/`:** test renders.
