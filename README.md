# rayplot

Path-traced 3-D scatter plots: glossy beads floating over a white square floor that catches
their shadows. Rendering runs on an NVIDIA GPU (Mitsuba 3 with OptiX), as a still image or in
an interactive viewer in a notebook or a web browser.

## Requirements and install

- An NVIDIA GPU with a driver that supports OptiX (535 or newer), and torch built with CUDA.
- Python 3.10+, Mitsuba 3.9+.

```bash
pip install -e ".[all]"        # from a checkout; [notebook] or [browser] for just one viewer
```

pip cannot choose a CUDA build of torch for you; install one first if needed
(see <https://pytorch.org/get-started/locally/>).

## Quick start

```python
import rayplot as rp

img = rp.render(points, colours, out='fig.png')   # still image (PIL.Image)
w = rp.notebook(points, colours)                  # interactive, in a notebook cell
v = rp.browser(points, colours, port=17100)       # interactive, at http://127.0.0.1:17100/
```

### Points and colours

- **`points`:** an `(n, 3)` array, or wider (only the first three columns are used). The third
  coordinate points up. Scale and offset don't matter. Rows with NaN or infinite coordinates
  are dropped, with a warning.
- **`colours`:** any of the following:
  - one colour for every point: a string, or a tuple `(r, g, b)`;
  - `n` hex / `rgb(...)` / `hsl(...)` / matplotlib-name strings;
  - an `(n, 3)` or `(n, 4)` array (alpha is ignored);
  - `n` numbers, mapped through `cmap=` (default `viridis`) between `vmin` and `vmax`.

  RGB numbers are read as 0–1, unless the array is `uint8` or a value exceeds 1.5, when they
  are read as 0–255. A list of 3 numbers for exactly 3 points is 3 values for `cmap`; pass a
  tuple to mean one colour.

### Still images: `rp.render`

```python
rp.render(points, colours, out=None, *, res=2400, spp=1024, elevation=30.0, zoom='fit',
          size=0.25, gap=1.5, rotation=None, flip=None, exposure=None, cmap='viridis',
          vmin=None, vmax=None, crop=True, style=None)
```

| argument | meaning |
|---|---|
| `res`, `spp` | output size (square, before cropping) and samples per pixel |
| `elevation` | camera height, degrees above the floor |
| `zoom` | `'fit'` (points and floor fill the frame) or a number (1.35 is the viewers' default) |
| `size` | bead radius, relative to the mean spacing of the points |
| `gap` | floor distance below the lowest bead, in bead radii |
| `rotation` | 3×3 matrix turning the points: `rp.plane_rotation(xy=, yz=, xz=)`, or a viewer's `.view` |
| `flip` | three booleans, turning each data axis over before `rotation` |
| `exposure` | `None` measures it from this image; a number (as in a viewer's `.view`) keeps it |
| `crop` | trim the white page to the drawing |
| `style` | an `rp.Style` (below) |

Unknown keyword arguments are an error.

### Interactive viewers: `rp.notebook` and `rp.browser`

Both take the same view arguments as `render` (except `res`/`spp`, `zoom='fit'` and `crop`), plus:

| argument | meaning |
|---|---|
| `res`, `target_spp` | the viewer's image: rebuilt at `res` after every change, then refined towards `target_spp` samples (notebook 720 / 256, browser 960 / 512) |
| `save_dir`, `save_res`, `save_spp`, `name` | where and how **Save PNG** renders (`rayplot_saves`, 2400 px, 1024 spp) |
| `title` | heading of the control panel (and, in the browser, the page title) |
| `max_fps` | notebook only: frames per second pushed to the front end while dragging |
| `host`, `port`, `block` | browser only: where to serve, and whether to serve until Ctrl-C |

Controls:

- Drag the image to turn the points freely.
- The xy / yz / xz sliders turn them within one plane only.
- The x / y / z tickboxes turn each data axis over. **Reset turn** clears the turns (not the flips).
- Other sliders set camera height, zoom, bead size and the gap to the floor.
- **Save PNG** (or `.save()`) renders the current view at `save_res` / `save_spp` into
  `save_dir`, with a JSON file beside it recording the whole view.

Both viewer objects have:

- `.view`: the current settings (`rotation` with the flips folded in, `elevation`, `zoom`,
  `size`, `gap`, `exposure`, `style`). `rp.render(points, colours, **v.view)` reproduces what
  is on screen; with `res=save_res, spp=save_spp` it is the same render as **Save PNG**.
- `.rotation`, `.flip`, `.save()`.
- `.close()`: stops rendering and frees the GPU. `.view`, `.rotation` and `.flip` still answer
  afterwards. (The browser viewer also has `.url`, and `.stop()` as another name for `.close()`.)

The browser viewer has no authentication. It serves on `127.0.0.1` by default; on a remote
machine, forward the port (VS Code's Ports panel, or `ssh -L`). Binding any other host warns.
A `?token=` on the page URL is passed on to the websocket, so it works behind a token proxy.

Several viewers, and stills, can run at once in one process; they take turns on the GPU.

### Saved views: `rp.load_view`

```python
rp.render(points, colours, **rp.load_view('rayplot_saves/view_..._gap1.5.png'))
```

reads the JSON beside a saved PNG and returns the view as keyword arguments.

### Style

`rp.Style(...)` sets lighting, floor, materials and camera:

| group | fields (defaults) |
|---|---|
| camera | `fov` 28, `azimuth` 45 (shows the floor as a diamond) |
| light | `light_half` 0.18, `light_height` 9, `light_offset` (0.35, −0.25), `key` 400, `sky` 0.2 |
| floor | `floor_albedo` 0.85, `floor_margin` 1.12, `coloured_floor` False, `ink` 0.02 |
| beads | `roughness` 0.25, `specular` 0.6 |
| framing | `anchor` `'points'`, `anchor_height` 0.30, `anchor_dist` 1.78 |

By default the camera follows the points and the floor is re-fitted under them, so the floor is
drawn at a different size, in a different place, in every figure: fine for one picture, useless
for comparing two. With `anchor='floor'` the floor's projection depends on `zoom` alone: its
corners land in the same pixels whatever the cloud, and only the points move.

### Plotly figures saved in notebooks

```python
points, colours = rp.points_from_notebook('report.ipynb', 'layer 28')   # by trace name or index
rp.notebook_traces('report.ipynb')    # every scatter3d trace: name, xyz, colour, values
```

Reads 3-D scatter traces from a notebook's saved outputs (plotly JSON outputs, and every
`Plotly.newPlot` call in HTML outputs), without re-running anything. Colours come back as the
figure shows them: the figure's own colour scale is applied when it is a list of stops or one of
viridis / plasma / inferno / magma / cividis; otherwise the numbers come back for `cmap`.

## Not supported yet

- Apple Silicon. Mitsuba 3.9 has a Metal backend, but rayplot uses CUDA-only parts (the OptiX
  denoiser, `cuda` torch tensors).
- Importing rayplot without an NVIDIA GPU: the import selects Mitsuba's CUDA variant.

## Development

```bash
pip install -e ".[dev]"
pytest                   # needs an NVIDIA GPU: importing rayplot selects Mitsuba's CUDA variant
```

---

**AI declaration:** The code in this package was written with the assistance of generative AI.
