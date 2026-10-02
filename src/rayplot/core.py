"""The renderer: beads over a white square floor, fully path traced (Mitsuba 3 + OptiX).

Every point is an analytic sphere in one Mitsuba `ellipsoids` shape (one BVH, no triangles).
The floor is a real white diffuse square; its shadows and the light it bounces back onto the
beads come out of the path tracer. After rendering: OptiX denoising, the beads' colour removed
from the floor (brightness kept), one exposure that brings the lit floor to near white, a white
page where the camera sees nothing, and the floor's outline drawn behind the beads.

Coordinates: the points are centred and scaled so their bounding box fits a unit ball; the
third coordinate is "up". The floor is re-fitted under the (turned) points every time: `gap`
bead radii below the lowest bead, covering their footprint. The camera looks at the points
from a fixed azimuth, at `elevation` degrees, and never orbits; `rotation` turns the points.
"""
import functools
import gc
import math
import threading
from dataclasses import dataclass

import numpy as np
import torch
import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')            # before PrimColour: mi.Texture depends on the variant

LUM = (0.2126, 0.7152, 0.0722)
# Smallest bead size a viewer offers. It also sets the world unit (see FloorScene), and stills
# use the same one, so a still and a viewer save of the same view are the same render.
SIZE_MIN = 0.08

# Mitsuba is not safe to drive from two threads at once: building a scene while another thread
# renders fails ("Endpoint::pdf_direction(): not implemented") and can crash the process. Every
# Mitsuba call goes through this lock, held for one step at a time (a scene, one pass of
# samples, one denoise), so two viewers, or a still and a viewer, take turns rather than wait
# for each other to finish.
MITSUBA = threading.RLock()


def serial(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with MITSUBA:
            return fn(*args, **kwargs)
    return run
# A turn "in a plane" is about the axis normal to it (z is up); the sign makes positive angles
# carry the plane's first axis towards its second (x -> y, y -> z, x -> z).
PLANES = {'xy': (np.array([0, 0, 1.0]), 1), 'yz': (np.array([1.0, 0, 0]), 1),
          'xz': (np.array([0, 1.0, 0]), -1)}


@dataclass
class Style:
    """Look of the figure. Lengths are in units of the normalised cloud (radius ~1)."""
    fov: float = 28.0                 # degrees
    azimuth: float = 45.0             # camera direction; 45 shows the floor as a diamond
    light_half: float = 0.18          # soft box half-size: smaller = crisper shadows
    light_height: float = 9.0         # above the floor
    light_offset: tuple = (0.35, -0.25)   # soft box position relative to the floor's centre
    key: float = 400.0                # soft box radiance
    sky: float = 0.2                  # uniform ambient light (lifts the shadows)
    floor_albedo: float = 0.85
    floor_margin: float = 1.12        # floor half-width / half the points' footprint
    anchor: str = 'points'            # what the camera is pinned to:
                                      # 'points' -- it looks at the cloud's centre, and the floor
                                      #   is re-fitted under whatever is there, so the floor is
                                      #   drawn a different size and in a different place in
                                      #   every figure (fine alone, useless for comparing two);
                                      # 'floor'  -- it looks at the floor, from a distance in
                                      #   floor half-widths, so the floor's projection is
                                      #   identical in every figure and only the cloud moves.
    anchor_height: float = 0.30       # 'floor': camera target above the floor, in half-widths
    anchor_dist: float = 1.78         # 'floor': camera distance, in half-widths (times 1/zoom).
                                      # Both are the median of what 'points' chooses over a
                                      # spread of real clouds, so the default view barely moves.
    roughness: float = 0.25           # beads
    specular: float = 0.6
    coloured_floor: bool = False      # keep the colour the beads bounce onto the floor
    ink: float = 0.02                 # outline colour (linear grey)


class PrimColour(mi.Texture):
    """Colour stored per primitive on the shape. (The built-in `mesh_attribute` texture
    returns black on `ellipsoids`, though the shape's own attribute lookup works.)"""
    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.name = props.get('name', 'color')

    def eval(self, si, active=True):
        return si.shape.eval_attribute_3(self.name, si, active)

    def eval_1(self, si, active=True):
        return mi.luminance(self.eval(si, active))

    def mean(self):
        return 0.5

    def is_spatially_varying(self):
        return True

    def traverse(self, cb):
        pass


mi.register_texture('prim_colour', PrimColour)


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def sphere_cloud(centres, rgb, radius, bsdf):
    """Analytic spheres in one shape; `rgb` is sRGB, stored linear in the `color` attribute.

    The ellipsoid intersection has an absolute tolerance: keep `radius` around 1 or more
    (build scenes in units of the bead radius), or small beads render as translucent facets."""
    n = len(centres)
    T = lambda x: mi.TensorXf(np.ascontiguousarray(x, dtype=np.float32))
    return mi.load_dict({
        'type': 'ellipsoids', 'extent': 1.0,     # extent scales the radii; 1 = as given
        'centers': T(centres), 'scales': T(np.full((n, 3), radius)),
        'quaternions': T(np.tile([0, 0, 0, 1], (n, 1))), 'opacities': T(np.ones((n, 1))),
        'color': T(srgb_to_linear(rgb)), 'bsdf': bsdf})


def wrap(deg):
    return (deg + 180) % 360 - 180


def rot(axis, deg):
    """3x3 rotation about a unit axis."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def plane_rotation(xy=0.0, yz=0.0, xz=0.0):
    """Turns in the xy, then yz, then xz plane, as one matrix (for `rotation=`)."""
    R = np.eye(3)
    for plane, deg in (('xy', xy), ('yz', yz), ('xz', xz)):
        axis, sign = PLANES[plane]
        R = rot(axis, sign * deg) @ R
    return R


def project(to_world, fov, res, pts):
    """Pixel coordinates of world points through a Mitsuba perspective camera (fov on x)."""
    m = np.array(to_world.inverse().matrix, dtype=np.float64)
    ph = np.c_[pts, np.ones(len(pts))] @ m.T
    t = np.tan(np.radians(fov) / 2)
    x, y, z = ph[:, 0], ph[:, 1], ph[:, 2]           # camera x points left, y up, z forward
    return np.c_[(0.5 - 0.5 * x / (z * t)) * res, (0.5 - 0.5 * y / (z * t)) * res]


class FloorScene:
    """One renderable view. Change `state` / `R`, call `apply()`, then render.

    state: elevation (deg), zoom, size (bead radius, relative to the mean point spacing),
    gap (floor below the lowest bead, in bead radii). R: 3x3 rotation of the points.
    `size_min` sets the world unit (a bead of that size has radius 1); sizes far below it
    lose precision. Viewers and stills both use min(SIZE_MIN, size), so a still of a viewer's
    view is the same render as the viewer's save.
    """
    STATE = ('elevation', 'zoom', 'size', 'gap')

    @serial
    def __init__(self, xyz, rgb, res=960, *, rotation=None, flip=None, elevation=30.0,
                 zoom=1.35, size=0.25, gap=1.5, size_min=None, style=None):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 1:
            raise ValueError(f'points must be (n, 3), got {xyz.shape}')
        self.xyz_in, self.rgb = xyz, np.asarray(rgb, dtype=np.float64)
        lo, hi = np.percentile(xyz, [0.2, 99.8], axis=0)
        scale = np.linalg.norm(hi - lo) / 2 or 1.0
        self.xyz = (xyz - (lo + hi) / 2) / scale              # bbox in the unit ball
        self.n, self.res = len(xyz), int(res)
        self.style = style or Style()
        self.r1 = 0.3 * (4 / 3 * np.pi * 0.6 / self.n) ** (1 / 3)   # bead radius at size 1
        self.size_min = min(size_min or size, size)
        self.k = 1 / (self.size_min * self.r1)                  # world units per unit
        self.state = dict(elevation=float(elevation), zoom=float(zoom), size=float(size),
                          gap=float(gap))
        self.R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        # Per-axis sign, applied to the points before `R`. A flip is a reflection of the data,
        # so it belongs inside the turn: `matrix` composes the two into the one rotation that
        # `render(rotation=...)` reproduces the view from.
        self.sign = np.ones(3) if flip is None else np.where(np.asarray(flip, bool), -1.0, 1.0)
        self.exposure = None

        st = self.style
        bead = {'type': 'principled', 'base_color': {'type': 'prim_colour', 'name': 'color'},
                'roughness': st.roughness, 'specular': st.specular}
        self.scene = mi.load_dict(self._scene_dict(bead, full=True))
        self.cover = mi.load_dict(self._scene_dict({'type': 'diffuse'}, full=False))
        self.p, self.pc = mi.traverse(self.scene), mi.traverse(self.cover)
        self.denoiser = mi.OptixDenoiser(input_size=[self.res, self.res], albedo=True,
                                         normals=True, temporal=False)
        c = torch.arange(self.res, device='cuda') + 0.5
        yy, xx = torch.meshgrid(c, c, indexing='ij')
        self.pix = torch.stack([xx, yy], -1)
        self.lum = torch.tensor(LUM, device='cuda')
        self.apply()

    # ---- scene -------------------------------------------------------------------------
    def _scene_dict(self, bead, full):
        k, st, res = self.k, self.style, self.res
        d = {'type': 'scene',
             # The scene is built in units of the bead radius, so its size in world units is
             # whatever `k` says -- and `k` grows with the point count and with a small
             # `size_min` (a viewer's is 0.08). Mitsuba's default far_clip of
             # 1e4 is an absolute number, so it silently cuts the far half of a large scene away
             # (the floor first, being furthest). far_clip therefore scales with k. near_clip is
             # left at Mitsuba's default: it sets where each ray starts, so changing it moves
             # every sample slightly and would renumber the noise in renders that were fine.
             'sensor': {'type': 'perspective', 'fov': st.fov, 'far_clip': 1e3 * k,
                        'to_world': mi.ScalarTransform4f().look_at(
                            origin=[0, -5 * k, 0], target=[0, 0, 0], up=[0, 0, 1]),
                        'film': {'type': 'hdrfilm', 'width': res, 'height': res,
                                 'pixel_format': 'rgba', 'rfilter': {'type': 'gaussian'}}},
             'points': sphere_cloud(self.xyz * k, self.rgb, 1.0, bead)}
        if not full:                                           # coverage pass: beads only
            d['integrator'] = {'type': 'path', 'max_depth': 1}
            return d
        d['integrator'] = {'type': 'aov', 'aovs': 'albedo:albedo,nn:sh_normal',
                           'img': {'type': 'path', 'max_depth': 12, 'hide_emitters': True}}
        d['floor'] = {'type': 'rectangle', 'to_world': mi.ScalarTransform4f(),
                      'bsdf': {'type': 'diffuse',
                               'reflectance': {'type': 'rgb', 'value': st.floor_albedo}}}
        # The light never moves (moving an emitter recompiles the renderer, ~1 s): the
        # scene is shifted instead, so the floor's centre is always at the origin.
        ox, oy = st.light_offset
        key = mi.ScalarTransform4f().translate([ox * k, oy * k, st.light_height * k]) \
            .scale([st.light_half * k, st.light_half * k, 1]) \
            @ mi.ScalarTransform4f().rotate([1, 0, 0], 180)
        d['key'] = {'type': 'rectangle', 'to_world': key,
                    'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': st.key}}}
        d['sky'] = {'type': 'constant', 'radiance': {'type': 'rgb', 'value': st.sky}}
        return d

    def layout(self):
        """World-space beads, floor and camera for the current state."""
        s, k, st = self.state, self.k, self.style
        xyz = (self.xyz * self.sign) @ self.R.T            # turn the points, not the floor
        lo, hi = np.percentile(xyz[:, :2], [0.2, 99.8], axis=0)
        c = (lo + hi) / 2                                  # floor follows the footprint
        half = st.floor_margin * max((hi - lo).max() / 2, 0.05)
        az, el = math.radians(st.azimuth), math.radians(s['elevation'])
        dirn = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        if st.anchor == 'floor':
            # Target and distance are both multiples of the floor's half-width, so the whole
            # camera-and-floor arrangement is the same shape whatever the cloud: the floor lands
            # in the same place, at the same size, and only the points move.
            target = np.array([0.0, 0.0, st.anchor_height * half])
            dist = st.anchor_dist * half / s['zoom'] / math.sin(math.radians(st.fov / 2))
        else:
            # Framing about the points' centre, independent of their turn, so the view does not
            # jump when the floor is re-fitted.
            dist = 1.5 / s['zoom'] / math.sin(math.radians(st.fov / 2))
        r = s['size'] * self.r1                            # bead radius, unit coords
        fz = xyz[:, 2].min() - r * (1 + s['gap'])
        off = np.array([c[0], c[1], fz])                   # floor centre -> world origin
        corners = np.array([[-half, -half, 0], [half, -half, 0],
                            [half, half, 0], [-half, half, 0]]) * k
        floor = mi.ScalarTransform4f().scale([half * k, half * k, 1])
        if st.anchor != 'floor':
            target = np.array([0, 0, -0.3]) - off
        eye = (target + dirn * dist) * k
        cam = mi.ScalarTransform4f().look_at(origin=eye.tolist(), target=(target * k).tolist(),
                                             up=[0, 0, 1])
        data = np.zeros((self.n, 10), np.float32)
        data[:, :3] = (xyz - off) * k
        data[:, 3:6] = r * k                               # bead radius, world units
        data[:, 9] = 1                                     # identity quaternion (x, y, z, w)
        return dict(data=data, floor=floor, cam=cam, corners=corners, dirn=dirn,
                    radius=r * k)

    @serial
    def apply(self):
        """Push the current state to the GPU and restart the running image."""
        L = self.layout()
        for p in (self.p, self.pc):
            p['points.data'] = mi.Float(L['data'].ravel())
            p['sensor.to_world'] = L['cam']
        self.p['floor.to_world'] = L['floor']
        self.p.update(); self.pc.update()
        self.L = L
        self.acc, self.spp = None, 0
        self.bead_a = self._coverage(8)
        self.line = self._outline()

    @serial
    def fit(self, fill=0.92, iters=4):
        """Set `zoom` so the points and the floor fill `fill` of the frame."""
        for _ in range(iters):
            L = self.layout()
            pts = L['data'][:, :3]
            if len(pts) > 20000:
                pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
            uv = project(L['cam'], self.style.fov, 2.0, np.r_[pts, L['corners']])
            ext = np.abs(uv - 1).max() * (1 + 0.02)
            self.state['zoom'] *= fill / ext
        self.apply()

    @serial
    def _coverage(self, spp, seed=0):
        return torch.clamp(mi.render(self.cover, spp=spp, seed=seed).torch()[..., 3:4], 0, 1)

    def _outline(self):
        """Antialiased coverage of the floor's outline, (res, res, 1)."""
        L, res = self.L, self.res
        uv = torch.tensor(project(L['cam'], self.style.fov, res, L['corners']), device='cuda',
                          dtype=torch.float32)
        d = torch.full(self.pix.shape[:2], 1e9, device='cuda')
        for i in range(4):
            a, b = uv[i], uv[(i + 1) % 4]
            ab = b - a
            t = torch.clamp(((self.pix - a) @ ab) / (ab @ ab), 0, 1)
            d = torch.minimum(d, torch.linalg.norm(self.pix - (a + t[..., None] * ab), dim=-1))
        w = max(1.0, res / 900)
        return torch.clamp(w / 2 + 0.5 - d, 0, 1)[..., None]

    # ---- rendering ---------------------------------------------------------------------
    @serial
    def step(self, target):
        """Add samples to the running image (1, 1, 2, 4, ... 64 at a time); False when done."""
        if self.spp >= target:
            return False
        chunk = int(min(64, max(1, self.spp), target - self.spp))
        img = mi.render(self.scene, spp=chunk, seed=self.spp + 7919).torch()
        self.acc = img * chunk if self.acc is None else self.acc + img * chunk
        self.spp += chunk
        return True

    def _denoise(self, img):
        T = lambda x: mi.TensorXf(x.contiguous())
        return self.denoiser(T(img[..., :3]), albedo=T(img[..., 4:7]), normals=T(img[..., 7:10]),
                             to_sensor=self.L['cam'].inverse()).torch()

    @serial
    def calibrate(self, img=None):
        """Exposure that maps the lit floor (90th percentile; shadows excluded) to 0.97."""
        if img is None:
            img = sum(mi.render(self.scene, spp=32, seed=100 + i).torch() for i in range(2)) / 2
        rgb = self._denoise(img)
        floor = (img[..., 9] > 0.999) & (img[..., 3] > 0.999)
        lum = (rgb @ self.lum)[floor]
        if lum.numel() < 100:                 # floor out of view: keep what we had, if anything
            self.exposure = self.exposure or 1.0   # never leave it None: develop() multiplies by it
            return self.exposure
        # A fixed seed: the same view always gets the same exposure.
        g = torch.Generator(device='cuda').manual_seed(0)
        idx = torch.randperm(lum.numel(), device='cuda', generator=g)[:1_000_000]
        self.exposure = float(0.97 / torch.quantile(lum[idx].float(), 0.9))
        return self.exposure

    @serial
    def develop(self, img=None, bead_a=None):
        """Finished image, uint8 (res, res, 3), from the running (or a given) mean image."""
        img = self.acc / self.spp if img is None else img
        bead_a = self.bead_a if bead_a is None else bead_a
        if self.exposure is None:
            self.calibrate(img)
        rgb = self._denoise(img)
        if not self.style.coloured_floor:
            rgb = bead_a * rgb + (1 - bead_a) * (rgb @ self.lum)[..., None]
        out = torch.clamp(rgb * self.exposure, min=0) + (1 - img[..., 3:4])   # white page
        line = self.line * (1 - bead_a)                        # outline behind the beads
        out = torch.clamp(out * (1 - line) + self.style.ink * line, 0, 1)
        srgb = torch.where(out <= 0.0031308, 12.92 * out, 1.055 * out ** (1 / 2.4) - 0.055)
        return (srgb * 255 + 0.5).to(torch.uint8).cpu().numpy()

    def still(self, spp=1024, crop=True, calibrate=True):
        """Print-quality image of the current view (uint8), optionally cropped to the drawing.
        `calibrate=False` keeps the current exposure (so stills match an interactive view)."""
        acc, done, chunk = None, 0, min(128, spp)     # one big pass would not fit in memory
        while done < spp:
            with MITSUBA:
                img = mi.render(self.scene, spp=chunk, seed=10_000 + done).torch()
            acc = img * chunk if acc is None else acc + img * chunk
            done += chunk
        img = acc / done
        if calibrate or self.exposure is None:
            self.calibrate(img)
        out = self.develop(img, self._coverage(32, seed=1))
        if crop:
            ys, xs = np.nonzero((out < 254).any(-1))
            if len(ys):
                m = self.res // 40
                out = out[max(ys.min() - m, 0):ys.max() + m + 1,
                          max(xs.min() - m, 0):xs.max() + m + 1]
        return out

    @serial
    def release(self):
        """Free this scene's GPU memory now, rather than whenever Python gets round to it."""
        for name in ('scene', 'cover', 'p', 'pc', 'denoiser', 'pix', 'lum', 'acc', 'bead_a',
                     'line'):
            self.__dict__.pop(name, None)
        gc.collect()
        dr.flush_malloc_cache()
        torch.cuda.empty_cache()

    @property
    def matrix(self):
        """The turn and the axis flips as one matrix: `xyz @ matrix.T` is what is drawn."""
        return self.R @ np.diag(self.sign)

    def turn(self, plane=None, deg=0.0, axis=None):
        """Turn the points in one plane ('xy', 'yz', 'xz') or about an arbitrary axis."""
        if plane is not None:
            axis, sign = PLANES[plane]
            deg = sign * deg
        self.R = rot(axis, deg) @ self.R

    def sibling(self, res):
        """The same view at another resolution (same world unit, rotation and exposure)."""
        s = FloorScene(self.xyz_in, self.rgb, res, rotation=self.R, flip=self.sign < 0,
                       size_min=self.size_min, style=self.style, **self.state)
        s.exposure = self.exposure
        return s
