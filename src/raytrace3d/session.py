"""The interactive render loop, shared by the notebook widget and the browser server.

A `ViewerSession` owns a `FloorScene` on its own thread (every Mitsuba call happens there).
Front ends `send()` messages and `subscribe()` to frames (JPEG bytes + a meta dict).

Messages (JSON-like dicts):
  {'type': 'rotate', 'dx', 'dy'}          free turn from a mouse drag (pixels * 0.4 deg)
  {'type': 'plane', 'plane', 'angle'}     turn within one plane to the slider's angle
  {'type': 'set', elevation/zoom/size/gap}
  {'type': 'flip', 'axis': 0|1|2, 'on': bool}   turn one data axis over (a reflection)
  {'type': 'reset'}                       undo all turns
  {'type': 'save'}                        print-quality PNG into save_dir
  {'type': 'hello'}                       re-send the latest frame (a new front end)
"""
import io
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .core import FloorScene, PLANES, rot, wrap

SIZE_MIN = 0.08                    # smallest bead size a viewer offers; sets the world unit


class ViewerSession:
    def __init__(self, xyz, rgb, *, res=960, target_spp=512, save_dir='raytrace_saves',
                 save_res=2400, save_spp=1024, style=None, name='view', **view):
        self.args = dict(xyz=xyz, rgb=rgb, res=res, style=style, **view)
        self.target, self.save_res, self.save_spp = target_spp, save_res, save_spp
        self.save_dir, self.name = Path(save_dir), name
        self.lock = threading.Lock()
        self.pending, self.listeners = [], []
        self.wake, self.ready, self.stopped = threading.Event(), threading.Event(), False
        self.latest, self.error, self.scene = None, None, None
        self.angles = dict(xy=0.0, yz=0.0, xz=0.0)
        self._waiters = []
        self.thread = threading.Thread(target=self._run, daemon=True, name='raytrace3d')
        self.thread.start()

    # ---- API for front ends and callers ------------------------------------------------
    def send(self, msg):
        with self.lock:
            self.pending.append(dict(msg))
        self.wake.set()

    def subscribe(self, fn):
        """fn(jpeg_bytes, meta) is called on the render thread for every new frame."""
        self.listeners.append(fn)
        if self.latest:
            fn(*self.latest)

    def unsubscribe(self, fn):
        if fn in self.listeners:
            self.listeners.remove(fn)

    def wait_ready(self, timeout=120):
        self.ready.wait(timeout)
        if self.error:
            raise RuntimeError(self.error)
        return self

    def save(self, timeout=600):
        """Render the current view at print quality; returns the PNG's path."""
        done = threading.Event()
        box = {}
        with self.lock:
            self._waiters.append((done, box))
        self.send({'type': 'save'})
        done.wait(timeout)
        if 'error' in box:
            raise RuntimeError(box['error'])
        return box.get('path')

    @property
    def rotation(self):
        """The turn and the axis flips, as the one matrix that reproduces the view."""
        return None if self.scene is None else self.scene.matrix

    @property
    def flip(self):
        return None if self.scene is None else tuple(bool(x) for x in (self.scene.sign < 0))

    @property
    def view(self):
        """Current settings, as keyword arguments for `raytrace3d.render`.

        The flips ride inside `rotation`, so this stays a complete description of the view
        without a `flip=` of its own.
        """
        s = self.scene
        return {} if s is None else dict(rotation=s.matrix, **s.state)

    def stop(self):
        self.stopped = True
        self.wake.set()

    # ---- render thread -----------------------------------------------------------------
    def _run(self):
        try:
            a = self.args
            size = a.get('size', 0.25)
            self.scene = FloorScene(a['xyz'], a['rgb'], a['res'], style=a['style'],
                                    size_min=min(SIZE_MIN, size),
                                    **{k: a[k] for k in ('rotation', 'flip', *FloorScene.STATE)
                                       if k in a})
            self.scene.calibrate()
        except Exception:
            self.error = traceback.format_exc()
            self.ready.set()
            return
        self.ready.set()
        v, status = self.scene, ''
        while not self.stopped:
            with self.lock:
                msgs, self.pending = self.pending, []
            changed = False
            for m in msgs:
                t = m.get('type')
                if t == 'rotate':
                    turn = float(m.get('dx', 0)) * 0.4            # sideways = xy plane
                    v.turn('xy', turn)
                    self.angles['xy'] = wrap(self.angles['xy'] + turn)
                    right = np.cross(v.L['dirn'], [0, 0, 1])      # screen-horizontal axis
                    v.turn(axis=right, deg=-float(m.get('dy', 0)) * 0.4)
                    changed = True
                elif t == 'plane' and m.get('plane') in PLANES:
                    p = m['plane']
                    angle = float(m['angle'])
                    v.turn(p, angle - self.angles[p])
                    self.angles[p] = wrap(angle)
                    changed = True
                elif t == 'set':
                    for key in FloorScene.STATE:
                        if key in m:
                            val = float(m[key])
                            if key == 'size':
                                val = max(val, v.size_min)
                            v.state[key] = val
                    changed = True
                elif t == 'flip':
                    ax = int(m.get('axis', -1))
                    if 0 <= ax < 3:
                        v.sign[ax] = -1.0 if m.get('on') else 1.0
                        changed = True
                elif t == 'reset':
                    v.R = np.eye(3)
                    self.angles = dict(xy=0.0, yz=0.0, xz=0.0)
                    changed = True
                elif t == 'save':
                    status = self._save(v)
                    if v.acc is not None:                 # report it even when idle
                        self._publish(v, status)
                elif t == 'hello' and self.latest:
                    self._emit(*self.latest)
            if changed:
                v.apply()
                status = ''
            if v.step(self.target):
                self._publish(v, status)
            else:
                self.wake.wait(timeout=0.5)
                self.wake.clear()

    def _save(self, v):
        with self.lock:
            waiters, self._waiters = self._waiters, []
        self._publish(v, 'saving (about 20 s)...', saving=True)
        try:
            big = v.sibling(self.save_res)
            out = big.still(self.save_spp, calibrate=False)
            del big
            self.save_dir.mkdir(parents=True, exist_ok=True)
            s = v.state
            name = (f'{self.name}_{time.strftime("%Y%m%d-%H%M%S")}_el{s["elevation"]:.0f}'
                    f'_size{s["size"]:.2f}_gap{s["gap"]:.1f}.png')
            path = self.save_dir / name
            Image.fromarray(out).save(path)
            np.save(path.with_name(path.stem + '_rotation.npy'), v.matrix)
            for done, box in waiters:
                box['path'] = path
                done.set()
            self._saved = name
            return f'saved {path}'
        except Exception as e:
            for done, box in waiters:
                box['error'] = repr(e)
                done.set()
            return f'save failed: {e!r}'

    def _publish(self, v, status, saving=False):
        buf = io.BytesIO()
        Image.fromarray(v.develop()).save(buf, 'JPEG', quality=92, subsampling=0)
        meta = dict(spp=v.spp, target=self.target, status=status, saving=saving,
                    saved=getattr(self, '_saved', None) if status.startswith('saved') else None,
                    flip=[bool(x) for x in (v.sign < 0)],
                    **v.state, **{f'rot_{k}': round(a, 3) for k, a in self.angles.items()})
        self._emit(buf.getvalue(), meta)

    def _emit(self, jpeg, meta):
        self.latest = (jpeg, meta)
        for fn in list(self.listeners):
            try:
                fn(jpeg, meta)
            except Exception:
                traceback.print_exc()
