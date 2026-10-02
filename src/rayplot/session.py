"""The interactive render loop, shared by the notebook widget and the browser server.

A `ViewerSession` owns a `FloorScene` on its own thread (every Mitsuba call happens there).
Front ends `send()` messages and `subscribe()` to frames (JPEG bytes + a meta dict).

Messages (JSON-like dicts):
  {'type': 'rotate', 'dx', 'dy'}          free turn from a mouse drag (pixels * 0.4 deg)
  {'type': 'plane', 'plane', 'angle'}     turn within one plane to the slider's angle
  {'type': 'set', elevation/zoom/size/gap}
  {'type': 'flip', 'axis': 0|1|2, 'on': bool}   turn one data axis over (a reflection)
  {'type': 'reset'}                       undo all turns
  {'type': 'save'}                        print-quality PNG (and its view, as JSON) into save_dir
  {'type': 'hello'}                       re-send the latest frame (a new front end)
"""
import atexit
import dataclasses
import io
import json
import threading
import time
import traceback
import weakref
from pathlib import Path

import numpy as np
from PIL import Image

from .core import SIZE_MIN, FloorScene, PLANES, Style, wrap

# Open sessions, stopped at exit: a render thread still inside Mitsuba while the interpreter
# shuts down aborts the process. (Registered after Mitsuba is imported, so it runs first.)
_live = weakref.WeakSet()


@atexit.register
def _stop_all():
    for s in list(_live):
        s.stop()


class ViewerSession:
    def __init__(self, xyz, rgb, *, res=960, target_spp=512, save_dir='rayplot_saves',
                 save_res=2400, save_spp=1024, style=None, exposure=None, name='view', **view):
        self.args = dict(xyz=xyz, rgb=rgb, res=res, style=style, exposure=exposure, **view)
        self.target, self.save_res, self.save_spp = target_spp, save_res, save_spp
        self.save_dir, self.name = Path(save_dir), name
        self.lock = threading.Lock()
        self.pending, self.listeners = [], []
        self.wake, self.ready, self.stopped = threading.Event(), threading.Event(), False
        self.latest, self.error, self.scene = None, None, None
        self.angles = dict(xy=0.0, yz=0.0, xz=0.0)
        self._waiters, self._saved, self._last_view, self._last_flip = [], None, {}, None
        self.thread = threading.Thread(target=self._run, daemon=True, name='rayplot')
        _live.add(self)
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
        if not self.ready.wait(timeout):
            self.stop(wait=False)
            raise TimeoutError(f'rayplot: the viewer was not ready after {timeout} s')
        if self.error:
            raise RuntimeError(self.error)
        return self

    def save(self, timeout=600):
        """Render the current view at print quality; returns the PNG's path."""
        if self.stopped:
            raise RuntimeError('rayplot: this viewer is closed')
        done = threading.Event()
        box = {}
        with self.lock:
            self._waiters.append((done, box))
        self.send({'type': 'save'})
        if not done.wait(timeout):
            raise TimeoutError(f'rayplot: the save did not finish within {timeout} s')
        if 'error' in box:
            raise RuntimeError(box['error'])
        return box['path']

    @property
    def rotation(self):
        """The turn and the axis flips, as the one matrix that reproduces the view."""
        return self.view.get('rotation')

    @property
    def flip(self):
        s = self.scene
        return self._last_flip if s is None else tuple(bool(x) for x in (s.sign < 0))

    @property
    def view(self):
        """Current settings, as keyword arguments for `rayplot.render`.

        The flips ride inside `rotation`, so this stays a complete description of the view
        without a `flip=` of its own. `exposure` is the viewer's, so a still made from this
        matches the screen and the viewer's own saves. Still available after the viewer closes.
        """
        s = self.scene
        if s is None:
            return dict(self._last_view)
        return dict(rotation=s.matrix, **s.state, exposure=s.exposure, style=s.style)

    def stop(self, wait=True, timeout=30):
        """End the render thread and free the GPU (waiting for that, unless wait=False)."""
        self.stopped = True
        self.wake.set()
        if wait and self.thread is not threading.current_thread():
            self.thread.join(timeout)

    # ---- render thread -----------------------------------------------------------------
    def _run(self):
        try:
            a = self.args
            size = a.get('size', 0.25)
            self.scene = FloorScene(a['xyz'], a['rgb'], a['res'], style=a['style'],
                                    size_min=min(SIZE_MIN, size),
                                    **{k: a[k] for k in ('rotation', 'flip', *FloorScene.STATE)
                                       if k in a})
            if a['exposure'] is None:
                self.scene.calibrate()
            else:
                self.scene.exposure = float(a['exposure'])
        except Exception:
            self.error = traceback.format_exc()
            self.ready.set()
            return
        self.ready.set()
        try:
            self._loop(self.scene)
        finally:
            self._last_view, self._last_flip = self.view, self.flip
            v, self.scene = self.scene, None
            v.release()

    def _loop(self, v):
        status, failed = '', False
        while not self.stopped:
            with self.lock:
                msgs, self.pending = self.pending, []
            try:
                changed = self._handle(v, msgs)
                if changed is None:                       # a save: report it even when idle
                    status = self._status
                    if v.acc is not None:
                        self._publish(v, status)
                elif changed:
                    v.apply()
                    status, failed = '', False
                if not failed and v.step(self.target):
                    self._publish(v, status)
                    continue
            except Exception as e:
                # Keep the viewer alive: say what went wrong and wait for the next change,
                # rather than dying and leaving the front end frozen on its last frame.
                traceback.print_exc()
                status, failed = f'error: {e!r}', True
                try:
                    if v.acc is not None:
                        self._publish(v, status)
                except Exception:
                    pass
            self.wake.wait(timeout=0.5)
            self.wake.clear()

    def _handle(self, v, msgs):
        """Apply messages; True if the view changed, None after a save, else False."""
        changed, saved = False, False
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
                self._status = self._save(v)
                saved = True
            elif t == 'hello' and self.latest:
                self._emit(*self.latest)
        return None if saved and not changed else changed

    def _save(self, v):
        with self.lock:
            waiters, self._waiters = self._waiters, []
        self._publish(v, 'saving…', saving=True)
        try:
            big = v.sibling(self.save_res)
            out = big.still(self.save_spp, calibrate=False)
            big.release()
            self.save_dir.mkdir(parents=True, exist_ok=True)
            s = v.state
            name = (f'{self.name}_{time.strftime("%Y%m%d-%H%M%S")}_el{s["elevation"]:.0f}'
                    f'_size{s["size"]:.2f}_gap{s["gap"]:.1f}.png')
            path = self.save_dir / name
            Image.fromarray(out).save(path)
            record = dict(view=_jsonable(dict(rotation=v.matrix, **s, exposure=v.exposure,
                                              style=v.style)),
                          res=self.save_res, spp=self.save_spp, points=v.n)
            path.with_suffix('.json').write_text(json.dumps(record, indent=1))
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
                    saved=self._saved if status.startswith('saved') else None,
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


def _jsonable(view):
    out = dict(view)
    out['rotation'] = np.asarray(view['rotation']).tolist()
    out['style'] = dataclasses.asdict(view['style'])
    return out


def load_view(path):
    """The view a viewer saved, as keyword arguments for `rayplot.render`.

    `path` is the saved PNG or the JSON beside it. `render(points, colours, **load_view(png))`
    redraws the saved image (with the same `res` and `spp` too, it is the same render)."""
    path = Path(path)
    record = json.loads(path.with_suffix('.json').read_text())
    view = dict(record['view'])
    view['rotation'] = np.array(view['rotation'])
    style = dict(view['style'])
    style['light_offset'] = tuple(style['light_offset'])
    view['style'] = Style(**style)
    return view


class ViewerAPI:
    """What both viewers offer on top of their front end; `self.session` is a ViewerSession."""

    def save(self):
        """Render the current view at print quality into `save_dir`; returns the PNG's path.
        A JSON file beside it records the whole view (see `rayplot.load_view`)."""
        return self.session.save()

    @property
    def view(self):
        """Current settings, as keyword arguments: `rayplot.render(points, colours, **v.view)`
        reproduces what is on screen."""
        return self.session.view

    @property
    def rotation(self):
        """The turn, with the axis flips folded in (as in `.view`)."""
        return self.session.rotation

    @property
    def flip(self):
        """The axis-flip tickboxes, as three booleans."""
        return self.session.flip
