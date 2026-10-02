"""Notebook front end: an anywidget showing a ViewerSession (works in Jupyter and VS Code)."""
import time
from pathlib import Path

try:
    import anywidget
    import traitlets
except ImportError as e:
    raise ImportError('rayplot.notebook needs anywidget: pip install "rayplot[notebook]"') from e

from .session import ViewerAPI

STATIC = Path(__file__).parent / 'static'


class NotebookViewer(ViewerAPI, anywidget.AnyWidget):
    """Interactive path-traced view in a notebook cell. `close()` also frees the GPU."""
    _esm = STATIC / 'viewer.js'
    _css = STATIC / 'viewer.css'
    title = traitlets.Unicode('Raytraced scatter').tag(sync=True)

    def __init__(self, session, max_fps=20, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self._min_dt, self._last = 1 / max_fps, 0.0
        self.on_msg(self._on_msg)
        session.subscribe(self._push)

    def _on_msg(self, widget, content, buffers):
        if content.get('type') == 'hello' and self.session.latest:
            self._send_frame(*self.session.latest)
        elif content.get('type') != 'hello':
            self.session.send(content)

    def _push(self, jpeg, meta):
        # Frames arrive faster than a notebook comm should carry while dragging; skip some,
        # but never the last of a refinement (spp == target) or a status change.
        now = time.monotonic()
        if now - self._last < self._min_dt and meta['spp'] < meta['target'] and not meta['status']:
            return
        self._last = now
        self._send_frame(jpeg, meta)

    def _send_frame(self, jpeg, meta):
        try:
            self.send(meta, [jpeg])
        except Exception as e:
            # The render thread must not die for a dropped frame, but a silent `pass` here hides
            # exactly the fault that leaves the viewer stuck on "starting...", so keep the last one.
            self.send_error = e

    def close(self):
        """Stop rendering and free the GPU; `.view` and `.flip` still answer afterwards."""
        self.session.unsubscribe(self._push)
        self.session.stop()
        super().close()
