"""Browser front end: a small aiohttp server streaming a ViewerSession over a websocket."""
import asyncio
import json
import threading
from pathlib import Path

from aiohttp import web, WSMsgType

STATIC = Path(__file__).parent / 'static'


def page(title):
    html = (STATIC / 'index.html').read_text()
    return (html.replace('__TITLE__', title)
                .replace('__CSS__', (STATIC / 'viewer.css').read_text())
                .replace('__JS__', (STATIC / 'viewer.js').read_text()))


class BrowserViewer:
    """Serves one session at http://host:port/. Use `stop()` to shut it down."""

    def __init__(self, session, host='127.0.0.1', port=17100, title='Raytraced scatter'):
        self.session, self.host, self.port, self.title = session, host, port, title
        self.url = f'http://{host}:{port}/'
        self._stop = None
        self._started = threading.Event()
        self.error = None

    def __repr__(self):
        return f'<raytrace3d browser viewer at {self.url}>'

    def _repr_html_(self):
        return (f'raytrace3d viewer running at <a href="{self.url}" target="_blank">{self.url}</a>'
                ' (forward the port if this machine is remote)')

    async def _serve(self):
        loop = asyncio.get_running_loop()
        notify, clients, sent = asyncio.Event(), set(), [None]
        self._stop = asyncio.Event()

        def on_frame(jpeg, meta):                     # render thread -> event loop
            loop.call_soon_threadsafe(notify.set)

        async def fanout():
            while True:
                await notify.wait()
                notify.clear()
                frame = self.session.latest
                if frame is None or frame is sent[0]:
                    continue
                sent[0] = frame
                for ws in list(clients):
                    try:
                        await ws.send_str(json.dumps(frame[1]))
                        await ws.send_bytes(frame[0])
                    except Exception:
                        clients.discard(ws)

        async def index(request):
            return web.Response(text=page(self.title), content_type='text/html')

        async def ws_handler(request):
            ws = web.WebSocketResponse(heartbeat=30)
            await ws.prepare(request)
            clients.add(ws)
            if self.session.latest:
                await ws.send_str(json.dumps(self.session.latest[1]))
                await ws.send_bytes(self.session.latest[0])
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    m = json.loads(msg.data)
                    if m.get('type') != 'hello':          # latest frame already sent
                        self.session.send(m)
            clients.discard(ws)
            return ws

        save_dir = self.session.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        app = web.Application()
        app.add_routes([web.get('/', index), web.get('/ws', ws_handler),
                        web.static('/saves', save_dir)])
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, self.host, self.port).start()
        self.session.subscribe(on_frame)
        task = asyncio.create_task(fanout())
        self._started.set()
        await self._stop.wait()
        self.session.unsubscribe(on_frame)
        task.cancel()
        for ws in list(clients):
            await ws.close()
        await runner.cleanup()

    def _main(self):
        try:
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            self.error = e
            self._started.set()

    def start(self, block=False):
        if block:
            print(f'raytrace3d viewer at {self.url}  (Ctrl-C to stop)', flush=True)
            try:
                self._main()
            finally:
                self.session.stop()
            return self
        self._thread = threading.Thread(target=self._main, daemon=True, name='raytrace3d-web')
        self._thread.start()
        self._started.wait(30)
        if self.error:
            raise self.error
        return self

    def save(self):
        """Render the current view at print quality; returns the PNG's path."""
        return self.session.save()

    @property
    def view(self):
        """Current settings, as keyword arguments for `raytrace3d.render`."""
        return self.session.view

    def stop(self):
        """Stop serving and free the GPU."""
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self.session.stop()
