"""Rendering and the viewers, on the GPU (small images, so each takes about a second)."""
import asyncio
import json
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from PIL import Image

import rayplot as rp
from rayplot import _prepare
from rayplot.core import FloorScene
from rayplot.session import ViewerSession

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs an NVIDIA GPU')


@pytest.fixture(scope='module')
def cloud():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(3000, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz, xyz[:, 2]


def session(cloud, tmp_path, **kw):
    xyz, rgb = _prepare(*cloud, 'viridis', None, None)
    kw = dict(res=256, target_spp=8, save_dir=tmp_path, save_res=400, save_spp=32) | kw
    return ViewerSession(xyz, rgb, **kw).wait_ready()


def settle(s, target=None, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        if s.latest and s.latest[1]['spp'] >= (target or s.target):
            return s.latest[1]
        time.sleep(0.05)
    raise TimeoutError


def free_port():
    with socket.socket() as k:
        k.bind(('127.0.0.1', 0))
        return k.getsockname()[1]


def test_render(cloud, tmp_path):
    img = rp.render(*cloud, out=tmp_path / 'a.png', res=256, spp=8)
    assert (tmp_path / 'a.png').exists() and max(img.size) <= 256
    assert (np.asarray(img) < 200).any()                 # something drawn on the white page


def test_exposure_is_repeatable(cloud):
    xyz, rgb = _prepare(*cloud, 'viridis', None, None)
    sc = FloorScene(xyz, rgb, 1600, zoom=3.0)            # > 1M floor pixels: subsampled
    import mitsuba as mi
    from rayplot.core import MITSUBA
    with MITSUBA:
        img = mi.render(sc.scene, spp=2, seed=0).torch()
    assert ((img[..., 9] > 0.999) & (img[..., 3] > 0.999)).sum() > 1_000_000
    assert sc.calibrate(img) == sc.calibrate(img)
    sc.release()


def test_viewer_save_is_the_same_render_as_render_of_its_view(cloud, tmp_path):
    s = session(cloud, tmp_path, rotation=rp.plane_rotation(xy=30), size=0.3, gap=2.0)
    s.send({'type': 'rotate', 'dx': 40, 'dy': -25})
    s.send({'type': 'flip', 'axis': 1, 'on': True})
    s.send({'type': 'set', 'zoom': 1.1, 'elevation': 40})
    settle(s)
    png = s.save()
    view = s.view
    s.stop()
    assert png.with_suffix('.json').exists()
    saved = json.loads(png.with_suffix('.json').read_text())
    assert saved['view']['zoom'] == pytest.approx(1.1) and saved['res'] == 400
    loaded = rp.load_view(png)
    assert np.allclose(loaded['rotation'], view['rotation'])
    assert loaded['exposure'] == view['exposure'] and loaded['style'] == view['style']
    again = np.asarray(rp.render(*cloud, res=400, spp=32, **loaded)).astype(int)
    first = np.asarray(Image.open(png)).astype(int)
    assert again.shape == first.shape
    assert np.abs(again - first).max() <= 3              # GPU sums are not bit-exact run to run


def test_view_rotation_and_flip_survive_close(cloud, tmp_path):
    s = session(cloud, tmp_path, flip=(True, False, True))
    from rayplot.widget import NotebookViewer
    w = NotebookViewer(s, title='T')
    settle(s)
    before = w.view
    w.close()
    assert w.flip == (True, False, True)
    assert np.allclose(w.rotation, np.diag([-1, 1, -1.0]))
    assert set(w.view) == {'rotation', 'elevation', 'zoom', 'size', 'gap', 'exposure', 'style'}
    assert w.view['exposure'] == before['exposure']
    with pytest.raises(RuntimeError):
        w.save()


def test_close_frees_the_gpu(cloud, tmp_path):
    free = lambda: torch.cuda.mem_get_info()[0]
    session(cloud, tmp_path, res=960).stop()           # first use allocates caches for good
    f0 = free()
    s = session(cloud, tmp_path, res=960, target_spp=64)
    settle(s)
    used = f0 - free()
    s.stop()
    assert not s.thread.is_alive()
    assert f0 - free() < 0.3 * used


def test_a_bad_message_does_not_kill_the_viewer(cloud, tmp_path):
    s = session(cloud, tmp_path)
    settle(s)
    s.send({'type': 'set', 'zoom': 'abc'})
    t = time.time()
    while 'error' not in s.latest[1]['status'] and time.time() - t < 10:
        time.sleep(0.05)
    assert 'error' in s.latest[1]['status'] and s.thread.is_alive()
    s.send({'type': 'set', 'zoom': 1.2})
    t = time.time()
    while s.latest[1]['zoom'] != 1.2 and time.time() - t < 10:
        time.sleep(0.05)
    assert settle(s)['status'] == '' and s.latest[1]['zoom'] == 1.2
    s.stop()


def test_wait_ready_times_out(cloud, tmp_path):
    xyz, rgb = _prepare(*cloud, 'viridis', None, None)
    s = ViewerSession(xyz, rgb, res=256, save_dir=tmp_path)
    with pytest.raises(TimeoutError):
        s.wait_ready(timeout=1e-4)
    s.thread.join(30)
    assert not s.thread.is_alive()


def test_render_while_a_viewer_renders(cloud, tmp_path):
    s = session(cloud, tmp_path, res=720, target_spp=100_000)    # never settles
    settle(s, target=16)
    img = rp.render(*cloud, res=256, spp=16)
    assert 'error' not in s.latest[1]['status']
    s.stop()
    assert img.size[0] > 100


def test_exit_without_closing_a_viewer(tmp_path):
    code = ('import numpy as np, time, rayplot as rp\n'
            'v = rp.browser(np.random.rand(500, 3), "red", port=%d, res=480, target_spp=10**6,'
            ' save_dir=%r)\n'
            'time.sleep(1)\n') % (free_port(), str(tmp_path))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


def test_browser(cloud, tmp_path):
    import aiohttp
    port = free_port()
    v = rp.browser(*cloud, port=port, res=256, target_spp=8, save_dir=tmp_path, save_res=300,
                   save_spp=8, title='My <plot>')

    async def drive():
        async with aiohttp.ClientSession() as http:
            page = await (await http.get(v.url)).text()
            assert '<title>My &lt;plot&gt;</title>' in page
            async with http.ws_connect(v.url + 'ws') as ws:
                meta = json.loads((await ws.receive()).data)
                assert isinstance((await ws.receive()).data, bytes) and 'spp' in meta
                await ws.send_str(json.dumps({'type': 'save'}))
                while True:
                    msg = await ws.receive(timeout=60)
                    if msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data)['saved']:
                        name = json.loads(msg.data)['saved']
                        break
            r = await http.get(v.url + 'saves/' + name)
            assert r.status == 200 and (await r.read())[:4] == b'\x89PNG'

    asyncio.run(drive())
    assert v.flip == (False, False, False) and v.rotation.shape == (3, 3)
    v.close()
    w = rp.browser(*cloud, port=port, res=256, target_spp=8, save_dir=tmp_path)  # port is free
    w.stop()


def test_browser_warns_off_loopback(cloud, tmp_path):
    with pytest.warns(UserWarning, match='no authentication'):
        v = rp.browser(*cloud, host='0.0.0.0', port=free_port(), res=256, target_spp=8,
                       save_dir=tmp_path)
    v.close()
