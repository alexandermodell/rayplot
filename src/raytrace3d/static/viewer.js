// raytrace3d viewer UI, shared by the browser page (websocket) and the notebook widget
// (anywidget). A transport is { send(obj), listen(fn(meta | null, bytes | null)) }.

const SLIDERS = ['elevation', 'zoom', 'size', 'gap'];
const PLANES = ['xy', 'yz', 'xz'];
const deg = v => `${(+v).toFixed(1)}°`;
const FMT = { elevation: v => `${(+v).toFixed(0)}°`, zoom: v => `${(+v).toFixed(2)}×`,
              size: v => (+v).toFixed(2), gap: v => (+v).toFixed(1), xy: deg, yz: deg, xz: deg };

const TEMPLATE = `
  <div class="rt3d-stage">
    <img alt="Path-traced view of the points above the floor">
    <div class="rt3d-hint">drag to turn the points freely</div>
    <div class="rt3d-badge">starting…</div>
  </div>
  <div class="rt3d-panel">
    <div class="rt3d-title">Raytraced scatter</div>
    <label>Turn in xy plane (about vertical) <span data-v="xy"></span></label>
    <input type="range" data-k="xy" min="-180" max="180" step="0.5" value="0">
    <label>Turn in yz plane <span data-v="yz"></span></label>
    <input type="range" data-k="yz" min="-180" max="180" step="0.5" value="0">
    <label>Turn in xz plane <span data-v="xz"></span></label>
    <input type="range" data-k="xz" min="-180" max="180" step="0.5" value="0">
    <label>Flip an axis over</label>
    <div class="rt3d-flips">
      <label><input type="checkbox" data-flip="0"> x</label>
      <label><input type="checkbox" data-flip="1"> y</label>
      <label><input type="checkbox" data-flip="2"> z</label>
    </div>
    <hr>
    <label>Camera height <span data-v="elevation"></span></label>
    <input type="range" data-k="elevation" min="2" max="88" step="1" value="30">
    <label>Zoom <span data-v="zoom"></span></label>
    <input type="range" data-k="zoom" min="0.4" max="4" step="0.01" value="1.35">
    <label>Bead size <span data-v="size"></span></label>
    <input type="range" data-k="size" min="0.08" max="0.8" step="0.01" value="0.25">
    <label>Gap to floor (bead radii) <span data-v="gap"></span></label>
    <input type="range" data-k="gap" min="0.2" max="10" step="0.1" value="1.5">
    <div class="rt3d-buttons">
      <button data-act="reset">Reset turn</button>
      <button data-act="save" class="rt3d-primary">Save PNG</button>
    </div>
    <div class="rt3d-status"></div>
  </div>`;

export function mount(el, transport, { saveLink = null } = {}) {
  const root = document.createElement('div');
  root.className = 'rt3d';
  root.innerHTML = TEMPLATE;
  el.appendChild(root);
  const q = s => root.querySelector(s);
  const input = k => q(`[data-k="${k}"]`);
  const label = k => q(`[data-v="${k}"]`);
  const stage = q('.rt3d-stage'), img = q('img'), badge = q('.rt3d-badge');
  const flips = [0, 1, 2].map(i => q(`[data-flip="${i}"]`));
  const status = q('.rt3d-status'), saveBtn = q('[data-act="save"]');

  let url = null, pendingSet = {}, pendingPlane = {}, drag = null, dx = 0, dy = 0, raf = false;
  let gotFrame = false;
  const touched = {};                      // last local edit per control: ignore echoes briefly
  const fresh = k => performance.now() - (touched[k] || 0) < 400;
  const show = (k, v) => { input(k).value = v; label(k).textContent = FMT[k](v); };

  function onMeta(m) {
    badge.textContent = m.spp >= m.target ? `${m.spp} spp · done` : `${m.spp} / ${m.target} spp`;
    for (const k of SLIDERS) if (!fresh(k) && m[k] !== undefined) show(k, m[k]);
    for (const k of PLANES) if (!fresh(k) && m['rot_' + k] !== undefined) show(k, m['rot_' + k]);
    if (m.flip) flips.forEach((c, i) => { if (!fresh('flip' + i)) c.checked = !!m.flip[i]; });
    status.textContent = m.status || '';
    if (m.saved && saveLink) {
      const a = document.createElement('a');
      a.href = saveLink(m.saved); a.target = '_blank'; a.textContent = 'open';
      status.append(' — ', a);
    }
    saveBtn.disabled = !!m.saving;
  }
  function onFrame(bytes) {
    const next = URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));
    img.onload = () => { if (url) URL.revokeObjectURL(url); url = next; };
    img.src = next;
  }
  transport.listen((meta, bytes) => {
    if (meta) { gotFrame = true; onMeta(meta); }
    if (bytes) onFrame(bytes);
  });

  function flush() {
    raf = false;
    if (Object.keys(pendingSet).length) { transport.send({ type: 'set', ...pendingSet }); pendingSet = {}; }
    for (const [plane, angle] of Object.entries(pendingPlane)) transport.send({ type: 'plane', plane, angle });
    pendingPlane = {};
    if (dx || dy) { transport.send({ type: 'rotate', dx, dy }); dx = dy = 0; }
  }
  const schedule = () => { if (!raf) { raf = true; requestAnimationFrame(flush); } };

  for (const k of [...SLIDERS, ...PLANES]) {
    label(k).textContent = FMT[k](input(k).value);
    input(k).addEventListener('input', () => {
      label(k).textContent = FMT[k](input(k).value);
      touched[k] = performance.now();
      (SLIDERS.includes(k) ? pendingSet : pendingPlane)[k] = +input(k).value;
      schedule();
    });
  }
  flips.forEach((c, i) => c.addEventListener('change', () => {
    touched['flip' + i] = performance.now();
    transport.send({ type: 'flip', axis: i, on: c.checked });
  }));
  stage.addEventListener('pointerdown', e => {
    drag = { x: e.clientX, y: e.clientY }; stage.setPointerCapture(e.pointerId);
    stage.classList.add('rt3d-dragging');
  });
  stage.addEventListener('pointermove', e => {
    if (!drag) return;
    const s = 900 / stage.clientWidth;     // a full stage width is about one full turn
    dx += (e.clientX - drag.x) * s; dy += (e.clientY - drag.y) * s;
    drag = { x: e.clientX, y: e.clientY }; schedule();
  });
  const end = () => { drag = null; stage.classList.remove('rt3d-dragging'); };
  stage.addEventListener('pointerup', end);
  stage.addEventListener('pointercancel', end);
  q('[data-act="reset"]').onclick = () => transport.send({ type: 'reset' });
  saveBtn.onclick = () => { saveBtn.disabled = true; transport.send({ type: 'save' }); };
  // A session that has reached target_spp publishes nothing further, so a front end that mounts
  // after the render settled -- at a low target_spp that is almost immediately -- is only ever
  // shown a frame in answer to its own hello. Keep asking until one arrives, in case the first is
  // lost while the comm is still being wired up; otherwise the badge sits on "starting..." for
  // good.
  transport.send({ type: 'hello' });
  let asks = 0;
  const again = setInterval(() => {
    if (gotFrame || ++asks > 25) clearInterval(again);
    else transport.send({ type: 'hello' });
  }, 600);
  return () => { clearInterval(again); if (url) URL.revokeObjectURL(url); root.remove(); };
}

// anywidget entry point (notebook)
export default {
  render({ model, el }) {
    const transport = {
      send: obj => model.send(obj),
      listen: fn => model.on('msg:custom', (msg, buffers) => {
        const b = buffers && buffers.length ? buffers[0] : null;
        fn(msg, b ? b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) : null);
      }),
    };
    return mount(el, transport);
  },
};
