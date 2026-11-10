// 推論用の Web Worker。1 フレームに数秒かかるので、ページのスレッドで回すと動画が止まる。
// 入: {type:'init', model:ArrayBuffer} / {type:'frame', pixels:ArrayBuffer(RGBA), w, h, conf}
// 出: {type:'ready', nodes, iw, ih} / {type:'result', boxes:Float32Array(n*5), n, ms} / {type:'error', text}
let M = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      if (!M) { self.importScripts('vdet.js'); M = await createVdet(); }
      const b = new Uint8Array(m.model);
      const p = M._malloc(b.length);
      M.HEAPU8.set(b, p);
      const nodes = M._vd_load(p, b.length);
      M._free(p);
      self.postMessage({ type: 'ready', nodes, iw: M._vd_input_w(), ih: M._vd_input_h() });
      return;
    }
    if (m.type === 'frame') {
      const px = new Uint8Array(m.pixels);
      const p = M._malloc(px.length);
      M.HEAPU8.set(px, p);
      const t0 = performance.now();
      const n = M._vd_detect(p, m.w, m.h, m.conf, 0.45);
      const ms = performance.now() - t0;
      M._free(p);
      const boxes = new Float32Array(M.HEAPF32.buffer, M._vd_result(), Math.max(0, n) * 5).slice();
      self.postMessage({ type: 'result', boxes, n, ms, w: m.w, h: m.h }, [boxes.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', text: String(err) });
  }
};
