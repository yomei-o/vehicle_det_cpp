// node wasm/test_node.js docs/models/vehicle_det.onnx frame.rgba 1280 720
const fs = require('fs');
const createVdet = require('../docs/vdet.js');
(async () => {
  const [onnx, rgbaPath, w, h] = process.argv.slice(2);
  const M = await createVdet();
  const ob = fs.readFileSync(onnx);
  let p = M._malloc(ob.length); M.HEAPU8.set(ob, p);
  console.log('nodes', M._vd_load(p, ob.length), 'input', M._vd_input_w(), 'x', M._vd_input_h()); M._free(p);
  const px = fs.readFileSync(rgbaPath);
  p = M._malloc(px.length); M.HEAPU8.set(px, p);
  for (let k = 0; k < 2; k++) {
    const t0 = Date.now();
    const n = M._vd_detect(p, +w, +h, 0.25, 0.45);
    const r = new Float32Array(M.HEAPF32.buffer, M._vd_result(), n * 5);
    console.log('vehicles', n, 'ms', Date.now() - t0, 'first', Array.from(r.slice(0, 5)).map(v => v.toFixed(1)).join(' '));
  }
})();
