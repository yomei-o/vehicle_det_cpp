// WASM の入口。ページ（の Web Worker）が ONNX のバイト列を渡し、RGBA のフレームを押し込む。
// CLI（vdet detect）と同じヘッダ・同じグラフ・同じデコードなので、ブラウザでも同じ箱が出る。
//
// build: sh wasm/build.sh   → docs/vdet.js, docs/vdet.wasm
#include "pipeline.hpp"
#include <emscripten/emscripten.h>
#include <vector>

static onx::Graph g_det;
static bool g_ok = false;
static std::vector<float> g_out;      // [n, x1, y1, x2, y2, score, ...]
static std::vector<unsigned char> g_rgb;

extern "C" {

// ONNX を読む。戻り値はノード数（失敗なら 0）
EMSCRIPTEN_KEEPALIVE int vd_load(const unsigned char* bytes, int n) {
  g_det = onx::parse_onnx(bytes, (size_t)n);
  int iw = 0, ih = 0;
  jl::graph_input_hw(g_det, iw, ih);
  g_ok = iw > 0 && ih > 0 && !g_det.nodes.empty();
  return g_ok ? (int)g_det.nodes.size() : 0;
}

// グラフの入力の幅・高さ
EMSCRIPTEN_KEEPALIVE int vd_input_w() { int iw = 0, ih = 0; jl::graph_input_hw(g_det, iw, ih); return iw; }
EMSCRIPTEN_KEEPALIVE int vd_input_h() { int iw = 0, ih = 0; jl::graph_input_hw(g_det, iw, ih); return ih; }

// RGBA のフレームから車を検出する。戻り値は台数、箱は vd_result() から [x1, y1, x2, y2, score] × 台数
EMSCRIPTEN_KEEPALIVE int vd_detect(const unsigned char* rgba, int w, int h, float conf, float nms) {
  g_out.clear();
  if (!g_ok) return -1;
  g_rgb.resize((size_t)w * h * 3);
  for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
    g_rgb[i * 3] = rgba[i * 4]; g_rgb[i * 3 + 1] = rgba[i * 4 + 1]; g_rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  jl::DetCfg cfg;
  cfg.kind = jl::DetKind::V8;
  cfg.nc = 1;
  cfg.plate_class = 0;
  cfg.imgsz = 0;
  cfg.conf = conf;
  cfg.nms = nms;
  cfg.v8_fmt = BoxFmt::CXCYWH;
  std::vector<jl::Box> boxes = jl::detect_plates(g_det, g_rgb.data(), w, h, cfg);
  for (const jl::Box& b : boxes) { g_out.push_back(b.x1); g_out.push_back(b.y1); g_out.push_back(b.x2); g_out.push_back(b.y2); g_out.push_back(b.score); }
  return (int)boxes.size();
}

EMSCRIPTEN_KEEPALIVE const float* vd_result() { return g_out.data(); }

}  // extern "C"
