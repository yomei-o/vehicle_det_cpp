// vehicle_det.hpp — 車両検出器を他のプログラムから使うための小さな入口（ヘッダーオンリー、依存なし）
//
//   vd::Detector det;
//   det.load("models/vehicle_det.onnx");
//   std::vector<vd::Box> boxes = det.detect(rgb, W, H);   // rgb: RGB24、W x H
//
// 中身は pure/ の自前 ONNX ランナと yolov8 のデコード（姉妹リポ yolo_lpr_cpp と同じもの）。
// 入力はグラフが宣言する H x W にそのまま縮める（1280x720 → 960x544 なら縦横比はほぼ保たれる）。
// stb は使わないので、STB_IMAGE_IMPLEMENTATION は呼ぶ側の都合でよい。
#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "pipeline.hpp"

namespace vd {

struct Box { float x1, y1, x2, y2, score; };

struct Detector {
    onx::Graph g;
    jl::DetCfg cfg;
    bool ok = false;

    bool load(const std::string& path, float conf = 0.25f, float nms = 0.45f) {
        g = onx::load_onnx(path);
        cfg.kind = jl::DetKind::V8;
        cfg.nc = 1;
        cfg.plate_class = 0;
        cfg.imgsz = 0;  // グラフの H x W
        cfg.conf = conf;
        cfg.nms = nms;
        cfg.v8_fmt = BoxFmt::CXCYWH;  // Ultralytics の書き出し
        int iw = 0, ih = 0;
        jl::graph_input_hw(g, iw, ih);
        ok = iw > 0 && ih > 0;
        return ok;
    }

    std::vector<Box> detect(const unsigned char* rgb, int W, int H) const {
        std::vector<Box> out;
        if (!ok) return out;
        for (const jl::Box& b : jl::detect_plates(g, rgb, W, H, cfg)) out.push_back({b.x1, b.y1, b.x2, b.y2, b.score});
        return out;
    }
};

}  // namespace vd
