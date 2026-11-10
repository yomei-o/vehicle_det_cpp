// vdet — 車両検出器（yolov8 系 ONNX、1 クラス vehicle）の推論・評価・学習。依存なし（自前 ONNX ランナ + stb）。
//
//   vdet detect --det models/vehicle_det.onnx --img a.jpg [--out a_det.png] [--conf 0.25] [--json]
//   vdet val    --det models/vehicle_det.onnx --data dataset/val            （mAP50 / mAP50-95 / P / R）
//   vdet train  --init models/vehicle_det.onnx --data dataset/train --val dataset/val --epochs N --export out.onnx
//
// エンジン（pure/）と、評価・学習のコードは姉妹リポ yolo_lpr_cpp（jlpr val/train --model det）から持ってきたもの。
// 学習は ONNX グラフをそのまま学習する（ultralytics の損失と項ごとに一致させてある）。
#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "pipeline.hpp"
#include "onnx_train.hpp"
#include "trainrt.hpp"
#include "train_ocr.hpp"
#include "train_det.hpp"
#include "eval_det.hpp"
#include "train_corner.hpp"   // crn::clip_grad_norm
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static std::string arg_of(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 2; i + 1 < argc; ++i) if (key == argv[i]) return argv[i + 1];
  return def;
}
static bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 2; i < argc; ++i) if (key == argv[i]) return true;
  return false;
}

// Detector evaluation over a YOLO-format directory: the product path (full graph, decode tail, NMS)
// scored with pure/eval_det.hpp. Shared by `jlpr val --model det` and by the in-training validation,
// so "the number the trainer chased" and "the number the CLI reports" cannot drift apart.
struct DetEval {
  evd::Report r;
  std::vector<int> tp_b, gt_b;
  int fp_bucket = 0, n_det_at_conf = 0, empty_frames = 0, fp_empty = 0, n_gt = 0;
  size_t frames = 0;
};

static DetEval eval_det_dir(const onx::Graph& det, jl::DetCfg cfg, const std::string& data, int limit,
                            float conf, float iou_thr, bool progress) {
  DetEval e;
  e.tp_b.assign(evd::buckets().size(), 0);
  e.gt_b.assign(evd::buckets().size(), 0);
  std::vector<std::string> files;
  trn::list_images_recursive(data + "/images", files);
  std::sort(files.begin(), files.end());
  if (limit > 0 && (int)files.size() > limit) files.resize((size_t)limit);
  e.frames = files.size();
  const std::vector<float> thr = evd::iou_thresholds();
  std::vector<evd::Scored> all;
  for (size_t k = 0; k < files.size(); ++k) {
    int W = 0, H = 0, C = 0;
    std::vector<unsigned char> blob = trn::read_file(files[k]);
    unsigned char* im = blob.empty() ? nullptr
                                     : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
    if (!im) { printf("cannot read %s\n", files[k].c_str()); continue; }
    std::string stem = files[k].substr(files[k].find_last_of("/\\") + 1);
    stem = stem.substr(0, stem.find_last_of('.'));
    std::vector<evd::GtBox> gts;
    {
      std::vector<unsigned char> lb = trn::read_file(data + "/labels/" + stem + ".txt");
      std::istringstream ss(std::string(lb.begin(), lb.end()));
      std::string line;
      while (std::getline(ss, line)) {
        std::istringstream ls(line);
        int c; float xc, yc, w, h;
        if (!(ls >> c >> xc >> yc >> w >> h)) continue;
        if (w <= 0 || h <= 0) continue;
        gts.push_back({(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H, w});
      }
    }
    std::vector<jl::Box> boxes = jl::detect_plates(det, im, W, H, cfg);
    stbi_image_free(im);
    std::vector<evd::DetBox> dets, strong;
    for (const jl::Box& b : boxes) {
      dets.push_back({b.x1, b.y1, b.x2, b.y2, b.score});
      if (b.score >= conf) strong.push_back(dets.back());
    }
    e.n_gt += (int)gts.size();
    e.n_det_at_conf += (int)strong.size();
    if (gts.empty()) { ++e.empty_frames; e.fp_empty += (int)strong.size(); }
    evd::match_frame(dets, gts, thr, all);                            // confidence-ranked, for mAP
    evd::bucket_match(strong, gts, iou_thr, e.tp_b, e.gt_b, e.fp_bucket);   // per-gt best IoU
    if (progress && (k + 1) % 100 == 0) { printf("  %zu/%zu\n", k + 1, files.size()); fflush(stdout); }
  }
  e.r = evd::summarize(all, e.n_gt, conf);
  return e;
}

static int cmd_train_det(int argc, char** argv) {
  std::string onnx_in = arg_of(argc, argv, "--init", "models/vehicle_det.onnx");
  std::string data = arg_of(argc, argv, "--data", "");
  std::string valdir = arg_of(argc, argv, "--val", "");
  std::string out = arg_of(argc, argv, "--export", "");
  std::string out_best = arg_of(argc, argv, "--export-best", "");
  std::string fixture = arg_of(argc, argv, "--dump-fixture", "");
  std::string optim = arg_of(argc, argv, "--optim", "sgd");
  // run plumbing (pure/trainrt.hpp): a long run must be resumable, stoppable and reviewable
  std::string ckpt_p = arg_of(argc, argv, "--ckpt", "");
  std::string resume_p = arg_of(argc, argv, "--resume", "");
  std::string log_p = arg_of(argc, argv, "--log", "");
  int ckpt_every = std::atoi(arg_of(argc, argv, "--ckpt-every", "0").c_str());
  int patience = std::atoi(arg_of(argc, argv, "--patience", "0").c_str());
  // Stop cleanly at a step and checkpoint. Written for the wall clock: a Kaggle session dies at 9
  // hours, so "run 4000 steps of a 20000-step schedule, then resume tomorrow" has to be expressible.
  int stop_at = std::atoi(arg_of(argc, argv, "--stop-at", "0").c_str());
  // Gradient clipping, off by default because Ultralytics does not clip and the recorded loss
  // sequences in RESUME were measured without it. It exists for one specific situation, measured:
  // a `--from-pt` start puts a *fresh* class head on a *pretrained* trunk, and at the default lr the
  // cls term explodes (step 3: 3.6e11, then NaN) while the same graph at lr 1e-6 is stable. Clipping
  // is the cheap fix; a longer warmup or a lower lr are the other two.
  float clip = (float)atof(arg_of(argc, argv, "--clip", "0").c_str());
  int imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());
  int steps = std::atoi(arg_of(argc, argv, "--steps", "30").c_str());
  int epochs = std::atoi(arg_of(argc, argv, "--epochs", "0").c_str());
  int batch = std::atoi(arg_of(argc, argv, "--batch", "2").c_str());
  int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  int freeze = std::atoi(arg_of(argc, argv, "--freeze", "0").c_str());
  int val_every = std::atoi(arg_of(argc, argv, "--val-every", "0").c_str());
  int val_limit = std::atoi(arg_of(argc, argv, "--val-limit", "0").c_str());
  float lr0 = (float)atof(arg_of(argc, argv, "--lr", optim == "sgd" ? "0.002" : "1e-4").c_str());
  float lrf = (float)atof(arg_of(argc, argv, "--lrf", "0.01").c_str());
  float momentum = (float)atof(arg_of(argc, argv, "--momentum", "0.937").c_str());
  float warm_mom = (float)atof(arg_of(argc, argv, "--warmup-momentum", "0.8").c_str());
  float wd = (float)atof(arg_of(argc, argv, "--weight-decay", "0.0005").c_str());
  float ema_decay = (float)atof(arg_of(argc, argv, "--ema-decay", "0.9999").c_str());
  float close_mosaic = (float)atof(arg_of(argc, argv, "--close-mosaic", "0.1").c_str());
  uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "1234").c_str(), nullptr, 10);
  bool cos_lr = has_flag(argc, argv, "--cos-lr");
  bool no_aug = has_flag(argc, argv, "--no-aug");
  bool no_ema = has_flag(argc, argv, "--no-ema");
  bool dump_loss = has_flag(argc, argv, "--dump-loss");

  det::AugCfg aug;
  aug.mosaic = (float)atof(arg_of(argc, argv, "--mosaic", "1.0").c_str());
  aug.fliplr = (float)atof(arg_of(argc, argv, "--fliplr", "0.5").c_str());
  aug.degrees = (float)atof(arg_of(argc, argv, "--degrees", "5.0").c_str());
  aug.translate = (float)atof(arg_of(argc, argv, "--translate", "0.15").c_str());
  aug.scale = (float)atof(arg_of(argc, argv, "--scale", "0.6").c_str());
  aug.hsv_h = (float)atof(arg_of(argc, argv, "--hsv-h", "0.015").c_str());
  aug.hsv_s = (float)atof(arg_of(argc, argv, "--hsv-s", "0.6").c_str());
  aug.hsv_v = (float)atof(arg_of(argc, argv, "--hsv-v", "0.4").c_str());

  if (data.empty()) { printf("vdet train: pass --data <dir with images/ and labels/>\n"); return 1; }
  std::vector<det::Item> items = det::read_yolo(data);
  if (limit > 0 && (int)items.size() > limit) items.resize((size_t)limit);
  if (items.empty()) { printf("no images under %s/images\n", data.c_str()); return 1; }
  size_t nbox = 0;
  for (const det::Item& it : items) nbox += it.boxes.size();
  const int per_epoch = std::max(1, (int)((items.size() + batch - 1) / batch));
  if (epochs > 0) steps = epochs * per_epoch;
  // Ultralytics warms up over 3 epochs, floored at 100 iterations; a short smoke run must not spend
  // itself entirely in the warmup, so it is also capped at a fifth of the run.
  int warmup = std::atoi(arg_of(argc, argv, "--warmup", "-1").c_str());
  if (warmup < 0) warmup = std::min(std::max(100, 3 * per_epoch), std::max(1, steps / 5));
  // close_mosaic: a fraction of the run (0.1 = the last tenth), or a step count if > 1.
  const int mosaic_off_at = close_mosaic > 1.f ? steps - (int)close_mosaic
                                              : steps - (int)(close_mosaic * steps);

  onx::Graph g = onx::load_onnx(onnx_in);
  det::HeadNames hn;
  std::string why;
  if (!det::find_v8_heads(g, hn, &why)) { printf("%s: %s\n", onnx_in.c_str(), why.c_str()); return 1; }
  if (imgsz <= 0) {
    int iw = 0, ih = 0;
    jl::graph_input_hw(g, iw, ih);
    imgsz = iw > 0 ? iw : 320;
  }
  // Only the six head convs feed the loss, so only the subgraph behind them holds parameters:
  // the DFL projection and the decode tail must stay exactly as exported (see weight_initializers).
  std::set<std::string> needed(hn.box.begin(), hn.box.end());
  needed.insert(hn.cls.begin(), hn.cls.end());
  onx::Trainable t = onx::make_trainable(g, false, needed);

  // --freeze N: hold the first N `model.<i>.` modules still, the same way Ultralytics' --freeze does.
  // The tensors stay in the graph (and in write_back); they just leave the optimiser.
  std::vector<Tensor> opt_params;
  std::vector<std::string> opt_names;
  int frozen = 0;
  for (size_t i = 0; i < t.params.size(); ++i) {
    bool hold = false;
    if (freeze > 0) {
      const std::string& n = t.param_names[i];
      if (n.rfind("model.", 0) == 0) {
        const size_t dot = n.find('.', 6);
        const int mod = std::atoi(n.substr(6, dot - 6).c_str());
        hold = mod < freeze;
      }
    }
    if (hold) ++frozen; else { opt_params.push_back(t.params[i]); opt_names.push_back(t.param_names[i]); }
  }

  if (!dump_loss) {
    printf("%s: %zu trainable tensors (%d frozen), %zu parameters\n", onnx_in.c_str(),
           t.params.size(), frozen, onx::param_count(t));
    printf("heads: box %s ... (%zu levels), cls %s ...\n", hn.box[0].c_str(), hn.box.size(),
           hn.cls[0].c_str());
    printf("data: %zu images, %zu boxes, imgsz %d, batch %d (%d/epoch), %d steps%s\n", items.size(),
           nbox, imgsz, batch, per_epoch, steps, epochs > 0 ? " (from --epochs)" : "");
    printf("optim: %s lr %g -> %g (%s), warmup %d, momentum %g, wd %g, EMA %s\n", optim.c_str(), lr0,
           lr0 * lrf, cos_lr ? "cosine" : "linear", warmup, momentum, wd, no_ema ? "off" : "on");
    printf("aug: %s\n", no_aug ? "off (--no-aug)" : "mosaic/flip/hsv/affine, mosaic closes at step "
                                                    "(see --close-mosaic)");
  }

  // --check-aug: does the augmentation move the boxes with the pixels? Build augmented batches and
  // run the *already trained* detector over them: if the affine/mosaic/flip maths were wrong, the
  // labels would no longer sit on the plates and the recall would collapse. Cheap, and it tests the
  // one thing a loss-parity test cannot see (the loss happily trains on wrong boxes).
  int check_aug = std::atoi(arg_of(argc, argv, "--check-aug", "0").c_str());
  if (check_aug > 0) {
    jl::DetCfg ccfg;
    ccfg.kind = jl::DetKind::V8;
    ccfg.nc = 1;
    ccfg.plate_class = 0;
    ccfg.imgsz = imgsz;
    ccfg.conf = 0.25f;
    ccfg.nms = 0.45f;
    ccfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;
    for (int pass = 0; pass < 2; ++pass) {              // 0 = plain resize, 1 = augmented
      Rng crng(seed);
      int gt_n = 0, hit = 0;
      double iou_sum = 0;
      for (int k = 0; k < check_aug; ++k) {
        std::vector<int> idx;
        for (int b = 0; b < batch; ++b) idx.push_back((int)crng.below((uint64_t)items.size()));
        det::Batch ba = pass ? det::make_batch_aug(items, idx, imgsz, crng, aug, true)
                             : det::make_batch(items, idx, imgsz);
        for (int b = 0; b < batch; ++b) {
          std::vector<unsigned char> rgb((size_t)imgsz * imgsz * 3);
          for (int c = 0; c < 3; ++c)
            for (int y = 0; y < imgsz; ++y)
              for (int x = 0; x < imgsz; ++x)
                rgb[((size_t)y * imgsz + x) * 3 + c] = (unsigned char)std::lround(
                    255.f * ba.x->data[((size_t)b * 3 + c) * imgsz * imgsz + (size_t)y * imgsz + x]);
          std::vector<jl::Box> got = jl::detect_plates(g, rgb.data(), imgsz, imgsz, ccfg);
          for (const auto& q : ba.gts[(size_t)b]) {
            ++gt_n;
            evd::GtBox gt{q[1], q[2], q[3], q[4], 0};
            float best = 0;
            for (const jl::Box& d : got) best = std::max(best, evd::iou_xyxy(gt, {d.x1, d.y1, d.x2, d.y2, d.score}));
            iou_sum += best;
            if (best >= 0.5f) ++hit;
          }
        }
      }
      printf("check-aug [%s]: %d labels, the shipped detector finds %.1f%% of them (mean best IoU %.3f)\n",
             pass ? "augmented" : "plain resize", gt_n, 100.0 * hit / std::max(1, gt_n),
             iou_sum / std::max(1, gt_n));
    }
    return 0;
  }

  SGD sgd(opt_params, lr0, momentum, wd, true);
  Adam adam(opt_params, lr0, 0.9f, 0.999f, 1e-8f, wd, true);
  const bool use_sgd = optim != "adam";
  Ema ema(t.params, ema_decay);
  Rng rng(seed);
  det::LossCfg cfg;

  jl::DetCfg vcfg;                       // validation runs the product path: full graph + NMS
  vcfg.kind = jl::DetKind::V8;
  vcfg.nc = 1;
  vcfg.plate_class = 0;
  vcfg.imgsz = imgsz;
  vcfg.conf = 0.001f;
  vcfg.nms = 0.45f;
  vcfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;

  std::map<std::string, std::vector<float>> best_snap;
  float best_map = -1.f;
  int best_step = 0;
  auto snapshot = [&]() {
    std::map<std::string, std::vector<float>> s;
    for (const auto& kv : t.init) s[kv.first] = kv.second->data;
    return s;
  };
  auto restore = [&](const std::map<std::string, std::vector<float>>& s) {
    for (const auto& kv : s) {
      auto it = t.init.find(kv.first);
      if (it != t.init.end()) it->second->data = kv.second;
    }
  };
  // Validation and export both want the EMA weights, not the live ones (measured everywhere in the
  // YOLO literature; the live weights at step k are noisier than their average).
  auto with_ema = [&](const std::function<void()>& fn) {
    if (!no_ema) ema.swap();
    fn();
    if (!no_ema) ema.swap();
  };
  float last_map50 = -1.f, last_map5095 = -1.f;
  auto validate = [&](int step) {
    if (valdir.empty()) return;
    with_ema([&] {
      onx::write_back(t);
      DetEval e = eval_det_dir(t.g, vcfg, valdir, val_limit, 0.25f, 0.5f, false);
      printf("  val @%d: mAP50 %.4f  mAP50-95 %.4f  P %.3f R %.3f  (%d gt over %zu frames)\n", step,
             e.r.map50, e.r.map5095, e.r.precision, e.r.recall, e.n_gt, e.frames);
      if (e.r.map5095 > best_map) {
        best_map = e.r.map5095;
        best_step = step;
        best_snap = snapshot();          // inside with_ema: this saves the averaged weights
        printf("    <- best\n");
      }
    });
    fflush(stdout);
  };

  // --- run plumbing: checkpoint / resume / early stop / csv log (pure/trainrt.hpp) ---------------
  rt::Patience pat(patience, true);                      // mAP50-95: higher is better
  rt::Log tlog;
  if (!log_p.empty()) tlog.open(log_p, "step,loss,box,cls,dfl,fg,lr,val_map50,val_map5095");
  int start_step = 0;
  auto save_ckpt = [&](int step) {
    if (ckpt_p.empty()) return;
    rt::Ckpt ck;
    ck.step = step;
    ck.total = steps;
    ck.opt_kind = use_sgd ? 1 : 2;
    ck.opt_t = use_sgd ? (sgd.started ? 1 : 0) : adam.t;
    ck.ema_updates = ema.updates;
    ck.best = best_map;
    ck.rng = rng.s;
    // every trainable tensor's weights, plus the optimizer moment(s) and the EMA shadow for the ones
    // that have them — resuming from weights alone would restart momentum at zero
    for (size_t i = 0; i < t.params.size(); ++i) {
      std::vector<float> s1, s2, sh;
      for (size_t k = 0; k < opt_params.size(); ++k)
        if (opt_params[k] == t.params[i]) {
          if (use_sgd) s1 = sgd.buf[k];
          else { s1 = adam.m[k]; s2 = adam.v[k]; }
          break;
        }
      if (!no_ema && i < ema.shadow.size()) sh = ema.shadow[i];
      ck.add(t.param_names[i], t.params[i]->data, s1, s2, sh);
    }
    if (!ck.save(ckpt_p)) printf("warn: cannot write %s\n", ckpt_p.c_str());
  };
  if (!resume_p.empty()) {
    rt::Ckpt ck;
    std::string why;
    if (!ck.load(resume_p, &why)) { printf("--resume: %s\n", why.c_str()); return 1; }
    if (ck.slots.size() != t.params.size()) {
      printf("--resume: %s has %zu tensors, this model has %zu\n", resume_p.c_str(), ck.slots.size(),
             t.params.size());
      return 1;
    }
    for (size_t i = 0; i < ck.slots.size(); ++i) {
      const rt::Slot& sl = ck.slots[i];
      if (sl.name != t.param_names[i] || sl.data.size() != t.params[i]->data.size()) {
        printf("--resume: tensor %zu is %s[%zu], checkpoint has %s[%zu]\n", i,
               t.param_names[i].c_str(), t.params[i]->data.size(), sl.name.c_str(), sl.data.size());
        return 1;
      }
      t.params[i]->data = sl.data;
      if (!no_ema && i < ema.shadow.size() && !sl.ema.empty()) ema.shadow[i] = sl.ema;
      for (size_t k = 0; k < opt_params.size(); ++k)
        if (opt_params[k] == t.params[i]) {
          if (use_sgd) { if (!sl.s1.empty()) sgd.buf[k] = sl.s1; }
          else { if (!sl.s1.empty()) adam.m[k] = sl.s1; if (!sl.s2.empty()) adam.v[k] = sl.s2; }
          break;
        }
    }
    if (ck.total != 0 && ck.total != (int64_t)steps)
      printf("warn: this checkpoint was written by a %lld-step schedule and --steps is %d — the lr "
             "curve will differ from an uninterrupted run\n", (long long)ck.total, steps);
    if (use_sgd) sgd.started = ck.opt_t != 0; else adam.t = ck.opt_t;
    ema.updates = ck.ema_updates;
    best_map = (float)ck.best;
    rng.s = ck.rng;
    start_step = (int)ck.step;
    printf("resumed %s at step %d (best mAP50-95 %.4f, optimizer and EMA state restored)\n",
           resume_p.c_str(), start_step, best_map);
    if (start_step >= steps) { printf("nothing to do: --steps %d already reached\n", steps); return 0; }
  }

  for (int step = start_step; step < steps; ++step) {
    // schedule: linear warmup (momentum too, as Ultralytics does), then linear or cosine decay to
    // lr0*lrf. Both are computed per step rather than per epoch because this trainer counts steps.
    const float x = (float)step / (float)std::max(1, steps);
    float lr = cos_lr ? lr0 * (lrf + (1 - lrf) * 0.5f * (1 + std::cos(3.14159265358979f * x)))
                      : lr0 * ((1 - x) * (1 - lrf) + lrf);
    float mom = momentum;
    if (step < warmup) {
      const float w = (float)(step + 1) / (float)warmup;
      lr *= w;
      mom = warm_mom + (momentum - warm_mom) * w;
    }
    sgd.lr = lr; sgd.momentum = mom; adam.lr = lr;

    // With fewer images than the batch size, every step sees all of them in order: that is the
    // "does this train at all" case, and a random sampler would hide the answer behind which images
    // the draw happened to pick (measured: batch 2 out of 8 frames swings the loss by 2x per step,
    // purely because frames carry 0-3 plates).
    std::vector<int> idx;
    if ((int)items.size() <= batch) for (size_t i = 0; i < items.size(); ++i) idx.push_back((int)i);
    else for (int b = 0; b < batch; ++b) idx.push_back((int)rng.below((uint64_t)items.size()));
    det::Batch ba = no_aug ? det::make_batch(items, idx, imgsz)
                           : det::make_batch_aug(items, idx, imgsz, rng, aug, step < mosaic_off_at);
    det::LossOut rep;
    std::vector<Tensor> bxs, css;
    Tensor loss = det::forward_loss(t, hn, ba.x, ba.gts, imgsz, cfg, &rep, &bxs, &css);
    if (use_sgd) sgd.zero_grad(); else adam.zero_grad();
    backward(loss);
    if (clip > 0) crn::clip_grad_norm(t.params, clip);
    if (dump_loss) printf("%d %.6f %.6f %.6f %.6f\n", step, rep.total, rep.box, rep.cls, rep.dfl);
    else printf("step %d: loss %.4f (box %.4f cls %.4f dfl %.4f) fg %d tss %.2f lr %.2e\n", step,
                rep.total, rep.box, rep.cls, rep.dfl, rep.fg, rep.tss, lr);
    fflush(stdout);
    if (!fixture.empty() && step == 0) {
      // Step 0's head tensors, gts, loss and gradients — the input tools/parity/train_det.py hands
      // to ultralytics' own v8DetectionLoss so both sides differentiate the *same* numbers.
      FILE* f = fopen(fixture.c_str(), "wb");
      if (f) {
        auto wi = [&](int32_t v) { fwrite(&v, 4, 1, f); };
        fwrite("JLPRDET1", 1, 8, f);
        wi((int32_t)bxs.size()); wi((int32_t)bxs[0]->shape[0]);
        wi((int32_t)css[0]->shape[1]); wi((int32_t)(bxs[0]->shape[1] / 4)); wi((int32_t)imgsz);
        for (size_t l = 0; l < bxs.size(); ++l) wi((int32_t)bxs[l]->shape[2]);   // square feature maps
        for (const Tensor& b : bxs) fwrite(b->data.data(), 4, (size_t)b->numel(), f);
        for (const Tensor& c : css) fwrite(c->data.data(), 4, (size_t)c->numel(), f);
        int32_t ngt = 0;
        for (const auto& v : ba.gts) ngt += (int32_t)v.size();
        wi(ngt);
        for (size_t b = 0; b < ba.gts.size(); ++b)
          for (const auto& q : ba.gts[b]) { wi((int32_t)b); fwrite(q.data(), 4, 5, f); }
        float parts[4] = {rep.total, rep.box, rep.cls, rep.dfl};
        fwrite(parts, 4, 4, f);
        for (const Tensor& b : bxs) fwrite(b->grad.data(), 4, (size_t)b->numel(), f);
        for (const Tensor& c : css) fwrite(c->grad.data(), 4, (size_t)c->numel(), f);
        fclose(f);
        printf("wrote %s (head tensors, gts, loss and gradients of step 0)\n", fixture.c_str());
      }
    }
    if (use_sgd) sgd.step(); else adam.step();
    if (!no_ema) ema.update();
    free_graph(loss);
    bool stop = false;
    if (val_every > 0 && (step + 1) % val_every == 0) {
      last_map50 = last_map5095 = -1.f;
      validate(step + 1);
      if (last_map5095 >= 0.f && pat.done(last_map5095)) {
        printf("early stop: mAP50-95 has not improved for %d validations (best %.4f at step %d)\n",
               patience, best_map, best_step);
        stop = true;
      }
    }
    tlog.row("%d,%.6f,%.6f,%.6f,%.6f,%d,%.3e,%.4f,%.4f", step, rep.total, rep.box, rep.cls, rep.dfl,
             rep.fg, lr, last_map50, last_map5095);
    if (!ckpt_p.empty() && (stop || (ckpt_every > 0 && (step + 1) % ckpt_every == 0))) {
      save_ckpt(step + 1);
      printf("  wrote %s (step %d)\n", ckpt_p.c_str(), step + 1);
    }
    if (stop_at > 0 && step + 1 >= stop_at) {
      if (!ckpt_p.empty()) { save_ckpt(step + 1); printf("  wrote %s (step %d)\n", ckpt_p.c_str(), step + 1); }
      printf("stopped at step %d of %d (--stop-at); resume with --resume %s --steps %d\n", step + 1,
             steps, ckpt_p.empty() ? "<ckpt>" : ckpt_p.c_str(), steps);
      return 0;
    }
    if (stop) { steps = step + 1; break; }
  }
  if (!ckpt_p.empty()) save_ckpt(steps);
  if (!valdir.empty() && (val_every <= 0 || steps % val_every != 0)) validate(steps);   // final

  if (!out_best.empty() && !best_snap.empty()) {
    std::map<std::string, std::vector<float>> live = snapshot();
    restore(best_snap);
    onx::write_back(t);
    onx::save_onnx(t.g, out_best);
    printf("wrote %s (best val: mAP50-95 %.4f at step %d)\n", out_best.c_str(), best_map, best_step);
    restore(live);
  }
  if (!out.empty()) {
    with_ema([&] {
      onx::write_back(t);
      onx::save_onnx(t.g, out);
    });
    printf("wrote %s%s\n", out.c_str(), no_ema ? "" : " (EMA weights)");
  }
  return 0;
}



// jlpr val --model det — the C++ side of tools/eval_det.py, plus the mAP that used to need
// Ultralytics. The metrics and their two matchings live in pure/eval_det.hpp; the work is in
// eval_det_dir() above, which the trainer's in-run validation calls too.
static int cmd_val_det(int argc, char** argv) {
  std::string data = arg_of(argc, argv, "--data", "");
  std::string det_p = arg_of(argc, argv, "--det", "models/vehicle_det.onnx");
  int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  int imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());
  float conf = (float)atof(arg_of(argc, argv, "--conf", "0.25").c_str());
  float iou_thr = (float)atof(arg_of(argc, argv, "--iou", "0.5").c_str());
  float conf_lo = (float)atof(arg_of(argc, argv, "--conf-lo", "0.001").c_str());
  bool as_json = has_flag(argc, argv, "--json");
  if (data.empty()) {
    printf("usage: vdet val --data <dir with images/ labels/> [--det onnx] "
           "[--fmt xyxy|cxcywh] [--conf f] [--iou f] [--imgsz N] [--limit N] [--json]\n");
    return 1;
  }
  jl::DetCfg cfg;
  cfg.kind = jl::DetKind::V8;
  cfg.nc = 1;
  cfg.plate_class = 0;
  cfg.imgsz = imgsz;
  cfg.conf = std::min(conf, conf_lo);        // one pass at the low threshold: NMS cannot let a
  cfg.nms = (float)atof(arg_of(argc, argv, "--nms", "0.45").c_str());   // weak box suppress a strong
  cfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;

  onx::Graph det = onx::load_onnx(det_p);
  DetEval e = eval_det_dir(det, cfg, data, limit, conf, iou_thr, !as_json);
  if (e.frames == 0) { printf("no images under %s/images\n", data.c_str()); return 1; }
  const evd::Report& r = e.r;
  if (as_json) {
    printf("{\"frames\":%zu,\"gt\":%d,\"map50\":%.6f,\"map50_95\":%.6f,\"precision\":%.6f,"
           "\"recall\":%.6f,\"f1\":%.6f,\"tp\":%d,\"fp\":%d,\"buckets\":[",
           e.frames, e.n_gt, r.map50, r.map5095, r.precision, r.recall, r.f1, r.tp, r.fp);
    for (size_t b = 0; b < e.gt_b.size(); ++b)
      printf("%s{\"lo\":%.2f,\"hi\":%.2f,\"gt\":%d,\"found\":%d}", b ? "," : "", evd::buckets()[b][0],
             evd::buckets()[b][1], e.gt_b[b], e.tp_b[b]);
    printf("],\"bucket_fp\":%d,\"empty_frames\":%d,\"fp_on_empty\":%d}\n", e.fp_bucket, e.empty_frames,
           e.fp_empty);
    return 0;
  }
  printf("\n%zu frames, conf>=%.2f, IoU>=%.2f, model=%s\n", e.frames, conf, iou_thr, det_p.c_str());
  printf("%-16s %8s %8s %8s\n", "box share", "GT", "found", "recall");
  for (size_t b = 0; b < e.gt_b.size(); ++b) {
    if (!e.gt_b[b]) continue;
    char name[32];
    snprintf(name, sizeof name, "%.0f-%.0f%%", evd::buckets()[b][0] * 100,
             std::min(1.f, evd::buckets()[b][1]) * 100);
    printf("%-16s %8d %8d %7.1f%%\n", name, e.gt_b[b], e.tp_b[b], 100.0 * e.tp_b[b] / e.gt_b[b]);
  }
  int tot_g = 0, tot_t = 0;
  for (size_t b = 0; b < e.gt_b.size(); ++b) { tot_g += e.gt_b[b]; tot_t += e.tp_b[b]; }
  printf("%-16s %8d %8d %7.1f%%\n", "all", tot_g, tot_t, 100.0 * tot_t / std::max(1, tot_g));
  printf("false positives: %d (of %d detections); on the %d empty frames: %d\n", e.fp_bucket,
         e.n_det_at_conf, e.empty_frames, e.fp_empty);
  printf("mAP50 %.4f  mAP50-95 %.4f   (P %.4f  R %.4f  F1 %.4f at conf %.2f, TP %d FP %d FN %d)\n",
         r.map50, r.map5095, r.precision, r.recall, r.f1, conf, r.tp, r.fp, r.fn);
  return 0;
}

// ---- vdet detect ---------------------------------------------------------------------------------
static void draw_rect(unsigned char* rgb, int W, int H, int x1, int y1, int x2, int y2, const unsigned char c[3], int th) {
  auto put = [&](int x, int y) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    unsigned char* p = rgb + ((size_t)y * W + x) * 3; p[0] = c[0]; p[1] = c[1]; p[2] = c[2];
  };
  for (int t = 0; t < th; ++t) {
    for (int x = x1; x <= x2; ++x) { put(x, y1 + t); put(x, y2 - t); }
    for (int y = y1; y <= y2; ++y) { put(x1 + t, y); put(x2 - t, y); }
  }
}

static int cmd_detect(int argc, char** argv) {
  std::string det_p = arg_of(argc, argv, "--det", "models/vehicle_det.onnx");
  std::string img = arg_of(argc, argv, "--img", "");
  std::string out = arg_of(argc, argv, "--out", "");
  bool as_json = has_flag(argc, argv, "--json");
  if (img.empty()) { printf("usage: vdet detect --det onnx --img file [--out png] [--conf f] [--nms f] [--imgsz N] [--json]\n"); return 1; }
  jl::DetCfg cfg;
  cfg.kind = jl::DetKind::V8;
  cfg.nc = 1;
  cfg.plate_class = 0;
  cfg.imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());   // 0 = the graph's own H x W
  cfg.conf = (float)atof(arg_of(argc, argv, "--conf", "0.25").c_str());
  cfg.nms = (float)atof(arg_of(argc, argv, "--nms", "0.45").c_str());
  cfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;
  onx::Graph g = onx::load_onnx(det_p);
  int W = 0, H = 0, C = 0;
  std::vector<unsigned char> blob = trn::read_file(img);
  unsigned char* im = blob.empty() ? nullptr : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
  if (!im) { printf("cannot read %s\n", img.c_str()); return 1; }
  std::vector<jl::Box> boxes = jl::detect_plates(g, im, W, H, cfg);
  if (as_json) {
    printf("{\"count\":%zu,\"boxes\":[", boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
      printf("%s[%.1f,%.1f,%.1f,%.1f,%.4f]", i ? "," : "", boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].score);
    printf("]}\n");
  } else {
    printf("%s: %d x %d, %zu vehicles (conf >= %.2f)\n", img.c_str(), W, H, boxes.size(), cfg.conf);
  }
  if (!out.empty()) {
    const unsigned char col[3] = {255, 220, 0};
    for (const jl::Box& b : boxes) draw_rect(im, W, H, (int)b.x1, (int)b.y1, (int)b.x2, (int)b.y2, col, 2);
    stbi_write_png(out.c_str(), W, H, 3, im, W * 3);
  }
  stbi_image_free(im);
  return 0;
}

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "detect") return cmd_detect(argc, argv);
  if (cmd == "val") return cmd_val_det(argc, argv);
  if (cmd == "train") return cmd_train_det(argc, argv);
  printf("usage: vdet detect|val|train ...\n");
  return 2;
}
