"""vdet detect の Python 版（onnxruntime）。前処理・デコード・NMS を C++ と同じにしてあるので、同じ ONNX なら同じ箱が出る。

    python detect.py --det models/vehicle_det.onnx --img a.jpg [--out a_det.png] [--conf 0.25] [--json]

前処理: グラフが宣言する H x W へ双線形でそのまま縮める（letterbox なし）、RGB、0-1。
デコード: 出力 [1, 5, N] = cx, cy, w, h, score（score は sigmoid 済み）。スコア降順の NMS（IoU 0.45）。
"""
import argparse
import json

import numpy as np


def resize_rgb01(rgb, iw, ih):
    """pure/crop.hpp の resize_rgb01 と同じ（画素中心合わせの双線形、端は複製）。"""
    H, W, _ = rgb.shape
    sx = (np.arange(iw) + 0.5) * W / iw - 0.5
    sy = (np.arange(ih) + 0.5) * H / ih - 0.5
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    fx = (sx - x0).astype(np.float32); fy = (sy - y0).astype(np.float32)
    xa = np.clip(x0, 0, W - 1); xb = np.clip(x0 + 1, 0, W - 1)
    ya = np.clip(y0, 0, H - 1); yb = np.clip(y0 + 1, 0, H - 1)
    f = rgb.astype(np.float32)
    top = f[ya][:, xa] * (1 - fx)[None, :, None] + f[ya][:, xb] * fx[None, :, None]
    bot = f[yb][:, xa] * (1 - fx)[None, :, None] + f[yb][:, xb] * fx[None, :, None]
    out = top * (1 - fy)[:, None, None] + bot * fy[:, None, None]
    return (out / 255.0).transpose(2, 0, 1)[None].astype(np.float32)


def nms(boxes, scores, thr):
    order = np.argsort(-scores, kind="stable")
    keep, dead = [], np.zeros(len(order), bool)
    for i, a in enumerate(order):
        if dead[i]:
            continue
        keep.append(a)
        for j in range(i + 1, len(order)):
            if dead[j]:
                continue
            b = order[j]
            iw = min(boxes[a, 2], boxes[b, 2]) - max(boxes[a, 0], boxes[b, 0])
            ih = min(boxes[a, 3], boxes[b, 3]) - max(boxes[a, 1], boxes[b, 1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            ua = (boxes[a, 2] - boxes[a, 0]) * (boxes[a, 3] - boxes[a, 1]) + (boxes[b, 2] - boxes[b, 0]) * (boxes[b, 3] - boxes[b, 1]) - inter
            if inter / ua > thr:
                dead[j] = True
    return keep


class Detector:
    def __init__(self, path, conf=0.25, nms_thr=0.45):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        i = self.s.get_inputs()[0]
        self.name = i.name
        self.ih, self.iw = int(i.shape[2]), int(i.shape[3])
        self.conf, self.nms = conf, nms_thr

    def detect(self, rgb):
        H, W, _ = rgb.shape
        raw = self.s.run(None, {self.name: resize_rgb01(rgb, self.iw, self.ih)})[0][0]  # (5, N)
        sc = raw[4]
        m = sc >= self.conf
        cx, cy, w, h = raw[0][m], raw[1][m], raw[2][m], raw[3][m]
        b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        s = sc[m]
        keep = nms(b, s, self.nms)
        sx, sy = W / self.iw, H / self.ih
        out = []
        for k in keep:
            x1 = float(np.clip(b[k, 0] * sx, 0, W - 1)); x2 = float(np.clip(b[k, 2] * sx, 0, W - 1))
            y1 = float(np.clip(b[k, 1] * sy, 0, H - 1)); y2 = float(np.clip(b[k, 3] * sy, 0, H - 1))
            out.append((x1, y1, x2, y2, float(s[k])))
        out.sort(key=lambda t: -t[4])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="models/vehicle_det.onnx")
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    import cv2
    rgb = cv2.imread(a.img)[:, :, ::-1].copy()
    boxes = Detector(a.det, a.conf, a.nms).detect(rgb)
    if a.json:
        print(json.dumps({"count": len(boxes), "boxes": [[round(v, 1) for v in b[:4]] + [round(b[4], 4)] for b in boxes]}))
    else:
        print(f"{a.img}: {rgb.shape[1]} x {rgb.shape[0]}, {len(boxes)} vehicles (conf >= {a.conf:.2f})")
    if a.out:
        im = rgb[:, :, ::-1].copy()
        for x1, y1, x2, y2, _ in boxes:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)
        cv2.imwrite(a.out, im)


if __name__ == "__main__":
    main()
