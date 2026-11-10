"""C++（vdet detect）と Python（detect.py）の推論が同じ箱を出すかを、画像の集まりで調べる。

    python parity_detect.py --vdet build/Release/vdet.exe --det models/vehicle_det.onnx dataset/images/val/*.jpg

箱は順番ではなく位置で対応づける（点数が同じ箱は並び順が実装で変わりうる）。
報告: 台数が一致した枚数、対応した箱の座標差の最大、点数差の最大。
"""
import argparse
import glob
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from detect import Detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--vdet", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    import cv2
    files = sorted(sum([glob.glob(p) for p in a.images], []))
    py = Detector(a.det, a.conf)
    same_n, n_box, diffs, sdiffs, lone = 0, 0, [], [], []

    def iou(a, b):
        iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return 0.0
        i = iw * ih
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

    for f in files:
        c = json.loads(subprocess.check_output([a.vdet, "detect", "--det", a.det, "--img", f, "--conf", str(a.conf), "--json"], text=True))
        A = np.array(c["boxes"], dtype=np.float64).reshape(-1, 5)
        B = np.array(py.detect(cv2.imread(f)[:, :, ::-1].copy()), dtype=np.float64).reshape(-1, 5)
        same_n += len(A) == len(B)
        n_box += len(A)
        used = set()
        for b in A:
            best, j = 0.0, -1
            for k, q in enumerate(B):
                if k in used:
                    continue
                v = iou(b, q)
                if v > best:
                    best, j = v, k
            if best >= 0.5:
                used.add(j)
                diffs.append(float(np.abs(B[j, :4] - b[:4]).max()))
                sdiffs.append(abs(float(B[j, 4]) - b[4]))
            else:
                lone.append(("C++", float(b[4])))
        for k, q in enumerate(B):
            if k not in used:
                lone.append(("Py", float(q[4])))
    d = np.array(diffs); sd = np.array(sdiffs)
    print(f"{len(files)} images, {n_box} boxes (C++): same count in {same_n}/{len(files)}")
    print(f"matched {len(d)} boxes (IoU>=0.5): box diff median {np.median(d):.3f} / p99 {np.percentile(d, 99):.3f} / max {d.max():.3f} px, "
          f"score diff max {sd.max():.4f}")
    print(f"unmatched {len(lone)}: scores " + ", ".join(f"{w}:{s:.3f}" for w, s in sorted(lone, key=lambda t: t[1])[:20]))


if __name__ == "__main__":
    main()
