"""ライブ映像の録画から、車両検出器の教師データ（YOLO 形式）を自動で作る。

教師は COCO で学習済みの大きいモデル（既定 yolo11x、入力 1280）。car / motorcycle / bus / truck を
1 クラス "vehicle" にまとめる（教師は乗用車とトラックの区別がときどき揺れるが、数えるには要らない）。

    python make_teacher.py --out dataset rec/*.ts rec/*.mp4 [--step 15] [--val 0.2]

- 各動画から step フレームおきに抜き出す（30fps で 15 なら 0.5 秒おき）
- 各動画の最後の val 割を検証用にする（時間で分けるので、同じ車が学習と検証にまたがりにくい）
- 教師の信頼度が conf 未満、または短辺 min_px 未満の箱は捨てる
- 出力: dataset/{train,val}/images/*.jpg, dataset/{train,val}/labels/*.txt（vdet val/train もこの形を読む）,
        dataset/data.yaml, dataset/teacher_<最初の動画名>.json（箱の数など）
"""
import argparse
import json
import os
import sys

import av
import cv2

VEHICLE = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def teacher_boxes(model, img, imgsz, conf, tiles, overlap=0.2):
    """教師の箱（xyxy, 画素）。tiles > 1 なら、全体に加えて tiles x tiles の重なりのあるタイルも
    同じ入力の大きさで見せる（＝拡大して見せる）。遠くの小さい車・夜の車列の見落としが減る。
    重なった箱は IoU 0.5 でまとめる（点数の高い方を残す、ultralytics の NMS と同じ考え方）。"""
    import numpy as np
    import torch
    from torchvision.ops import nms
    h, w = img.shape[:2]
    boxes, scores = [], []
    r = model.predict(img, imgsz=imgsz, conf=conf, classes=list(VEHICLE), verbose=False)[0]
    boxes.append(r.boxes.xyxy.cpu().numpy()); scores.append(r.boxes.conf.cpu().numpy())
    if tiles > 1:
        tw, th = int(w / (tiles - (tiles - 1) * overlap)), int(h / (tiles - (tiles - 1) * overlap))
        for ty in range(tiles):
            for tx in range(tiles):
                x0 = int(round(tx * (w - tw) / (tiles - 1))); y0 = int(round(ty * (h - th) / (tiles - 1)))
                crop = img[y0:y0 + th, x0:x0 + tw]
                r = model.predict(crop, imgsz=imgsz, conf=conf, classes=list(VEHICLE), verbose=False)[0]
                b = r.boxes.xyxy.cpu().numpy().copy()
                if len(b) == 0:
                    continue
                # タイルの端で切れた箱は捨てる（隣のタイルか全体の方で丸ごと見えている）
                keep = ((b[:, 0] > 2) | (x0 == 0)) & ((b[:, 2] < tw - 2) | (x0 + tw >= w)) & \
                       ((b[:, 1] > 2) | (y0 == 0)) & ((b[:, 3] < th - 2) | (y0 + th >= h))
                b = b[keep]; s = r.boxes.conf.cpu().numpy()[keep]
                b[:, [0, 2]] += x0; b[:, [1, 3]] += y0
                boxes.append(b); scores.append(s)
    B = np.concatenate(boxes) if boxes else np.zeros((0, 4)); S = np.concatenate(scores) if scores else np.zeros(0)
    if len(B) == 0:
        return B
    k = nms(torch.tensor(B, dtype=torch.float32), torch.tensor(S, dtype=torch.float32), 0.5).numpy()
    return B[k]


def frames_of(path, step):
    c = av.open(path)
    for i, fr in enumerate(c.decode(video=0)):
        if i % step == 0:
            yield i, fr.to_ndarray(format="bgr24")
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--teacher", default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-px", type=float, default=6)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--tiles", type=int, default=1, help="2 なら 2x2 のタイルも拡大して見せる（遠くの車・夜の車列に効く）")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.teacher)
    for s in ("train", "val"):
        os.makedirs(os.path.join(a.out, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(a.out, s, "labels"), exist_ok=True)

    stats = {"videos": {}, "boxes": 0, "images": {"train": 0, "val": 0}, "teacher": a.teacher, "imgsz": a.imgsz, "conf": a.conf}
    for v in a.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        fr = list(frames_of(v, a.step))
        if not fr:
            print(f"{v}: no frames", file=sys.stderr)
            continue
        n_val = int(round(len(fr) * a.val))
        nb = 0
        for k, (i, img) in enumerate(fr):
            split = "val" if k >= len(fr) - n_val else "train"
            h, w = img.shape[:2]
            lines = []
            for x0, y0, x1, y1 in teacher_boxes(model, img, a.imgsz, a.conf, a.tiles).tolist():
                if min(x1 - x0, y1 - y0) < a.min_px:
                    continue
                lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            stem = f"{name}_{i:05d}"
            cv2.imwrite(os.path.join(a.out, split, "images", stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(a.out, split, "labels", stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            nb += len(lines)
            stats["images"][split] += 1
        stats["videos"][name] = {"frames": len(fr), "val": n_val, "boxes": nb}
        stats["boxes"] += nb
        print(f"{name}: {len(fr)} frames ({n_val} val), {nb} boxes", flush=True)

    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: train/images\nval: val/images\nnames:\n  0: vehicle\n")
    # 何度かに分けて足していけるよう、記録は呼び出しごとに別ファイル
    first = os.path.splitext(os.path.basename(a.videos[0]))[0]
    with open(os.path.join(a.out, f"teacher_{first}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats["images"]), stats["boxes"], "boxes")


if __name__ == "__main__":
    main()
