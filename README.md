# vehicle_det_cpp — ライブ映像から自分で教師を作る車両検出器

道路のライブカメラの映像だけを材料に、車両検出器を学習します。

1. **教師データを作る**: COCO 学習済みの大きいモデル（yolo11x）を、録画から抜き出したフレームに走らせて枠を付けます。
2. **生徒を学習する**: その枠で、小さい yolov8n（1 クラス `vehicle`）を学習します。
3. **推論する**: C++・Python・ブラウザ（WASM）のどれでも、同じ ONNX で同じ枠が出ます。推論の依存はありません（自前の ONNX ランナと stb）。

**デモ**: https://yomei-o.github.io/vehicle_det_cpp/ （首都高 竹橋JCT の 60 秒の録画入り。手元の動画・画像、画面共有でも動きます）

渋滞の判別（[traffic_jam_cpp](https://github.com/yomei-o/traffic_jam_cpp)）で「台数」と「車頭間隔」を数えるのに使っています。

## 結果

学習データは、首都高のライブカメラ 5 か所の録画から 0.5 秒おきに抜き出した 923 枚です（呉服橋・葛西・竹橋は 2 分、浜崎橋・宝町は 50 秒）。
各録画の最後の 2 割は検証用にしました。時間で分けているので、同じ車が学習と検証の両方に入ることはほとんどありません。
枠は教師（yolo11x、入力 1280、信頼度 0.3 以上、短辺 6px 以上）が付けたもので、約 16,000 個あります。

| | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| 生徒 yolov8n 960（Python・ultralytics の評価） | 0.896 | 0.774 | 0.867 | 0.835 |
| 同じ ONNX を C++ で評価（`vdet val`、960×544 にそのまま縮める、信頼度 0.25 の P/R） | 0.891 | 0.765 | 0.788 | 0.884 |
| 軽い版 640×352（同じ重み、C++ で評価） | 0.873 | 0.701 | 0.791 | 0.862 |

正解は教師の枠なので、この数字は「教師にどれだけ近づけたか」を表します。教師自身の見落としは測っていません。

## C++ と Python が同じであること

| 何を | 結果 | 道具 |
|---|---|---|
| 推論（同じ ONNX・同じ画像 120 枚） | 対応した枠 2,298 個で、座標の差は中央値 0.05px・99% 点 0.4px。最大は 1 個だけ 10px（重なった枠のどちらを残すかが分かれた）。片方だけに出た枠は 3 個で、すべて点数がしきい値ぎりぎり（0.250〜0.252） | `tools/parity_detect.py` |
| 学習の損失と勾配（同じ 1 ステップの head の出力） | ultralytics 8.4.104 と、損失の差 8e-7、勾配の差 2e-5 | `vdet train --dump-fixture` → `tools/parity_train_det.py` |
| ブラウザ（WASM）とネイティブ | 同じ画像で台数・枠が一致（31 台）。速さは 960×544 で 1 フレーム約 1 秒、640×352 で約 0.4 秒（Node、2 回目以降） | `wasm/test_node.js` |

**ultralytics の版に注意**: C++ の損失は ultralytics **8.4.104** に合わせてあります。8.4.208 とは損失が 1% ずれ、勾配はまったく合いません（この間に損失の計算が変わった）。
Python 側の学習も 8.4.104 に固定しています（`requirements.txt`）。

## 使い方

### 1. 録画から教師データを作る（Python、GPU があれば速い）

```sh
python tools/make_teacher.py --out dataset rec/*.ts         # 0.5 秒おき、各録画の最後の 2 割を検証用に
```

録画は、traffic_jam_cpp の手順（yt-dlp + ffmpeg、配信の少し前から読むと止まりにくい）で作った `.ts` か `.mp4` です。

### 2. 生徒を学習する

Python（ultralytics 8.4.104、GPU）:

```sh
python tools/train_det.py --data dataset/data.yaml --imgsz 960 --epochs 60 --out models/vehicle_det.onnx
python tools/train_det.py --export-only <best.pt> --export-hw 352,640 --out models/vehicle_det_640.onnx   # 軽い版
```

C++（ONNX グラフをそのまま学習。CPU）:

```sh
vdet train --init models/vehicle_det.onnx --data dataset/train --val dataset/val --imgsz 640 --batch 4 \
           --epochs 1 --lr 0.0002 --clip 10 --export out.onnx
```

学習済みの重みから続けて学習するときは、学習率を下げて勾配クリップを付けてください。
`--lr 0.001` のままだと、数ステップで損失が発散しました（実測、クリップなし）。

### 3. 推論・評価

```sh
vdet detect --det models/vehicle_det.onnx --img a.jpg --out a_det.png        # C++
python tools/detect.py --det models/vehicle_det.onnx --img a.jpg --out a.png  # Python（onnxruntime）
vdet val --det models/vehicle_det.onnx --data dataset/val                     # mAP（C++）
```

他のプログラムからは `vehicle_det.hpp` だけを使います（`vd::Detector`）。

## ビルド

```sh
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target vdet
sh wasm/build.sh            # ブラウザ用（docs/vdet.js, docs/vdet.wasm）。emsdk が要る
```

## ファイル

- `vehicle_det.hpp` — 他のプログラムから使う入口（`vd::Detector`）
- `vdet.cpp` — CLI（detect / val / train）
- `pure/` — 推論・学習のエンジン（姉妹リポ [yolo_lpr_cpp](https://github.com/yomei-o/yolo_lpr_cpp) の `pure/` から必要なものを持ってきた）
- `tools/` — 教師データ作り、Python の学習・推論、C++ との照合
- `wasm/` — ブラウザ用の入口とビルド
- `docs/` — デモ（GitHub Pages）。モデルと録画を同梱
- `models/` — 学習済みの ONNX

## ライセンス

- 自分のコード（`pure/` のエンジン、`vdet.cpp`、`tools/`、`wasm/`）: BSD-3-Clause
- **モデルの重み**: 生徒は Ultralytics の `yolov8n.pt` から学習しているので **AGPL-3.0** を引き継ぎます。教師も Ultralytics のモデル（yolo11x、AGPL-3.0）です。
  ランダム初期化から学習する（`train_det.py --init scratch`）と、Ultralytics の重みには依存しません。ただし、教師データが少ないので精度は落ちます。
- Eigen（`pure/third_party/eigen_flat/`）: MPL-2.0。stb: MIT / パブリックドメイン
- 同梱の録画（`docs/takebashi_60s.mp4`）は、首都高の公開ライブカメラの映像です。
