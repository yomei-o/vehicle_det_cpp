"""教師データで生徒（小さい yolov8n、1 クラス vehicle）を学習し、ONNX に書き出す。

    python train_det.py --data dataset/data.yaml --imgsz 960 --epochs 60 --out models/vehicle_det.onnx
    python train_det.py --export-only runs/detect/runs/vehicle/weights/best.pt --out models/vehicle_det.onnx   # 書き出しだけ

出発点は COCO 学習済みの yolov8n.pt（少ないデータで遠くの小さい車まで拾うため。Ultralytics の重みなので AGPL-3.0）。
--init scratch でランダム初期化からも学習できる（ライセンスを気にする場合。データが少ないので精度は落ちる）。
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--init", default="yolov8n.pt", help="yolov8n.pt または scratch")
    ap.add_argument("--out", default="models/vehicle_det.onnx")
    ap.add_argument("--export-hw", default="544,960", help="ONNX の入力 H,W（1280x720 を縦横比そのままで入れる）")
    ap.add_argument("--export-only", default="", help="学習せず、この .pt を書き出すだけ")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="vehicle")
    a = ap.parse_args()

    from ultralytics import YOLO
    hw = [int(v) for v in a.export_hw.split(",")]
    if a.export_only:
        onnx = YOLO(a.export_only).export(format="onnx", imgsz=hw, opset=12, simplify=True, dynamic=False)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        shutil.copy(onnx, a.out)
        print("saved", a.out)
        return
    model = YOLO("yolov8n.yaml") if a.init == "scratch" else YOLO(a.init)
    model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, project=a.project, name=a.name, exist_ok=True,
                single_cls=True, fliplr=0.5, mosaic=1.0, close_mosaic=10, seed=0, deterministic=True, plots=False, verbose=False)
    best = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")  # 版によって project の下に detect/ が挟まる
    m = YOLO(best)
    metrics = m.val(data=a.data, imgsz=a.imgsz, verbose=False)
    print(f"val mAP50 {metrics.box.map50:.4f} mAP50-95 {metrics.box.map:.4f} P {metrics.box.mp:.4f} R {metrics.box.mr:.4f}")
    onnx = m.export(format="onnx", imgsz=hw, opset=12, simplify=True, dynamic=False)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    shutil.copy(onnx, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
