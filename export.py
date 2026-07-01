import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLO model for deployment.")
    parser.add_argument("--model", required=True, help="Path to trained YOLO model, for example best.pt.")
    parser.add_argument(
        "--format",
        default="onnx",
        help="Export format: onnx, torchscript, openvino, engine, tflite, etc.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--device", default=None, help="Device, for example 0, cpu, or cuda:0.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ultralytics. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
