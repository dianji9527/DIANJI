import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model for pet behavior recognition.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--project", default="runs/train", help="Training output directory.")
    parser.add_argument("--name", default="pet_behavior_yolo", help="Experiment name.")
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
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
