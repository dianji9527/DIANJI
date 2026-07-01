import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Test YOLO vision or YAMNet audio inference.")
    parser.add_argument("--task", choices=["vision", "audio"], required=True)
    parser.add_argument("--source", required=True, help="Image/video/audio file path.")
    parser.add_argument("--model", default=None, help="YOLO model path for vision task.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size.")
    parser.add_argument("--top-k", type=int, default=10, help="Top K audio classes for YAMNet.")
    return parser.parse_args()


def test_vision(args):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ultralytics. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    if not args.model:
        raise ValueError("--model is required when --task vision")

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project="runs/predict",
        name="vision",
    )

    for result in results:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = names.get(class_id, str(class_id))
            confidence = float(box.conf[0].item())
            xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            print({"label": label, "confidence": round(confidence, 4), "box": xyxy})


def test_audio(args):
    try:
        import librosa
        import numpy as np
        import tensorflow_hub as hub
    except ImportError as exc:
        raise RuntimeError(
            "Missing audio dependencies. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    model = hub.load("https://tfhub.dev/google/yamnet/1")
    waveform, _ = librosa.load(args.source, sr=16000, mono=True)
    scores, _, _ = model(waveform.astype(np.float32))

    mean_scores = np.mean(scores.numpy(), axis=0)
    top_indices = np.argsort(mean_scores)[::-1][: args.top_k]

    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        class_names = [row["display_name"] for row in reader]

    for index in top_indices:
        label = class_names[index] if index < len(class_names) else str(index)
        confidence = float(mean_scores[index])
        print({"label": label, "confidence": round(confidence, 4)})


def main():
    args = parse_args()

    if args.task == "vision":
        test_vision(args)
    else:
        test_audio(args)


if __name__ == "__main__":
    main()
