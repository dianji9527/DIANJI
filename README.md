# Zhichong AI Model Starter

This repository is the first AI model code base for the Zhichong smart pet monitoring project.

It focuses on two model directions:

1. Vision model: YOLO for pet detection and behavior recognition.
2. Audio model: YAMNet for pet sound event recognition.

The frontend is not included here. This repository is prepared for model training, testing, exporting, and later integration with a FastAPI backend.

## Project Goal

The final product should support:

- Pet video/image analysis.
- Pet behavior detection, such as standing, lying, eating, drinking, pacing, or abnormal behavior.
- Pet sound recognition, such as barking, howling, whimpering, meowing, or other abnormal sounds.
- Model export for backend deployment.
- Clean GitHub structure for future collaboration.

## Files

```text
.
├── train.py          # Train YOLO vision model
├── test.py           # Test YOLO image/video and YAMNet audio inference
├── export.py         # Export YOLO model to ONNX/TorchScript/etc.
├── requirements.txt  # Python dependencies
├── .gitignore        # Ignore model weights, datasets, cache files
├── LICENSE           # MIT license
└── README.md         # Project documentation
```

## Environment

Recommended Python version:

```text
Python 3.10 or Python 3.11
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation for YOLO

YOLO training requires a dataset configuration file, usually named `data.yaml`.

Example:

```yaml
path: datasets/pet_behavior
train: images/train
val: images/val
test: images/test

names:
  0: cat
  1: dog
  2: standing
  3: lying
  4: eating
  5: drinking
  6: pacing
  7: abnormal
```

Recommended dataset structure:

```text
datasets/pet_behavior/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Do not upload the dataset to GitHub unless it is small and public. Large datasets should be stored separately.

## Train YOLO

Basic training:

```bash
python train.py --data datasets/pet_behavior/data.yaml --model yolov8n.pt --epochs 50
```

Useful options:

```bash
python train.py \
  --data datasets/pet_behavior/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --project runs/train \
  --name pet_behavior_yolo
```

After training, the best model is usually saved at:

```text
runs/train/pet_behavior_yolo/weights/best.pt
```

## Test YOLO on Image or Video

Image:

```bash
python test.py --task vision --model runs/train/pet_behavior_yolo/weights/best.pt --source samples/pet.jpg
```

Video:

```bash
python test.py --task vision --model runs/train/pet_behavior_yolo/weights/best.pt --source samples/pet.mp4
```

The prediction results will be saved under:

```text
runs/predict/
```

## Test YAMNet Audio

YAMNet is a pretrained audio event classification model. It can be used first as a baseline for pet sound recognition.

```bash
python test.py --task audio --source samples/bark.wav
```

The script prints top audio classes with confidence scores.

Important note:

YAMNet is not a pet-specific model. It is useful for a first prototype, but later you may need a custom pet sound classifier if you want better accuracy for barking, whining, howling, or distress sounds.

## Export YOLO Model

Export to ONNX:

```bash
python export.py --model runs/train/pet_behavior_yolo/weights/best.pt --format onnx
```

Export to TorchScript:

```bash
python export.py --model runs/train/pet_behavior_yolo/weights/best.pt --format torchscript
```

Common export formats:

```text
onnx
torchscript
openvino
engine
tflite
```

For FastAPI backend deployment, ONNX is usually a practical first choice.

## GitHub Upload

Initialize Git:

```bash
git init
git add .
git commit -m "Initial AI model starter code"
```

Create a GitHub repository named:

```text
zhichong-ai-model-starter
```

Then connect and push:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/zhichong-ai-model-starter.git
git push -u origin main
```

## What Should Not Be Uploaded

The following files are ignored by `.gitignore`:

- Virtual environments.
- Python cache files.
- Datasets.
- Training outputs.
- Model weights.
- Audio/video/image samples.
- Local environment files.

This keeps the GitHub repository clean and lightweight.

## Next Development Steps

Recommended next work:

1. Add a real pet behavior dataset.
2. Train a YOLO model for pet behavior labels.
3. Collect pet audio samples.
4. Compare YAMNet baseline results with real pet audio.
5. Add FastAPI endpoints for image, video, and audio inference.
6. Add abnormal behavior rules, such as "static over 30 minutes" or "frequent barking".

## License

This project uses the MIT License. See `LICENSE` for details.
