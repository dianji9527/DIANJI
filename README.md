# Zhichong AI Models

智宠项目的视觉与听觉 AI 模型仓库。

本仓库只负责模型侧能力：

- 视觉模型：YOLO，用于宠物检测与行为识别
- 听觉模型：YAMNet，用于宠物声音事件识别
- 统一输出：JSON，方便后续 FastAPI 后端直接调用

前端不直接接入本仓库。后续架构应为：

```text
前端 App / Web
  -> FastAPI 后端
  -> zhichong-ai-models
  -> YOLO / YAMNet
```

## 目录结构

```text
zhichong-ai-models/
  configs/
    default.yaml
  scripts/
    demo_audio.py
    demo_vision.py
  src/
    zhichong_ai/
      audio/
        yamnet_service.py
      common/
        config.py
        schemas.py
      vision/
        yolo_service.py
  tests/
    test_schemas.py
  .gitignore
  pyproject.toml
```

## 环境要求

- Python 3.10 或 3.11
- 建议使用虚拟环境
- GPU 不是必须，但 YOLO 视频推理建议使用 GPU

安装：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

如果只运行视觉模型：

```bash
pip install ultralytics opencv-python pydantic pyyaml
```

如果只运行听觉模型：

```bash
pip install tensorflow tensorflow-hub librosa soundfile numpy pydantic pyyaml
```

## 模型文件

YOLO 权重文件不要提交到 GitHub。建议放在：

```text
models/yolo/pet_behavior.pt
```

如果你的文件名不同，修改：

```text
configs/default.yaml
```

YAMNet 默认从 TensorFlow Hub 加载：

```text
https://tfhub.dev/google/yamnet/1
```

如果运行环境不能联网，可以提前下载 YAMNet SavedModel，并把 `audio.yamnet_model_url` 改成本地路径。

## 运行 YOLO 视觉识别

图片：

```bash
python scripts/demo_vision.py --image data/samples/pet.jpg
```

视频抽帧：

```bash
python scripts/demo_vision.py --video data/samples/pet.mp4 --every-n-frames 15
```

输出示例：

```json
{
  "source": "data/samples/pet.jpg",
  "detections": [
    {
      "label": "dog",
      "confidence": 0.91,
      "box": [102.4, 88.1, 420.7, 360.2]
    }
  ]
}
```

## 运行 YAMNet 听觉识别

```bash
python scripts/demo_audio.py --audio data/samples/bark.wav
```

输出示例：

```json
{
  "source": "data/samples/bark.wav",
  "events": [
    {
      "label": "Bark",
      "confidence": 0.82,
      "start_sec": 0.0,
      "end_sec": 0.96
    }
  ]
}
```

## 后续接入 FastAPI 的方式

FastAPI 后端可以直接导入服务类：

```python
from zhichong_ai.common.config import load_config
from zhichong_ai.vision.yolo_service import YoloVisionService
from zhichong_ai.audio.yamnet_service import YamnetAudioService

config = load_config("configs/default.yaml")
yolo = YoloVisionService(config.vision)
yamnet = YamnetAudioService(config.audio)
```

后端接口建议保持模型无关：

```text
POST /api/ai/vision/analyze-image
POST /api/ai/vision/analyze-video
POST /api/ai/audio/analyze
```

这样以后更换 YOLO 权重或替换音频模型时，前端接口不需要变化。

## GitHub 初始化

```bash
git init
git add .
git commit -m "Initial YOLO and YAMNet model services"
git branch -M main
git remote add origin https://github.com/<your-name>/zhichong-ai-models.git
git push -u origin main
```

## 不要提交的内容

- `.venv/`
- `models/`
- `data/`
- `datasets/`
- `runs/`
- `.pt`、`.onnx`、`.engine`
- `.mp4`、`.wav`、`.mp3`

这些内容已经写入 `.gitignore`。
