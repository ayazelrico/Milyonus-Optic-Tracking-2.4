---
language:
- en
license: mit
library_name: transformers
tags:
- computer-vision
- face-tracking
- optic-tracking
- landmarks
- focus-tracking
- mediapipe
- real-time
- on-device
- multi-face
- head-pose
- blendshapes
pipeline_tag: image-feature-extraction
model-index:
- name: Milyonus-Optic-Tracking-2.4
  results: []
---

# Milyonus Optic Tracking 2.4

**Ultra-detailed multi-task facial optic tracking model**

Real-time facial landmark, head pose, blendshape, focus/attention, and multi-face tracking model. Designed as a full-stack alternative / complement to MediaPipe Face Landmarker, with ownership of the entire training and inference pipeline.

## Model Details

- **Architecture**: Hybrid multi-scale CNN stem + 12-layer ViT spatial transformer + temporal transformer + multi-task heads
- **Parameters**: ~116.7M
- **Tasks**:
  - 478-point 3D facial landmarks
  - 52 ARKit-style blendshapes
  - 3D head pose (yaw/pitch/roll + translation)
  - Continuous focus score (0–1)
  - Expression classification (8 classes)
  - Multi-face track embedding
- **Framework**: PyTorch + Transformers
- **Status**: Architecture + initialized weights available. Ready for fine-tuning on custom video/screenshot data.

## Intended Use

- Real-time on-device or server-side facial optic tracking
- Fine-tuning on proprietary interview / attention / focus datasets
- Replacement or augmentation of MediaPipe graphs when full ownership of weights is required
- Research on multi-task face understanding

## Limitations

- Current weights are randomly initialized (not yet trained on real data).
- Performance on real images will be poor until fine-tuned.
- No age/gender estimation (by design, consistent with MediaPipe Face Landmarker).

## Training Data

Intended training data: MP4, MOV videos and screenshots. Labels generated via MediaPipe Face Landmarker for consistency with the browser application.

See `scripts/dataset_video.py` and `scripts/train_finetune.py`.

## Quick Start

```python
from transformers import AutoConfig
from modeling_optic_tracking import MilyonusOpticTrackingModel
from safetensors.torch import load_file
import torch

config = AutoConfig.from_pretrained("ayazelrico/Milyonus-Optic-Tracking-2.4")
model = MilyonusOpticTrackingModel(config)

state = load_file("model.safetensors")
model.load_state_dict(state, strict=False)
model.eval()

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out = model(x)
print(out.keys())
```

## Citation

```
@misc{milyonus-optic-tracking-2.4,
  title={Milyonus Optic Tracking 2.4},
  author={Ayaz Elrico},
  year={2026},
  publisher={Hugging Face}
}
```
