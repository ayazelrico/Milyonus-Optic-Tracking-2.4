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

Real-time facial landmark, head pose, blendshape, focus/attention, and multi-face tracking model. Designed as a full-stack alternative or complement to MediaPipe Face Landmarker, with full ownership of the training and inference pipeline.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Hybrid multi-scale CNN stem + 12-layer ViT spatial transformer + temporal transformer + multi-task heads |
| Parameters | ~116.7M |
| Tasks | 478-point 3D landmarks, 52 ARKit blendshapes, 3D head pose, continuous focus score, 8-class expression, multi-face track embedding |
| Framework | PyTorch + Transformers |
| Weights status | Initialized (`model.safetensors`). Ready for fine-tuning. |
| Input | RGB image (recommended 256×256) |
| Output | Landmarks, pose, blendshapes, focus score |

## Intended Use

- Real-time on-device or server-side facial optic tracking
- Fine-tuning on proprietary interview, attention, or focus datasets (MP4, MOV, screenshots)
- Replacement or augmentation of MediaPipe graphs when full weight ownership is required
- Research on multi-task face understanding

## Training Infrastructure

The repository includes a complete fine-tuning stack:

- `scripts/dataset_video.py` — loads MP4/MOV/screenshots and generates labels with MediaPipe Face Landmarker
- `scripts/train_finetune.py` — production training loop (multi-task loss, cosine scheduler, gradient clipping, resume support)
- `scripts/inference.py` — loading and inference example
- `scripts/evaluate.py` — evaluation skeleton (landmark NME, focus MAE, blendshape correlation)

### Quick training command

```bash
python scripts/train_finetune.py \
  --data /path/to/videos_and_screenshots \
  --epochs 10 \
  --batch_size 4 \
  --resume model.safetensors \
  --output finetuned_model.safetensors
```

## Limitations

- Current weights are initialized, not yet trained on real data. Performance on real images will be poor until fine-tuned.
- No age or gender estimation (by design, consistent with MediaPipe Face Landmarker).
- Requires MediaPipe Face Landmarker task file for automatic label generation during training.

## Quick Start (inference)

```python
from configuration_optic_tracking import MilyonusOpticTrackingConfig
from modeling_optic_tracking import MilyonusOpticTrackingModel
from safetensors.torch import load_file
import torch

config = MilyonusOpticTrackingConfig.from_pretrained("ayazelrico/Milyonus-Optic-Tracking-2.4")
model = MilyonusOpticTrackingModel(config)
state = load_file("model.safetensors")
model.load_state_dict(state, strict=False)
model.eval()

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out = model(x)
print(list(out.keys()))
```

## Citation

```bibtex
@misc{milyonus-optic-tracking-2.4,
  title={Milyonus Optic Tracking 2.4},
  author={Ayaz Elrico},
  year={2026},
  publisher={Hugging Face}
}
```
