---
language: en
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
pipeline_tag: image-feature-extraction
model-index:
  - name: Milyonus-Optic-Tracking-2.4
    results: []
---

<div align="center">

<img src="docs/assets/milyonus-wordmark-lg.png" alt="Milyonus" width="640" />

**Browser-based facial optic tracking by Milyonus.**

[Website](https://www.milyonus.com) · [Repository](https://github.com/ayazelrico/Milyonus-Optic-Tracking-2.4)

</div>

---

# Milyonus Optic Tracking 2.4

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**Ultra-detailed multi-task facial optic tracking model + real-time browser application.**

Real-time, fully on-device facial optic tracking: 478-point face mesh, 3D head pose, blendshapes, multi-face tracking with persistent IDs, focus/attention scoring, and expression analysis.

The repository contains:
1. **Production browser application** (TypeScript / React / MediaPipe) — zero-server, on-device.
2. **Full model foundation** (PyTorch / Transformers) — architecture, config, training skeleton, and inference code ready for large-scale training or fine-tuning.

Nothing leaves the device in the browser app. The Python model is designed so you own the full stack.

## Model Architecture (New)

- Hybrid backbone: EfficientNet-style stem + ViT spatial transformer (12 layers, 768 dim)
- Temporal transformer for multi-frame consistency and tracking
- Multi-task heads:
  - 478 × 3 landmarks
  - 52 ARKit-style blendshapes
  - 3D pose (yaw/pitch/roll + translation)
  - Continuous focus score (0–1)
  - 8-class expression
  - Detection confidence
  - Track embedding for multi-face ID stability
- Designed to be trained end-to-end or used as a drop-in replacement for MediaPipe graphs

See `modeling_optic_tracking.py` and `config.json`.

## Quick Start (Browser App)

Requires Node.js 18+ (or Bun).

```sh
git clone https://github.com/ayazelrico/Milyonus-Optic-Tracking-2.4
cd Milyonus-Optic-Tracking-2.4
npm install
npm run dev
```

## Quick Start (Python Model)

```sh
pip install -r requirements.txt
python scripts/inference.py
```

## Features

- **478-point face mesh** — full tesselation, contours and iris landmarks (468 face + 10 iris)
- **3D head pose** — yaw / pitch / roll + depth
- **Confidence score** — real detection confidence
- **Multi-face support** — up to 4 faces with stable Track IDs
- **Focus Tracking Score** — continuous 0–1 + discrete gaze labels
- **Expression matrix** — 8-bucket vibe from blendshapes
- **Per-second recording + per-minute reports + CSV export**
- **GPU + CPU fallback**

## How it works (Browser)

The browser app runs MediaPipe FaceLandmarker + FaceDetector fully on-device via WebAssembly/WebGL. Additional logic (pose decomposition, multi-face tracker, focus scoring, emotion mapping) lives in `src/lib/`.

The Python model is the path to a fully owned, larger, and more detailed model that can be trained on your own data and later exported to ONNX / TFLite / MediaPipe Task for the browser.

## Project Structure

```
├── config.json
├── configuration_optic_tracking.py
├── modeling_optic_tracking.py
├── preprocessor_config.json
├── requirements.txt
├── scripts/
│   ├── train.py
│   └── inference.py
├── src/                    # Existing browser application (unchanged)
│   ├── components/
│   └── lib/
├── docs/
└── ...
```

## Training

Use `scripts/train.py` as the starting point. Implement your dataset loader and multi-task loss (landmarks + pose + focus + blendshapes + expressions). The architecture is already large enough for high-capacity training.

## License

MIT — © Milyonus

Built on MediaPipe (Google AI Edge) for the current browser runtime. The custom model architecture is original.
