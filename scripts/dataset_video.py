"""
Dataset loader for MP4 / MOV / screenshots.
Uses MediaPipe Face Landmarker to generate instant landmark + pose + blendshape labels.
Additive — does not change existing algorithms.
"""
import cv2
import torch
from torch.utils.data import Dataset
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path

class VideoScreenshotDataset(Dataset):
    def __init__(self, root_dir, max_frames_per_video=30, image_size=256):
        self.root = Path(root_dir)
        self.files = list(self.root.glob("**/*.[mp][po][v4]")) + \
                     list(self.root.glob("**/*.[mp][ov]")) + \
                     list(self.root.glob("**/*.[jp][pn]g")) + \
                     list(self.root.glob("**/*.jpeg"))
        self.max_frames = max_frames_per_video
        self.image_size = image_size
        self._init_mediapipe()

    def _init_mediapipe(self):
        base = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def __len__(self):
        return len(self.files)

    def _load_frames(self, path: Path):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img = cv2.imread(str(path))
            return [img] if img is not None else []
        cap = cv2.VideoCapture(str(path))
        frames = []
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, idx):
        path = self.files[idx]
        frames = self._load_frames(path)
        if not frames:
            return self.__getitem__((idx + 1) % len(self))

        frame = frames[0]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        landmarks = torch.zeros(478, 3)
        blendshapes = torch.zeros(52)
        pose = torch.zeros(6)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            landmarks = torch.tensor([[p.x, p.y, p.z] for p in lm], dtype=torch.float32)

        if result.face_blendshapes:
            blendshapes = torch.tensor(
                [b.score for b in result.face_blendshapes[0]], dtype=torch.float32
            )

        if result.facial_transformation_matrixes:
            m = result.facial_transformation_matrixes[0]
            pose = torch.tensor([m[0][3], m[1][3], m[2][3], 0, 0, 0], dtype=torch.float32)

        img = cv2.resize(rgb, (self.image_size, self.image_size))
        pixel_values = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "pixel_values": pixel_values,
            "landmarks": landmarks,
            "blendshapes": blendshapes,
            "pose": pose,
            "focus": torch.tensor([1.0]),
            "path": str(path),
        }
