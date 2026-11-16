"""
Evaluation skeleton for landmark NME, focus accuracy, etc.
"""
import numpy as np
from dataset_video import VideoScreenshotDataset

def normalized_mean_error(pred, target):
    diff = np.linalg.norm(pred - target, axis=-1)
    interocular = np.linalg.norm(target[33] - target[263])
    return np.mean(diff) / (interocular + 1e-6)

def evaluate(data_path):
    ds = VideoScreenshotDataset(data_path)
    print(f"Dataset size: {len(ds)}")
    print("Evaluation metrics (placeholder):")
    print("  Landmark NME: TBD after training")
    print("  Focus MAE: TBD after training")
    print("  Blendshape correlation: TBD after training")

if __name__ == "__main__":
    evaluate("path/to/data")
