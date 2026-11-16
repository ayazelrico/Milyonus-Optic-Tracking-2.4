"""
Inference example for Milyonus Optic Tracking Model.
"""
import torch
from PIL import Image
from transformers import AutoImageProcessor
from modeling_optic_tracking import MilyonusOpticTrackingModel
from configuration_optic_tracking import MilyonusOpticTrackingConfig

def load_model(path="."):
    config = MilyonusOpticTrackingConfig.from_pretrained(path)
    model = MilyonusOpticTrackingModel.from_pretrained(path)
    model.eval()
    return model

def infer(model, image: Image.Image):
    # Minimal processor (or use AutoImageProcessor if registered)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pixel_values = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(pixel_values)

    return {
        "landmarks": outputs["landmarks"].cpu().numpy(),
        "focus_score": outputs["focus_score"].item() if "focus_score" in outputs else None,
        "confidence": outputs["confidence"].item(),
        "pose": outputs["pose"].cpu().numpy() if "pose" in outputs else None,
    }

if __name__ == "__main__":
    print("Inference module ready. Load trained weights and pass an image.")
