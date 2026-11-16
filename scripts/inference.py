"""
Inference example for Milyonus Optic Tracking 2.4
"""
import torch
from safetensors.torch import load_file
from configuration_optic_tracking import MilyonusOpticTrackingConfig
from modeling_optic_tracking import MilyonusOpticTrackingModel
from PIL import Image
import torchvision.transforms as T

def load_model(weights_path="model.safetensors"):
    config = MilyonusOpticTrackingConfig()
    model = MilyonusOpticTrackingModel(config)
    state = load_file(weights_path)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def predict(model, image_path):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    return out

if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully. Ready for inference.")
