"""
Production-grade fine-tuning script for Milyonus Optic Tracking.
Supports video/screenshot datasets with MediaPipe-generated labels.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file
from configuration_optic_tracking import MilyonusOpticTrackingConfig
from modeling_optic_tracking import MilyonusOpticTrackingModel
from dataset_video import VideoScreenshotDataset

def multi_task_loss(outputs, targets):
    loss = 0.0
    if "landmarks" in outputs:
        loss += nn.functional.mse_loss(outputs["landmarks"], targets["landmarks"])
    if "blendshapes" in outputs:
        loss += 0.5 * nn.functional.mse_loss(outputs["blendshapes"], targets["blendshapes"])
    if "pose" in outputs:
        loss += 0.3 * nn.functional.mse_loss(outputs["pose"], targets["pose"])
    if "focus_score" in outputs:
        loss += 0.2 * nn.functional.mse_loss(outputs["focus_score"], targets["focus"])
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to videos/screenshots folder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="finetuned_model.safetensors")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = MilyonusOpticTrackingConfig()
    model = MilyonusOpticTrackingModel(config)

    if args.resume:
        state = load_file(args.resume)
        model.load_state_dict(state, strict=False)
        print(f"Resumed from {args.resume}")

    dataset = VideoScreenshotDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch["pixel_values"])
            loss = multi_task_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | loss={total_loss/len(loader):.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    save_file(model.state_dict(), args.output)
    print(f"Saved trained weights to {args.output}")

if __name__ == "__main__":
    main()
