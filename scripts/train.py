"""
Training skeleton for Milyonus Optic Tracking Model.
Use this as the base for full training / fine-tuning on custom datasets.
"""
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from modeling_optic_tracking import MilyonusOpticTrackingModel
from configuration_optic_tracking import MilyonusOpticTrackingConfig

def main():
    config = MilyonusOpticTrackingConfig()
    model = MilyonusOpticTrackingModel(config)

    # TODO: load your dataset (images + landmarks, pose, focus labels, etc.)
    # train_loader = DataLoader(...)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = get_scheduler(...)

    model.train()
    # for epoch in range(epochs):
    #     for batch in train_loader:
    #         outputs = model(batch["pixel_values"])
    #         loss = compute_multi_task_loss(outputs, batch)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    print("Training skeleton ready. Implement dataset + multi-task loss (landmarks + pose + focus + blendshapes).")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

if __name__ == "__main__":
    main()
