"""Small 1D CNN for UCI HAR sensor windows (requires torch)."""

import torch
from torch import nn



class ActivityCNN_GravityBranch(nn.Module):
    """Input: (batch, sensor channels, time steps). Output: class logits."""

    def __init__(self, input_channels=9, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64 + 6, num_classes) # added 6 to account for the additional features from the gravity branch

    def forward(self, x, gravity_features):
        return self.classifier(torch.cat([self.features(x).squeeze(-1), gravity_features], dim=1))


if __name__ == "__main__":
    model = ActivityCNN_GravityBranch()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {count:,}")
    assert count < 20_000
    print("Output shape:", tuple(model(torch.zeros(2, 9, 128)).shape))
