import torch
import torch.nn as nn


class ShallowCNN1D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)      # [B, 32, 1]
        x = x.squeeze(-1)         # [B, 32]
        x = self.classifier(x)    # [B, num_classes]
        return x
