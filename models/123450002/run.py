import os
import json
import torch
import torch.nn as nn

class AutoencoderLux(nn.Module):
    def __init__(self):
        super(AutoencoderLux, self).__init__()
        # Encoder: kompresi citra 224x224
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        # Head Regresi: nilai lux
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        lux_pred = self.regressor(features)
        return lux_pred

def run():
    print("Inisialisasi model Autoencoder Tim 25 (123450002)...")

if __name__ == "__main__":
    run()
