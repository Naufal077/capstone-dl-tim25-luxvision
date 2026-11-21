
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "ae"


def ke_tensor(X):
    import torch

    X = np.asarray(X, dtype=np.float32)

    # Ubah NHWC menjadi NCHW untuk PyTorch
    if X.ndim == 4 and X.shape[-1] == 3:
        X = X.transpose(0, 3, 1, 2)

    if X.ndim != 4 or X.shape[1] != 3:
        raise ValueError(f"Bentuk citra tidak sesuai: {X.shape}")

    return torch.tensor(X, dtype=torch.float32)


def latih_dan_prediksi(seed: int, final: bool):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Memuat data dari protokol tim
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    X_train = ke_tensor(Xtr)
    X_val = ke_tensor(Xva)

    y_train = torch.tensor(
        np.asarray(ytr, dtype=np.float32).reshape(-1, 1)
    )
    y_val = np.asarray(yva, dtype=np.float32).reshape(-1)

    # Standardisasi target berdasarkan data train saja
    mean_y = y_train.mean()
    std_y = y_train.std().clamp(min=1e-6)
    y_train_norm = (y_train - mean_y) / std_y

    class AutoencoderLux(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU()
            )

            self.regression = nn.Linear(64, 1)

            self.decoder_input = nn.Linear(64, 128 * 14 * 14)

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    128, 64, 4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    64, 32, 4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    32, 16, 4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    16, 3, 4, stride=2, padding=1
                ),
                nn.Sigmoid()
            )

        def forward(self, x):
            fitur = self.encoder(x)

            prediksi = self.regression(fitur)

            hasil_decoder = self.decoder_input(fitur)
            hasil_decoder = hasil_decoder.reshape(
                -1, 128, 14, 14
            )
            rekonstruksi = self.decoder(hasil_decoder)

            return prediksi, rekonstruksi

    model = AutoencoderLux()

    # AdamW dipilih untuk optimisasi dengan weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    epochs = 30
    batch_size = min(8, len(X_train))

    if batch_size < 1:
        raise ValueError("Data training kosong.")

    model.train()

    for epoch in range(epochs):
        urutan = torch.randperm(len(X_train))

        for i in range(0, len(X_train), batch_size):
            idx = urutan[i:i + batch_size]

            xb = X_train[idx]
            yb = y_train_norm[idx]

            optimizer.zero_grad()

            pred, rekonstruksi = model(xb)

            loss_regresi = F.mse_loss(pred, yb)
            loss_rekonstruksi = F.mse_loss(rekonstruksi, xb)

            loss = loss_regresi + 0.1 * loss_rekonstruksi

            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        pred_val_norm, _ = model(X_val)

        pred_val = (
            pred_val_norm * std_y + mean_y
        ).numpy().reshape(-1)

        pred_test = None

        if final and Xte is not None:
            X_test = ke_tensor(Xte)
            pred_test_norm, _ = model(X_test)

            pred_test = (
                pred_test_norm * std_y + mean_y
            ).numpy().reshape(-1)

    return pred_val, pred_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        a.seed, a.final
    )

    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    file_val = out / "pred_val.npy"
    np.save(file_val, np.asarray(pred_val, dtype=np.float32))

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(a.seed),
        "--pred-val", str(file_val),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat
    ]

    if pred_test is not None:
        file_test = out / "pred_test.npy"
        np.save(file_test, pred_test.astype(np.float32))
        perintah += ["--pred-test", str(file_test)]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
