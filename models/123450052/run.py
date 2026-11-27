"""CNN regresi untuk memperkirakan lux dari citra RGB 224 x 224.

Contoh pemakaian dari akar repositori:
    python models/123450052/run.py --seed 1
    python models/123450052/run.py --seed 1 --fold 2

Loader protokol memilih fold validasi melalui LUXVISION_FOLD. Data uji final
belum tersedia pada protokol saat ini, jadi opsi --final sengaja ditolak.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
MODEL = "cnn"
BATCH_SIZE = 16
EPOCHS_DEFAULT = 10


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE persen; abaikan target nol agar pembagian terdefinisi."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def siapkan_citra(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validasi NHWC RGB 224x224, ubah ke NCHW, normalisasi dari data latih."""
    def ubah(X: np.ndarray, nama: str) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 4 or X.shape[1:3] != (224, 224) or X.shape[3] != 3:
            raise ValueError(
                f"{nama} harus berbentuk (N, 224, 224, 3), ditemukan {X.shape}."
            )
        if not np.isfinite(X).all():
            raise ValueError(f"{nama} mengandung NaN atau nilai tak hingga.")
        return np.transpose(X, (0, 3, 1, 2))

    Xtr = ubah(X_train, "X_train")
    Xva = ubah(X_val, "X_val")
    mean = Xtr.mean(axis=(0, 2, 3), keepdims=True)
    std = Xtr.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((Xtr - mean) / std).astype(np.float32), ((Xva - mean) / std).astype(np.float32)


def buat_model(jumlah_kanal: int = 3):
    import torch
    from torch import nn

    class LuxCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(jumlah_kanal, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.features(x).squeeze(1)

    return LuxCNN()


def latih(seed: int, epochs: int, device_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    if (Xte is None) != (yte is None):
        raise ValueError("Loader harus menyediakan X_test dan y_test bersamaan, atau keduanya None.")
    if Xte is not None:
        # The current protocol reserves evaluation for the validation fold.
        print("Info: loader menyediakan data uji; tidak digunakan tanpa --final.")
    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError("Subset latih dan validasi harus berisi citra.")
    if ytr is None or yva is None:
        raise ValueError("Label latih dan validasi harus tersedia.")

    ytr = np.asarray(ytr, dtype=np.float32).reshape(-1)
    yva = np.asarray(yva, dtype=np.float32).reshape(-1)
    if len(ytr) != len(Xtr) or len(yva) != len(Xva):
        raise ValueError("Jumlah citra dan label tidak sama.")
    if not np.isfinite(ytr).all() or not np.isfinite(yva).all():
        raise ValueError("Label mengandung NaN atau nilai tak hingga.")

    Xtr, Xva = siapkan_citra(Xtr, Xva)
    mean_target = float(ytr.mean())
    std_target = float(ytr.std())
    if not np.isfinite(std_target) or std_target < 1e-8:
        raise ValueError("Simpangan baku target latih tidak valid.")
    ytr_norm = ((ytr - mean_target) / std_target).astype(np.float32)

    device = torch.device(device_name)
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr_norm))
    loader = DataLoader(
        train_ds,
        batch_size=min(BATCH_SIZE, len(train_ds)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    model = buat_model(Xtr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_count += len(yb)
        print(f"Epoch {epoch + 1}/{epochs} | MSE={total_loss / total_count:.4f}")

    model.eval()
    with torch.no_grad():
        x_val_tensor = torch.from_numpy(Xva).to(device)
        pred_val_norm = model(x_val_tensor).cpu().numpy().astype(np.float32).reshape(-1)
    pred_val = pred_val_norm * std_target + mean_target
    pred_val = pred_val.astype(np.float32)

    return pred_val, yva, mape(yva, pred_val)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--perangkat", default=None, help="Nama perangkat untuk metrics.json")
    parser.add_argument("--final", action="store_true", help="Ditolak: test set belum tersedia.")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs harus minimal 1.")
    if args.final:
        parser.error("Protokol tim saat ini belum menyediakan test set; jangan gunakan --final.")
    if args.fold is not None:
        os.environ["LUXVISION_FOLD"] = str(args.fold)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda diminta, tetapi CUDA tidak tersedia.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    nama_perangkat = args.perangkat or (
        torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    )

    print(f"Model: {MODEL} | seed: {args.seed} | fold: {os.environ.get('LUXVISION_FOLD', '1')}")
    print(f"Device: {device} ({nama_perangkat})")
    t0 = time.time()
    pred_val, y_val, val_mape = latih(args.seed, args.epochs, device)
    durasi = time.time() - t0

    if len(pred_val) != len(y_val):
        raise ValueError("Panjang prediksi validasi tidak sama dengan label validasi.")

    output = ROOT / "results" / NIM
    output.mkdir(parents=True, exist_ok=True)
    pred_path = output / "pred_val.npy"
    np.save(pred_path, pred_val)

    cmd = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", MODEL,
        "--seed", str(args.seed),
        "--pred-val", str(pred_path),
        "--durasi", f"{durasi:.2f}",
        "--perangkat", nama_perangkat,
        "--out", str(output / "metrics.json"),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    print(f"val_mape: {val_mape:.6f}% (laporan tambahan; evaluator bersama tidak mencatat MAPE)")
    print(f"Prediksi tersimpan di {pred_path}")


if __name__ == "__main__":
    main()
