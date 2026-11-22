
"""Transfer Learning ResNet18 untuk regresi nilai lux.

Evaluasi menggunakan Fold 1 agar sama dengan baseline non-DL.
Seed training dapat ditentukan melalui --seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"

BATCH_SIZE = 16
EPOCH_HEAD = 3
EPOCH_FINE = 7
LR_HEAD = 1e-3
LR_FINE = 1e-4
NUM_WORKERS = 0

MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)


def buat_model(device):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    jumlah_fitur = model.fc.in_features
    model.fc = torch.nn.Linear(jumlah_fitur, 1)
    return model.to(device)


def siapkan_tensor(X, y):
    import torch

    citra = torch.from_numpy(
        np.ascontiguousarray(X.transpose(0, 3, 1, 2))
    ).float()

    mean = torch.tensor(MEAN_IMAGENET).view(1, 3, 1, 1)
    std = torch.tensor(STD_IMAGENET).view(1, 3, 1, 1)
    citra = (citra - mean) / std

    target = torch.from_numpy(
        np.asarray(y, dtype=np.float32).reshape(-1, 1)
    )

    return citra, target


def latih_epoch(model, loader, optimizer, loss_fn, device):
    import torch

    model.train()

    # BatchNorm pada bagian backbone yang dibekukan tidak diperbarui.
    for modul in model.modules():
        if isinstance(modul, torch.nn.BatchNorm2d):
            if not any(p.requires_grad for p in modul.parameters()):
                modul.eval()

    total_loss = 0.0
    total_data = 0

    for citra, target in loader:
        citra = citra.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(citra)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        jumlah = len(target)
        total_loss += loss.item() * jumlah
        total_data += jumlah

    return total_loss / total_data


def prediksi(model, X, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()

    dataset = TensorDataset(X)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    hasil = []

    with torch.no_grad():
        for (citra,) in loader:
            citra = citra.to(device)
            output = model(citra)
            hasil.append(output.cpu().numpy().reshape(-1))

    return np.concatenate(hasil).astype(np.float32)


def latih(seed, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Menggunakan loader default, sama seperti baseline (Fold 1).
    Xtr, ytr, Xva, yva, _, _ = data.muat()

    print("\nJumlah data latih:", len(ytr))
    print("Jumlah data validasi:", len(yva))

    # Standardisasi target hanya berdasarkan data latih.
    mean_target = float(np.mean(ytr))
    std_target = float(np.std(ytr))

    if not np.isfinite(std_target) or std_target < 1e-8:
        raise ValueError("Standar deviasi target data latih tidak valid.")

    ytr_norm = (ytr - mean_target) / std_target

    X_train, y_train = siapkan_tensor(Xtr, ytr_norm)
    X_val, _ = siapkan_tensor(
        Xva, np.zeros(len(Xva), dtype=np.float32)
    )

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = buat_model(device)
    loss_fn = torch.nn.SmoothL1Loss()

    # Tahap 1: bekukan backbone dan latih regression head.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LR_HEAD,
        weight_decay=1e-4,
    )

    print("\nTahap 1: training regression head")
    print("Rata-rata target latih:", mean_target)
    print("Standar deviasi target latih:", std_target)

    for epoch in range(EPOCH_HEAD):
        loss = latih_epoch(
            model, loader, optimizer, loss_fn, device
        )
        print(
            f"Epoch {epoch + 1}/{EPOCH_HEAD} "
            f"| Loss: {loss:.5f}"
        )

    # Tahap 2: fine-tuning layer4 dan fc.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": LR_FINE},
            {"params": model.fc.parameters(), "lr": LR_HEAD * 0.1},
        ],
        weight_decay=1e-4,
    )

    print("\nTahap 2: fine-tuning layer4 dan fc")

    for epoch in range(EPOCH_FINE):
        loss = latih_epoch(
            model, loader, optimizer, loss_fn, device
        )
        print(
            f"Epoch {epoch + 1}/{EPOCH_FINE} "
            f"| Loss: {loss:.5f}"
        )

    # Prediksi dikembalikan ke satuan lux.
    pred_norm = prediksi(model, X_val, device)
    pred_val = pred_norm * std_target + mean_target

    del model, X_train, y_train, X_val

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pred_val.astype(np.float32), yva


def hitung_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mask = y_true != 0

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask]) / y_true[mask]
            )
        ) * 100.0
    )


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Test set belum tersedia.",
    )
    parser.add_argument("--perangkat", default=None)
    args = parser.parse_args()

    if args.final:
        raise SystemExit(
            "Test set belum tersedia. Jangan gunakan --final."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    nama_perangkat = args.perangkat or (
        torch.cuda.get_device_name(0)
        if device.type == "cuda" else "cpu"
    )

    print("Model:", JALUR)
    print("Seed training:", args.seed)
    print("Fold evaluasi: 1")
    print("Seed split: 2026")
    print("Perangkat:", nama_perangkat)
    print("Device:", device)

    output = AKAR / "results" / NIM
    output.mkdir(parents=True, exist_ok=True)

    waktu_mulai = time.time()

    pred_val, y_val = latih(args.seed, device)

    if len(pred_val) != len(y_val):
        raise ValueError(
            "Jumlah prediksi tidak sama dengan jumlah label validasi."
        )

    file_pred = output / "pred_val.npy"
    file_metrics = output / "metrics.json"

    np.save(file_pred, pred_val)

    durasi = time.time() - waktu_mulai

    # Evaluator resmi tim menghitung MAE, RMSE, R2, dan data_hash.
    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(file_pred),
        "--durasi", f"{durasi:.2f}",
        "--perangkat", nama_perangkat,
        "--out", str(file_metrics),
    ]

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR,
    )

    # Tambahkan MAPE agar format metrik utama sama dengan baseline.
    hasil = json.loads(file_metrics.read_text(encoding="utf-8"))
    hasil["metrik"]["val_mape"] = hitung_mape(y_val, pred_val)
    hasil["model"] = JALUR
    hasil["seed"] = int(args.seed)
    hasil["jumlah_fold"] = 1
    hasil["n_val"] = int(len(y_val))
    hasil["n_test"] = 0
    hasil["durasi_detik"] = round(time.time() - waktu_mulai, 2)
    hasil["perangkat"] = nama_perangkat

    file_metrics.write_text(
        json.dumps(hasil, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 45)
    print("HASIL TRANSFER LEARNING - FOLD 1")
    print("=" * 45)
    print(json.dumps(hasil, indent=2, ensure_ascii=False))
    print("\nPrediksi tersimpan:", file_pred)
    print("Metrik tersimpan:", file_metrics)


if __name__ == "__main__":
    main()
