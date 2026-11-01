
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.protokol import data
from src.common import manifest


def hitung_brightness(X: np.ndarray) -> np.ndarray:
    """Menghitung rata-rata kecerahan setiap gambar."""
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 4:
        raise ValueError("Input harus berbentuk (N, H, W, C).")

    return X.mean(axis=(1, 2, 3)).reshape(-1, 1)


def hitung_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Menghitung MAPE dalam persen."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mask = y_true != 0

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
    )


def main():
    waktu_mulai = time.time()

    # Memuat dataset
    X_train, y_train, X_val, y_val, _, _ = data.muat()

    if y_val is None:
        raise ValueError("Label validasi tidak tersedia.")

    # Ekstraksi fitur kecerahan
    X_train_brightness = hitung_brightness(X_train)
    X_val_brightness = hitung_brightness(X_val)

    # Melatih Linear Regression
    model = LinearRegression()
    model.fit(X_train_brightness, y_train)

    # Prediksi data validasi
    pred_val = model.predict(X_val_brightness)

    # Menghitung metrik
    mae = mean_absolute_error(y_val, pred_val)
    mape = hitung_mape(y_val, pred_val)

    durasi = time.time() - waktu_mulai

    # Menyimpan hasil
    hasil = {
        "model": "brightness_linear_regression",
        "seed": 2026,
        "metrik": {
            "val_mae": float(mae),
            "val_mape": float(mape)
        },
        "durasi_detik": float(durasi),
        "perangkat": "cpu laptop",
        "n_val": int(len(y_val)),
        "data_hash": manifest.data_hash(b"split:2026")
    }

    output = (
        manifest.AKAR
        / "results"
        / "baseline_luxvision"
    )

    output.mkdir(parents=True, exist_ok=True)

    np.save(output / "pred_val.npy", pred_val)

    with open(output / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)

    print(json.dumps(hasil, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
