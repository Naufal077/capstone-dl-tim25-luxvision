from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from src.common import manifest, split

TUGAS = "regresi"


def muat():
    """Memuat dummy dataset LUXVISION untuk pengujian awal pipeline."""

    # Membaca manifest
    m = manifest.baca()

    # Lokasi dataset dummy
    folder = manifest.AKAR / "data" / "dummy"
    folder_gambar = folder / "images"
    file_label = folder / "labels.csv"

    # Membaca label
    df = pd.read_csv(file_label, sep=None, engine="python")

    # Memeriksa kolom
    if not {"filename", "lux"}.issubset(df.columns):
        raise ValueError(
            "labels.csv harus memiliki kolom filename dan lux."
        )

    X = []
    y = []

    # Membaca citra dan label
    for _, baris in df.iterrows():
        nama_gambar = str(baris["filename"])
        path_gambar = folder_gambar / nama_gambar

        if not path_gambar.is_file():
            raise FileNotFoundError(
                f"Gambar tidak ditemukan: {path_gambar}"
            )

        gambar = Image.open(path_gambar).convert("RGB")
        gambar = gambar.resize((224, 224))

        X.append(np.asarray(gambar, dtype=np.float32) / 255.0)
        y.append(float(baris["lux"]))

    X = np.stack(X)
    y = np.asarray(y, dtype=np.float32)

    # Split sementara untuk pengujian pipeline
    idx_latih, idx_val = split.indeks(
        len(X),
        porsi_val=0.1,
        seed=2026
    )

    X_train, y_train = X[idx_latih], y[idx_latih]
    X_val, y_val = X[idx_val], y[idx_val]

    # Dummy dataset belum menyediakan test terpisah
    X_test, y_test = None, None

    return X_train, y_train, X_val, y_val, X_test, y_test