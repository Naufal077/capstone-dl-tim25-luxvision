from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.common import manifest

TUGAS = "regresi"

FOLDER_SESI = (
    Path("data")
    / "primer"
    / "sesi-20261116-112518"
)

UKURAN_CITRA = (224, 224)
JUMLAH_FOLD = 3


def _muat_seluruh_data():
    """Memuat gambar primer dan memasangkan gambar dengan label lux."""

    manifest.baca()

    folder = manifest.AKAR / FOLDER_SESI
    file_label = folder / "label.csv"
    file_kamera = folder / "kamera_index.csv"
    folder_gambar = folder / "kamera"

    if not file_label.is_file():
        raise FileNotFoundError(file_label)

    if not file_kamera.is_file():
        raise FileNotFoundError(file_kamera)

    label = pd.read_csv(file_label)
    kamera = pd.read_csv(file_kamera)

    kolom_label = {
        "blok",
        "t_mulai_host",
        "t_selesai_host",
        "lux",
    }
    kolom_kamera = {"berkas", "t_host"}

    if not kolom_label.issubset(label.columns):
        raise ValueError(
            f"Kolom label.csv tidak lengkap: {kolom_label}"
        )

    if not kolom_kamera.issubset(kamera.columns):
        raise ValueError(
            f"Kolom kamera_index.csv tidak lengkap: {kolom_kamera}"
        )

    label = label.dropna(
        subset=["blok", "t_mulai_host", "t_selesai_host", "lux"]
    ).copy()

    kamera = kamera.dropna(
        subset=["berkas", "t_host"]
    ).copy()

    X = []
    y = []
    kelompok = []
    nama_gambar_terpakai = set()

    for _, kondisi in label.iterrows():
        awal = float(kondisi["t_mulai_host"]) + 1.0
        akhir = float(kondisi["t_selesai_host"])
        nilai_lux = float(kondisi["lux"])
        blok = int(kondisi["blok"])

        # Mengabaikan satu detik pertama setiap kondisi.
        cocok = kamera.loc[
            (kamera["t_host"] >= awal)
            & (kamera["t_host"] <= akhir)
        ]

        for _, baris_kamera in cocok.iterrows():
            nama = str(baris_kamera["berkas"])

            if nama in nama_gambar_terpakai:
                raise ValueError(
                    f"Gambar terpasangkan lebih dari sekali: {nama}"
                )

            path_gambar = folder_gambar / nama

            if not path_gambar.is_file():
                raise FileNotFoundError(path_gambar)

            with Image.open(path_gambar) as gambar:
                gambar = gambar.convert("RGB")
                gambar = gambar.resize(UKURAN_CITRA)
                citra = np.asarray(gambar, dtype=np.float32) / 255.0

            X.append(citra)
            y.append(nilai_lux)
            kelompok.append(blok)
            nama_gambar_terpakai.add(nama)

    if not X:
        raise ValueError("Tidak ada gambar yang berhasil dipasangkan.")

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.float32)
    kelompok = np.asarray(kelompok, dtype=np.int64)

    blok_tersedia = sorted(np.unique(kelompok).tolist())

    if len(blok_tersedia) != JUMLAH_FOLD:
        raise ValueError(
            f"Diharapkan {JUMLAH_FOLD} blok, "
            f"tetapi ditemukan {blok_tersedia}"
        )

    print(f"Jumlah gambar terpakai: {len(X)}")
    print(f"Jumlah kondisi berblok: {blok_tersedia}")

    for blok in blok_tersedia:
        jumlah = int(np.sum(kelompok == blok))
        print(f"Blok {blok}: {jumlah} gambar")

    return X, y, kelompok


def muat_fold(fold: int):
    """
    Memuat data untuk Group Cross-Validation 3-Fold.

    Fold 1: validasi blok 1, latih blok 2 dan 3.
    Fold 2: validasi blok 2, latih blok 1 dan 3.
    Fold 3: validasi blok 3, latih blok 1 dan 2.
    """

    if fold not in (1, 2, 3):
        raise ValueError("Nomor fold harus 1, 2, atau 3.")

    X, y, kelompok = _muat_seluruh_data()

    blok_validasi = fold

    indeks_latih = np.where(kelompok != blok_validasi)[0]
    indeks_val = np.where(kelompok == blok_validasi)[0]

    if len(indeks_latih) == 0 or len(indeks_val) == 0:
        raise ValueError("Data latih atau validasi kosong.")

    X_train = X[indeks_latih]
    y_train = y[indeks_latih]

    X_val = X[indeks_val]
    y_val = y[indeks_val]

    # Dataset belum menyediakan test terpisah.
    X_test, y_test = None, None

    print(f"\nFold {fold}")
    print(f"Jumlah data latih: {len(X_train)}")
    print(f"Jumlah data validasi: {len(X_val)}")
    print(f"Blok validasi: {blok_validasi}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def muat():
    """
    Antarmuka standar untuk evaluate.py.

    Fold dipilih melalui variabel lingkungan LUXVISION_FOLD.
    Default menggunakan fold 1.
    """

    fold = int(os.environ.get("LUXVISION_FOLD", "1"))
    return muat_fold(fold)
