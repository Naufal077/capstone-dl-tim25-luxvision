from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Lokasi penyimpanan
OUTPUT_DIR = Path("data/dummy")
IMAGE_DIR = OUTPUT_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Pengaturan data
JUMLAH_DATA = 100
UKURAN = 224

rng = np.random.default_rng(2026)
labels = []

for i in range(JUMLAH_DATA):
    # Tingkat kecerahan buatan
    brightness = rng.integers(20, 236)

    # Membuat gambar RGB dengan variasi tekstur
    noise = rng.normal(0, 12, (UKURAN, UKURAN, 3))
    image_array = np.clip(
        brightness + noise, 0, 255
    ).astype(np.uint8)

    # Simpan gambar
    filename = f"gambar_{i+1:03d}.png"
    Image.fromarray(image_array).save(IMAGE_DIR / filename)

    # Label lux sintetis untuk pengujian pipeline
    lux = float(brightness * 5)

    labels.append({
        "filename": filename,
        "lux": lux
    })

# Simpan label ke CSV
df = pd.DataFrame(labels)
df.to_csv(OUTPUT_DIR / "labels.csv", index=False)

print("Data dummy berhasil dibuat!")
print(f"Jumlah gambar: {len(df)}")
print(f"Lokasi: {OUTPUT_DIR}")
print(df.head())