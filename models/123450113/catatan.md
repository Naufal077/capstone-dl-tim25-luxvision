# Catatan Model 123450113 — Transfer Learning

## 1. Arsitektur Model

- Arsitektur: ResNet18 pretrained ImageNet.
- Jenis tugas: Regresi nilai lux berdasarkan citra.
- Ukuran input: 224 × 224 piksel.
- Output: 1 nilai prediksi lux.
- Normalisasi citra: ImageNet normalization.
- Pretrained weights: ImageNet.

## 2. Alasan Optimizer

Menggunakan optimizer Adam karena memiliki mekanisme adaptive learning rate yang menyesuaikan pembaruan parameter berdasarkan estimasi momen gradien. Optimizer ini digunakan pada dua tahap pelatihan dengan pengaturan learning rate yang berbeda. Weight decay sebesar 1e-4 diterapkan untuk membantu mengendalikan kompleksitas model.

## 3. Regularisasi

- Menggunakan pretrained weights ImageNet sebagai titik awal pembelajaran.
- Target lux distandardisasi berdasarkan data training.
- Menggunakan SmoothL1Loss.
- Backbone dibekukan pada tahap awal pelatihan.
- Pada tahap fine-tuning, hanya layer4 dan regression head yang dilatih.
- Menggunakan weight decay sebesar 1e-4.

## 4. Konfigurasi Pelatihan

| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Batch size | 16 |
| Loss function | SmoothL1Loss |
| Epoch tahap 1 | 3 |
| Epoch tahap 2 | 7 |
| Learning rate tahap 1 | 1e-3 |
| Learning rate layer4 | 1e-4 |
| Learning rate FC tahap 2 | 1e-4 |
| Weight decay | 1e-4 |
| Seed eksperimen | 1 |
| Evaluasi | Fold 1 |

## 5. Pencarian Hyperparameter

Pencarian hyperparameter secara menyeluruh belum dilakukan. Konfigurasi saat ini merupakan konfigurasi awal untuk memenuhi eksperimen Sprint 1. Penyetelan hyperparameter direncanakan pada Sprint 2.

## 6. Ablation (>= 2)

Eksperimen ablation belum dilakukan. Pengujian minimal dua konfigurasi ablation direncanakan pada Sprint 2.

## 7. Hasil Awal — Fold 1, Seed 1

| Metrik | Hasil |
|---|---:|
| MAE | 7.041013 lux |
| MAPE | 4.733936% |
| RMSE | 9.414479 |
| R² | -0.292789 |
| Durasi | 25.84 detik |

- Data training: 384 citra.
- Data validasi: 192 citra.
- Data hash: `82d4170d749cca628272977e45417aca`.

## 8. Perbandingan dengan Baseline

| Metrik | Baseline | Transfer Learning |
|---|---:|---:|
| MAE | 3.218964 lux | 7.041013 lux |
| MAPE | 2.131700% | 4.733936% |

Pada Fold 1, baseline brightness dengan Linear Regression menghasilkan MAE dan MAPE yang lebih rendah dibandingkan model Transfer Learning. Dengan demikian, baseline masih memberikan hasil lebih baik pada eksperimen awal ini.

## 9. Hasil 3 Seed Konfigurasi Akhir

Belum tersedia. Eksperimen saat ini baru menggunakan seed 1. Pengujian tiga seed dengan konfigurasi akhir direncanakan pada Sprint 2.