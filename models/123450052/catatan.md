# Catatan model CNN — 123450052

## Tujuan

Memprediksi nilai iluminansi (lux) dari citra RGB berukuran 224 × 224 menggunakan CNN 2D. Data, fold, dan label diambil dari `src.protokol.data.muat()`; split tidak dibuat ulang di skrip model.

## Arsitektur dan konfigurasi awal

- Masukan: `(N, 224, 224, 3)` dari loader; dikonversi ke `(N, 3, 224, 224)` untuk PyTorch.
- CNN: tiga blok konvolusi 3 × 3 dengan 16, 32, dan 64 kanal; BatchNorm, ReLU, dan max-pooling pada dua blok awal.
- Kepala regresi: adaptive average pooling, dropout 0,2, dan satu neuron keluaran.
- Loss: MSE.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Batch size: 16; konfigurasi awal: 10 epoch.
- Seed dari argumen `--seed` diterapkan ke NumPy dan PyTorch.
- Normalisasi kanal dihitung hanya dari citra latih, lalu diterapkan ke validasi.
- Target lux distandardisasi memakai rata-rata dan simpangan baku data latih; prediksi dikembalikan ke satuan lux sebelum evaluasi.

## Alasan desain

Konvolusi mengekstrak pola visual lokal dari citra. Pooling mengurangi ukuran peta fitur dan biaya komputasi. BatchNorm membantu kestabilan optimisasi, sedangkan dropout dan weight decay menjadi regularisasi untuk membatasi overfitting pada dataset yang relatif kecil. Target distandardisasi agar skala loss lebih mudah dioptimalkan; metrik tetap dihitung setelah prediksi dikembalikan ke satuan lux. AdamW dipilih sebagai optimizer adaptif dengan regularisasi weight decay yang eksplisit. MSE digunakan sebagai loss regresi yang sejalan dengan evaluasi RMSE, tetapi sensitif terhadap kesalahan besar.

## Pencarian hyperparameter dan ablation

Belum ada hasil pencarian hyperparameter atau ablation yang dicatat. Jangan mengisi angka sebelum eksperimen benar-benar dijalankan. Rencana ablation yang dapat dikerjakan pada fold dan seed yang sama:

1. CNN penuh dibanding CNN tanpa dropout.
2. CNN penuh dibanding CNN tanpa BatchNorm.
3. Bandingkan learning rate `1e-3` dan `3e-4`, atau jumlah epoch 5 dan 10.

Catat MAE, MAPE, RMSE, R², durasi, seed, dan fold untuk setiap konfigurasi. Gunakan metrik dari prediksi validasi yang sama; jangan memakai test set untuk pemilihan konfigurasi.

## Hasil awal

Run seed 1, fold 1, 10 epoch pada dataset primer menghasilkan MAE `9.159818`, MAPE `6.167449%`, RMSE `12.301683`, dan R² `-1.207314` pada 192 citra validasi. Durasi tercatat `281.55` detik di CPU. Hasil ini masih satu seed dan bukan ringkasan final.

Sebagai konteks, artefak baseline yang tersedia pada data hash sama melaporkan MAE `3.218964` dan MAPE `2.131700%`; model transfer learning seed 1 melaporkan MAE `7.041013`, RMSE `9.414479`, dan R² `-0.292789`. Pada run awal ini CNN belum mengungguli kedua pembanding menurut MAE; perlu dilaporkan apa adanya.

## Evaluasi

`run.py` menyimpan prediksi validasi ke `results/123450052/pred_val.npy` dan memanggil `src.common.evaluate` untuk membuat `metrics.json`. Evaluator bersama saat ini menghitung MAE, RMSE, dan R²; MAPE dicetak sebagai informasi tambahan oleh skrip. Test set belum tersedia pada loader utama, sehingga opsi `--final` ditolak.

## Status verifikasi

Perintah `python models/123450052/run.py --seed 1` berhasil pada data primer melalui loader terbaru. Loader memakai `data/primer/sesi-20261116-112518/` dan fold 1 (validasi blok 1). Dataset sesi sebelumnya berada di `data/sesi-20261116-112518/`; folder dipindahkan ke path resmi `data/primer/` agar sesuai dengan manifest terbaru. Dummy `data/dummy/` tidak digunakan karena loader terbaru tidak menyediakan mode simulasi.
