# Protokol Evaluasi Tim — LUXVISION

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | LUXVISION – Lux Meter from Camera Image |
| Sumber dataset | Data primer Tim 25 yang dikumpulkan melalui pengujian kamera dan sensor cahaya BH1750 pada test bench dengan pendampingan dosen. |
| Lokasi data primer | `data/primer/sesi-20261116-112518/` |
| Jenis tugas (`TUGAS`) | `regresi` |
| Target / label | Nilai iluminansi atau intensitas cahaya dalam satuan lux berdasarkan pengukuran sensor BH1750. |
| Jumlah sesi | 1 sesi pengambilan data, 16 November 2026. |
| Jumlah kondisi eksperimen | 72 kondisi. |
| Kondisi cahaya ruangan | 3 kondisi: lampu ruang mati, jendela/tirai terbuka, dan lampu ruang menyala. |
| Tingkat intensitas lampu otomatis | 2 tingkat: 0 dan 255. |
| Jumlah pengulangan | 12 pengulangan untuk setiap kombinasi kondisi cahaya ruangan dan intensitas lampu. |
| Total kombinasi kondisi | 3 kondisi cahaya ruangan × 2 tingkat intensitas lampu × 12 pengulangan = 72 kondisi. |
| Durasi setiap kondisi | 5 detik. |
| Interval pengambilan gambar | 0,5 detik. |
| Jumlah citra mentah | 1.728 citra kamera. |
| Jumlah label pengukuran | 72 nilai lux pada `label.csv`. |
| Jumlah citra setelah seleksi waktu | 576 citra, terdiri atas 8 citra per kondisi setelah pembuangan 1 detik pertama. |
| Ukuran citra masukan model | 224 × 224 piksel, format RGB. |
| Pencocokan citra dan label | Timestamp pada `kamera_index.csv` dicocokkan dengan interval pengukuran pada `label.csv`. Citra yang digunakan memiliki timestamp mulai dari 1 detik setelah waktu mulai kondisi hingga waktu selesai kondisi. |
| Cara split | GroupKFold berdasarkan blok eksperimen. Seluruh citra dari satu kondisi harus berada dalam kelompok yang sama. |
| Metrik utama | MAE. MAPE digunakan sebagai metrik tambahan. |
| Pelaporan hasil | Rata-rata ± standar deviasi dari 3 seed. |
| Prapemrosesan | Pencocokan citra dengan label berdasarkan timestamp, pembuangan citra pada 1 detik pertama setiap kondisi, pengubahan ukuran citra menjadi 224 × 224 piksel, dan normalisasi menggunakan statistik data latih saja. |
| Anggaran komputasi | Maksimal 20 menit menggunakan CPU laptop. Ukuran model ONNX (KB) dan latensi inferensi (ms) juga dicatat. |
| Evaluasi data uji | Evaluasi dilakukan satu kali oleh Penjaga Data & Evaluasi pada sprint terakhir menggunakan `--final`, setelah mekanisme data uji ditetapkan. |
| `data_hash` saat dibekukan | Belum tersedia. Diisi berdasarkan nilai `data_hash` pada `results/baseline/metrics.json` setelah dataset primer dan protokol dibekukan. |

## Catatan Pembagian Data dan Evaluasi

1. Pembagian data dilakukan berdasarkan kelompok blok eksperimen, bukan secara acak per citra.
2. Setiap kondisi eksperimen memiliki satu nilai lux dan sejumlah citra yang terkait dengan kondisi tersebut. Citra dari kondisi yang sama tidak boleh tersebar ke kelompok latih dan validasi yang berbeda.
3. Dataset aktual terdiri atas satu sesi dan tiga blok eksperimen. GroupKFold berdasarkan blok menghasilkan tiga kelompok.
4. Hasil evaluasi hanya menggambarkan kemampuan model pada kelompok blok yang tersedia. Generalisasi terhadap sesi pengambilan data yang berbeda belum dapat dievaluasi karena dataset saat ini baru mencakup satu sesi.
5. Seluruh citra mentah tetap disimpan. Pembuangan citra pada 1 detik pertama dilakukan pada tahap pemilihan data, bukan dengan menghapus berkas gambar.
6. Data validasi digunakan untuk pemilihan model dan konfigurasi. Data uji akhir tidak digunakan selama proses pelatihan dan pemilihan model.
7. Model CNN, Transfer Learning, dan Autoencoder menggunakan sumber data, target, dan aturan evaluasi yang sama.

## Sumber Data dan Metadata

Data primer sesi 16 November 2026 memiliki berkas utama:

- `kamera/`: citra hasil pengambilan kamera.
- `kamera_index.csv`: indeks nama citra dan timestamp pengambilan.
- `label.csv`: informasi kondisi eksperimen dan nilai lux BH1750.
- `lingkungan.csv`: catatan kondisi lingkungan.
- `aktuator.csv`: catatan aktuator.
- `daya.csv`: catatan daya.
- `manifest.json`: metadata dan checksum berkas sesi.
- `rencana.yaml`: konfigurasi pengambilan data.

## Persetujuan dan Pembekuan Protokol

**Disetujui pembimbing pada:** Menunggu persetujuan revisi protokol.

**Tautan pull request:** Diisi setelah PR revisi diajukan.

**Catatan:** Revisi ini menyesuaikan dokumentasi dengan data primer aktual sesi 16 November 2026. Pembekuan protokol dilakukan setelah revisi diperiksa dan disetujui oleh pembimbing serta tim.
