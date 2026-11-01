# Protokol evaluasi tim — LUXVISION

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | LUXVISION – Lux Meter from Camera Image |
| Sumber dataset | Data primer dikumpulkan oleh tim melalui pengujian kamera dan sensor cahaya BH1750 pada test bench. Data dari tim lain hanya digunakan sebagai data pengujian tambahan. |
| Jenis tugas (`TUGAS`) | `regresi` |
| Target / label | Nilai iluminansi atau intensitas cahaya dalam satuan lux berdasarkan pengukuran sensor BH1750. |
| Ukuran data latih / validasi / uji | Belum ditentukan. Menunggu proses pengambilan dan pengelompokan data primer. |
| Cara split | GroupKFold berdasarkan sesi, blok, atau partisipan agar data dari kelompok yang sama tidak tersebar ke data latih dan validasi. |
| Metrik utama untuk komparasi | MAE. MAPE digunakan sebagai metrik tambahan. Hasil dilaporkan dalam rata-rata ± standar deviasi dari 3 seed. |
| Praproses yang sama untuk semua model | Citra kamera berukuran 224 × 224 piksel. Pengambilan data dilakukan pada kondisi lampu ruangan menyala, lampu ruangan mati, dan jendela atau tirai terbuka. Intensitas lampu otomatis menggunakan nilai 0, 32, 64, 96, 128, 160, 192, 224, dan 255. Setiap kondisi direkam sebanyak 6 pengulangan selama 5 detik dengan interval pengambilan gambar 0,5 detik. Data pada 1 detik pertama setiap kondisi dibuang. Normalisasi menggunakan statistik data latih saja. |
| Anggaran komputasi per run | Maksimal 20 menit menggunakan CPU laptop, mengikuti ketentuan template dan menunggu persetujuan pembimbing. Ukuran model ONNX (KB) dan latensi inferensi (ms) pada laptop pengujian juga dicatat. |
| Evaluasi data uji | Evaluasi dilakukan satu kali oleh Penjaga Data & Evaluasi pada sprint terakhir (`--final`). |
| `data_hash` saat dibekukan | Belum tersedia. Diisi berdasarkan nilai `data_hash` pada `results/baseline/metrics.json` setelah dataset primer ditetapkan dan baseline dijalankan. |

**Disetujui pembimbing pada:** Belum disetujui.

**Tautan pull request:** Diisi setelah PR diajukan dan disetujui pembimbing.