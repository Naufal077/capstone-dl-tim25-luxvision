# Laporan tim — LUXVISION (Tim 25)

## Ringkasan
Eksperimen ini membandingkan lima keluarga model deep learning pada kasus klasifikasi/regresi dataset LUXVISION dengan protokol evaluasi yang dibakukan. Temuan utama akan dilaporkan setelah seluruh model menyelesaikan evaluasi pada Sprint akhir.

## Protokol
Perbandingan antar-model dijaga keadilannya dengan aturan berikut:
- **Split Data:** Menggunakan split beku dari `src.common.split` dengan seed 2026[cite: 1].
- **Metrik Utama:** Evaluasi menggunakan akurasi dan macro-F1 (atau RMSE/MAE/R² jika regresi)[cite: 1].
- **Anggaran Komputasi:** Maksimal 20 menit pelatihan per run di CPU/GPU laptop[cite: 1].
- **Eksperimen:** Setiap konfigurasi akhir dijalankan pada 3 seed (seed 1, 2, 3)[cite: 1].
- **Evaluasi Data Uji:** Data uji hanya dievaluasi satu kali pada sprint terakhir oleh Penjaga Data (`--final`)[cite: 1].

## Hasil
*(Belum diisi - akan terisi otomatis oleh `python run_all.py` pada Sprint akhir)*[cite: 1]

## Analisis
*(Belum diisi - akan dianalisis setelah seluruh hasil eksperimen Sprint 1-3 keluar)*

## Rekomendasi
*(Belum diisi - akan ditulis pada Sprint akhir)*

## Pembagian kerja
| Anggota | NIM | Jalur | Tugas sampingan | Bagian laporan yang ditulis |
|---|---|---|---|---|
| Muhammad Naufal Ramadhan | 123450113 | cnn / baseline | Penjaga Data & Evaluasi (Setup data.py, MANIFEST.json, baseline) | Bagian Protokol & Hasil |
| Tarisya | 123450052 | <Pilih: tl/rnn/trf/ae> | Pengelola Dokumentasi & AI Log | Bagian Rekomendasi & Pembagian kerja |
| Afifah (Fifah) | 123450002 | <Pilih: tl/rnn/trf/ae> | Penanggung Jawab QA / Reviewer PR | Bagian Ringkasan & Analisis |

Komponen yang tidak tersedia: Tidak ada. Semua modul protokol dasar Sprint 0 telah dikonfigurasi.
