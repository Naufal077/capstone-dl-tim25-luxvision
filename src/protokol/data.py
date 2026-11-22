# 1. Pindahkan aturan ignore .venvcapstone ke lokal Git agar .gitignore tidak bentrok saat merge
grep -qxF '.venvcapstone/' .git/info/exclude || echo '.venvcapstone/' >> .git/info/exclude
git checkout -- .gitignore 2>/dev/null || true

# 2. Simpan progres kode Autoencoder kamu terlebih dahulu
git add models/123450002/
git commit -m "wip(ae): simpan kerangka model AutoencoderLux" || true

# 3. Tarik pembaruan main dan gabungkan ke branch kerja kamu
git checkout main && git pull origin main
git checkout 123450002/kerja && git merge main

# 4. Cek struktur folder dataset primer yang wajib disiapkan
cat data/MANIFEST.json && echo "=== DATA.PY ===" && cat src/protokol/data.py
