# Camera Body Motion Controller untuk Subway Surfers dan Temple Run - Panduan Penggunaan

POC (Proof of Concept) ini pakai **webcam + MediaPipe Pose** buat membaca gerakan badan dan mengubahnya jadi tombol keyboard (`left`, `right`, `up`, `down`) yang dikirim ke game:

- Geser badan kiri/kanan/balik ke tengah → tombol `left` / `right`
- **Tangan kanan** diangkat ke atas (melebihi bahu) → tombol `up` (lompat)
- **Tangan kiri** diangkat ke atas (melebihi bahu) → tombol `down` (turun/slide)

---

## 0. Persiapan fisik (sebelum install apa-apa)

- Webcam yang berfungsi (laptop bawaan atau eksternal).
- Berdiri kira-kira **1.5–2.5 meter** dari kamera, posisikan kamera setinggi dada, supaya bahu sampai pinggul kelihatan penuh di frame.
- **Beri ruang ke atas kepala** - pastikan saat tangan diangkat lurus ke atas, pergelangan tangan masih kelihatan jelas di dalam frame (tidak terpotong di tepi atas). Ini penting buat akurasi deteksi lompat/turun.
- Ruang gerak ke kiri/kanan secukupnya buat geser badan.
- Pencahayaan cukup - hindari posisi membelakangi jendela/lampu (backlit), karena bikin model kurang yakin membaca posisi tangan.

---

## 1. Install Python

Disarankan **Python 3.10 atau 3.11** (MediaPipe paling stabil di rentang versi ini; versi Python yang terlalu baru kadang belum punya wheel MediaPipe yang kompatibel).

1. Download dari https://www.python.org/downloads/
2. Saat instalasi di **Windows**, **wajib centang "Add python.exe to PATH"** sebelum klik Install.
3. Cek di terminal/command prompt:
   ```
   python --version
   ```
   Harus muncul `Python 3.10.x` atau `3.11.x`. Kalau command tidak dikenali, install ulang dan pastikan centang opsi PATH di atas (Windows), atau pakai `python3` (Mac/Linux).

---

## 2. (Opsional tapi disarankan) Buat virtual environment

Supaya dependency project ini tidak campur dengan project Python lain di komputer kamu.

```bash
python -m venv venv
```

Aktifkan:
- **Windows (cmd):** `venv\Scripts\activate`
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **Mac/Linux:** `source venv/bin/activate`

Kalau berhasil, akan muncul `(venv)` di depan prompt terminal kamu.

---

## 3. Install dependency

Pastikan file `main.py` dan `requirements.txt` ada di folder yang sama, lalu masuk ke folder itu di terminal:

```bash
cd path/ke/folder/Camera_Game_Controller
```

lalu

```bash
pip install -r requirements.txt
```

Ini akan install:
- `opencv-python` - capture & tampilkan video webcam
- `mediapipe` - deteksi pose/landmark tubuh
- `pyautogui` - simulasi keypress ke game

Proses ini bisa makan waktu beberapa menit (mediapipe ukurannya cukup besar).

> **Mac users:** saat pertama jalan, sistem akan minta izin akses kamera untuk Terminal/IDE kamu. Klik **Allow**. Kalau ketolak, buka System Settings → Privacy & Security → Camera, aktifkan untuk Terminal/VS Code.

---

## 4. Siapkan game-nya

- Kalau target kamu Subway Surfers/Temple Run versi **browser** (misal di poki.com atau situs sejenis): buka di browser, posisikan window-nya supaya kelihatan dan **klik di area game dulu** supaya fokus keyboard ada di situ.

---

## 5. Jalankan script

```bash
python main.py
```

> **Catatan:** MediaPipe baru-baru ini menghapus API lama (`mp.solutions.pose`) dan menggantinya dengan API baru bernama **MediaPipe Tasks**. Script `main.py` di sini sudah memakai API baru tersebut. Saat **pertama kali** dijalankan, script akan otomatis mendownload file model `pose_landmarker_lite.task` (sekitar 5-10MB) - pastikan komputer kamu terkoneksi internet saat menjalankan untuk pertama kali. Setelah file model itu tersimpan di folder yang sama dengan `main.py`, run selanjutnya tidak perlu internet lagi.

Akan muncul window kamera dengan overlay skeleton tubuh kamu.

1. **Berdiri di posisi netral** (tegak, tangan di samping, badan menghadap kamera).
2. Tekan **`c`** untuk kalibrasi. Akan muncul teks "Kalibrasi OK" di window.
3. Sekarang coba gerakkan badan:
   - Geser ke kiri/kanan → trigger tombol `left`/`right`
   - Geser balik ke tengah dari kiri/kanan → otomatis trigger tombol kebalikannya (`right`/`left`) sekali, supaya posisi di game ikut balik ke tengah
   - Angkat **tangan kanan** ke atas melebihi bahu → trigger tombol `up` (lompat)
   - Angkat **tangan kiri** ke atas melebihi bahu → trigger tombol `down` (turun)
4. Klik window game (supaya keyboard fokus ke game), lalu mulai main sambil badan tetap dalam jangkauan kamera.
5. Tekan **`q`** di window kamera untuk keluar kapan saja.

> **Kalau tangan kanan/kiri terbalik saat ditest** (misal angkat tangan kanan tapi yang ke-trigger malah tombol `down`): ini wajar terjadi karena video di-mirror. Buka `main.py`, cari baris `USER_RIGHT_WRIST = 15` dan `USER_LEFT_WRIST = 16`, lalu tukar nilainya jadi `USER_RIGHT_WRIST = 16` dan `USER_LEFT_WRIST = 15`. Simpan, jalankan ulang.

---

## 6. Kalibrasi ulang & tuning sensitivitas

Kalau gerakan terasa kurang sensitif / terlalu sensitif, buka `main.py`, edit bagian konfigurasi paling atas:

| Variabel | Fungsi | Kalau kurang sensitif | Kalau terlalu sensitif |
|---|---|---|---|
| `HORIZONTAL_THRESHOLD` | jarak geser kiri/kanan | turunkan nilainya | naikkan nilainya |
| `HANDS_UP_MARGIN_RATIO` | seberapa tinggi tangan harus diangkat, relatif ke lebar bahu (bukan pixel statis lagi) | turunkan nilainya (cukup angkat sedikit) | naikkan nilainya (harus angkat lebih tinggi) |
| `HANDS_UP_MIN_MARGIN_PX` | batas bawah margin dalam pixel (jaga-jaga kalau badan kecil di frame) | turunkan | naikkan |
| `SMOOTHING_ALPHA` | kehalusan tracking posisi (0-1) | naikkan mendekati 1 (lebih responsif, tapi lebih "geter") | turunkan mendekati 0 (lebih halus, tapi sedikit delay) |
| `MIN_LANDMARK_VISIBILITY` | seberapa yakin model harus terhadap posisi tangan sebelum dipercaya | turunkan (lebih permisif, tapi lebih rawan salah baca) | naikkan (lebih strict, butuh tangan kelihatan jelas) |
| `ACTION_COOLDOWN` | jarak waktu minimal antar-trigger lompat/turun | turunkan (bisa trigger lebih rapat) | naikkan (jeda lebih lama antar aksi) |
| `CONSECUTIVE_FRAMES_NEEDED` | anti-getar/noise untuk kiri/kanan & gesture tangan | turunkan (respon lebih cepat) | naikkan (lebih stabil, agak lambat) |

> Window kamera sekarang menampilkan garis abu-abu (tinggi bahu) dan garis putih putus-putus (ambang batas trigger tangan) - titik tangan kanan jadi **hijau** dan titik tangan kiri jadi **oranye** begitu melewati garis putih itu. Posisikan diri supaya garis putih itu jelas terlihat di antara bahu dan posisi tangan terangkat, lalu sesuaikan `HANDS_UP_MARGIN_RATIO` sambil lihat reaksinya secara real-time.

Kalibrasi ulang (`c`) tiap kali posisi berdiri/jarak ke kamera berubah.

---

## 7. Troubleshooting

| Masalah | Solusi |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Jalankan ulang `pip install -r requirements.txt`, pastikan venv aktif |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Ini bukan bug di script kamu - Google menghapus API lama tersebut di rilis mediapipe terbaru. Pastikan kamu pakai `main.py` versi terbaru dari project ini (sudah memakai API baru "MediaPipe Tasks"), bukan versi lama |
| Gagal download model otomatis / `URLError` saat pertama run | Pastikan ada koneksi internet. Kalau tetap gagal, download manual file `pose_landmarker_lite.task` dari link yang ditampilkan di error, lalu letakkan di folder yang sama dengan `main.py` |
| Gagal install mediapipe / `No matching distribution found` | Python kamu kemungkinan terlalu baru/lama - pakai Python 3.10/3.11 |
| Webcam tidak terdeteksi / error saat `cv2.VideoCapture` | Ganti `CAM_INDEX = 0` jadi `1` atau `2` di `main.py` |
| Window game tidak merespon tombol | Klik dulu di area game supaya jadi window aktif/fokus sebelum bergerak |
| Posisi kiri/kanan ke-trigger terus-menerus | Naikkan `HORIZONTAL_THRESHOLD`, atau kalibrasi ulang di posisi lebih stabil |
| Tangan kanan ke-trigger tombol `down`, tangan kiri ke-trigger tombol `up` (kebalik) | Tukar nilai `USER_RIGHT_WRIST` dan `USER_LEFT_WRIST` di `main.py` (lihat catatan di langkah 5) |
| Posisi tidak balik ke tengah saat badan digeser balik ke tengah | Pastikan pakai `main.py` versi terbaru - versi lama belum menangani transisi balik ke tengah |
| Mau hentikan paksa kalau program "nyangkut" | Geser mouse cepat ke pojok kiri-atas layar (PyAutoGUI failsafe akan stop) |

---

## Catatan untuk pengembangan lanjut (kalau perlu dipresentasikan ke client)

- Logic deteksi saat ini berbasis **threshold sederhana** dari landmark bahu & pinggul - cukup buat demo/POC, real-time, dan ringan (jalan di CPU biasa).
- Untuk versi produksi, bisa ditambah: smoothing (moving average) biar lebih halus, kalibrasi otomatis di awal (auto-detect beberapa detik pertama sebagai baseline), atau ganti target ke versi APK kalau ternyata dibutuhkan integrasi langsung ke aplikasi mobile.
