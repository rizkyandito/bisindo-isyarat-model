# Model AI Penerjemah Isyarat BISINDO (bermakna sesuai konteks)

Folder ini berisi semua yang dibutuhkan untuk membuat model sendiri: menyiapkan data → mengunggah ke
Hugging Face → melatih di Google Colab → model dipakai di web (halaman **Terjemahkan** di `kamus-web`).

```
video/foto ─► MediaPipe (titik tangan, tubuh, 52 ekspresi wajah) ─┬► model KATA  (BiGRU)  ─► isyarat + 2 alternatif
                                                                  ├► model ABJAD (MLP)    ─► huruf (dari foto)
                                                                  └► ringkasan wajah ─► netral / alis_naik / alis_turun / senyum
konteks percakapan + rangkaian isyarat + label wajah ─► model MAKNA (Transformer) ─► kalimat bahasa Indonesia
```

Kenapa "bermakna": satu gerakan bisa berarti lain tergantung konteks. Contoh dari model yang sudah dilatih:

| Konteks | Isyarat | Wajah | Kalimat |
|---|---|---|---|
| – | DIA BAIK RAMAH | netral | Dia baik dan ramah. |
| "Apa kabar?" | BAIK | netral | Kabar saya baik. |
| – | KAMU MAKAN | netral | Kamu makan. |
| – | KAMU MAKAN | alis naik | Apakah kamu makan? |
| "Kamu sedang apa?" | MAKAN | netral | Saya sedang makan. |

## Isi folder

| File | Fungsi |
|---|---|
| `fitur.py` | ekstraksi 195 fitur per frame (tubuh, tangan, ekspresi wajah) — rumus yang sama dipakai di browser (`kamus-web/static/isyarat.js`) |
| `ekstrak_kata.py` | 2.000 video Kaggle *BISINDO 40 Kata* → `data_hf/kata/*.parquet` |
| `siapkan_abjad.py` | fitur abjad lama (`../features.npz`) → `data_hf/abjad/*.parquet` |
| `buat_korpus.py` | korpus kalimat (sintetis) → `data_hf/kalimat/*.parquet` |
| `data_hf/` | **dataset siap unggah** ke Hugging Face (README = kartu dataset) |
| `Latih_Model_Isyarat.ipynb` | **notebook Colab**: latih 3 model, evaluasi, ekspor ONNX, unggah ke Hugging Face |
| `buat_notebook.py` | sumber notebook (ubah di sini lalu `python buat_notebook.py`) |
| `unggah_hf.py` | unggah dataset / model ke Hugging Face dari laptop |
| `hasil_model/` | model hasil latih (ONNX + config + metrik + gambar evaluasi) |

## Langkah

1. **Siapkan data** (sudah dilakukan; ulangi bila data berubah):
   ```bash
   source .venv/bin/activate
   python ekstrak_kata.py --worker 6     # ±1 jam
   python siapkan_abjad.py
   python buat_korpus.py
   ```
2. **Unggah dataset ke Hugging Face** (token "write" dari huggingface.co → Settings → Access Tokens):
   ```bash
   export HF_TOKEN=hf_xxx
   python unggah_hf.py dataset            # -> https://huggingface.co/datasets/<username>/bisindo-isyarat
   ```
3. **Latih di Colab:** buka https://colab.research.google.com → *Upload* `Latih_Model_Isyarat.ipynb` → isi
   `HF_USER` dan `UNGGAH_KE_HF = True` → *Runtime → Run all*. Model terunggah ke
   `https://huggingface.co/<username>/bisindo-isyarat-model`.
4. **Pakai di web:** salin isi `hasil_model/` (kata/, abjad/, makna/) ke `kamus-web/static/model/`, atau arahkan
   web langsung ke repo model HF dengan `window.KAMUS_MODEL_BASE = "https://huggingface.co/<username>/bisindo-isyarat-model/resolve/main"`.

Menjalankan notebook di laptop (tanpa Hugging Face):
```bash
python jalankan_notebook.py    # memakai data_hf/, hasil di hasil_model/ (+ notebook lengkap dengan keluarannya)
```

## Menambah data asli (penting untuk TA)

- **Kalimat:** buat `data_hf/kalimat/asli.csv` atau tambahkan baris ke korpus dengan kolom
  `konteks, isyarat, wajah, kalimat` (+ `sintetis=False`), sebaiknya dari penutur/ahli BISINDO. Contoh baris:
  `"Apa kabar?", "SAYA BAIK", "netral", "Kabar saya baik."`
- **Isyarat dengan banyak arti:** rekam video isyaratnya dalam beberapa kalimat berbeda; arti dibedakan lewat
  kolom `kalimat` sesuai konteks.
- **Ekspresi wajah:** label wajah saat ini ditentukan dengan ambang sederhana dari data kata (persentil ke-90).
  Rekaman kalimat tanya/pernyataan yang diberi label akan membuat ambang ini lebih tepat.
- **Kata Kolok:** bisa ditambahkan sebagai kelas baru di konfigurasi `kata` (rekam video dengan izin).

## Keterbatasan (tulis di Bab 4/5)

- Model kata dilatih dan diuji pada dataset yang sama (peraga dan latar mirip): akurasi pada orang/kamera lain
  perlu diuji dengan rekaman partisipan.
- Korpus kalimat sintetis, sehingga skor model makna (mendekati 100%) hanya menunjukkan model mempelajari pola
  yang diberikan, belum kemampuan menerjemahkan percakapan nyata.
- Asumsi "alis naik = pertanyaan ya/tidak" perlu divalidasi untuk BISINDO. Analisis data menunjukkan video kata
  tanya disertai alis lebih terangkat (0,39 vs 0,26), bukan alis turun seperti yang semula diasumsikan.
- Fitur wajah tidak menaikkan akurasi pengenalan kata tunggal (98,4% dengan wajah vs 99,1% tanpa wajah);
  perannya ada pada penentuan jenis kalimat di model makna.

## Catatan repo

File besar tidak disimpan di GitHub:

- **Dataset** (`data_hf/**/*.parquet`) ada di Hugging Face, atau buat ulang dengan langkah 1.
- **Model MediaPipe** (`models/*.task`) diunduh ke folder `models/`:
  ```bash
  mkdir -p models && cd models
  curl -LO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  curl -LO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
  curl -LO https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
  ```
