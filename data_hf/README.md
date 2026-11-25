---
license: mit
language:
- id
pretty_name: BISINDO Isyarat — landmark tangan, tubuh, dan ekspresi wajah
task_categories:
- video-classification
- translation
tags:
- sign-language
- bisindo
- indonesian-sign-language
- mediapipe
- landmarks
configs:
- config_name: kata
  data_files:
  - split: train
    path: kata/train.parquet
  - split: validation
    path: kata/validation.parquet
  - split: test
    path: kata/test.parquet
- config_name: abjad
  data_files:
  - split: train
    path: abjad/train.parquet
  - split: validation
    path: abjad/validation.parquet
  - split: test
    path: abjad/test.parquet
- config_name: kalimat
  data_files:
  - split: train
    path: kalimat/train.parquet
  - split: validation
    path: kalimat/validation.parquet
  - split: test
    path: kalimat/test.parquet
  - split: test_kombinasi_baru
    path: kalimat/test_kombinasi_baru.parquet
---

# BISINDO Isyarat: landmark tangan, tubuh, dan ekspresi wajah

Dataset untuk melatih penerjemah Bahasa Isyarat Indonesia (BISINDO) yang memperhatikan **makna sesuai
konteks**: satu gerakan bisa bermakna berbeda tergantung kalimat dan ekspresi wajah. Dibuat untuk Tugas
Akhir *Pengembangan Aplikasi Kamus Bahasa Isyarat Desa Bengkala–BISINDO* (Teknik Biomedis, Telkom University).

Dataset ini **tidak berisi video atau foto**, hanya titik-titik (landmark) hasil MediaPipe, sehingga wajah
peraga tidak ikut tersebar.

## Konfigurasi

### `kata` — 2.000 video isyarat kata (40 kata × 50 video)

Sumber: dataset Kaggle [BISINDO 40 Kata mp4](https://www.kaggle.com/datasets/anggiyohanespardede/bisindo-40-kata-mp4)
(lisensi MIT). Setiap frame video diubah menjadi 195 angka:

| Indeks | Isi |
|---|---|
| 0–13 | 7 titik tubuh atas (hidung, bahu, siku, pergelangan kiri/kanan) × (x, y) |
| 14–139 | 2 tangan × 21 titik × (x, y, z); slot 0 = tangan di sisi pergelangan kiri tubuh, slot 1 = kanan |
| 140–141 | penanda tangan slot 0 / slot 1 terdeteksi |
| 142–193 | 52 skor ekspresi wajah (MediaPipe face blendshapes, 0–1), urutan di `spesifikasi_fitur.json` |
| 194 | penanda wajah terdeteksi |

Koordinat dinormalisasi terhadap tubuh: dikurangi titik tengah kedua bahu, lalu dibagi lebar bahu.

Kolom: `id`, `kata`, `label` (0–39, urut abjad), `video`, `nomor`, `fps`, `n_frame`, `fitur` (n_frame × 195).
Split berdasarkan nomor video: train 001–035, validation 036–042, test 043–050.

### `abjad` — 2.700 foto huruf A–Z

Rekaman sendiri (±100 foto per huruf). Fitur 127 angka: 2 tangan × 21 titik × (x, y, z), diurutkan kiri→kanan
di gambar, dinormalisasi terhadap pusat dan ukuran kotak pembatas kedua tangan, + penanda dua tangan.
Kolom: `huruf`, `label`, `foto`, `fitur`. Split acak terstratifikasi 80/10/10.

### `kalimat` — korpus paralel isyarat → kalimat (sintetis)

Masukan berisi konteks kalimat sebelumnya, rangkaian isyarat (gloss), dan ekspresi wajah; target berupa
kalimat bahasa Indonesia:

```
konteks: Apa kabar? | isyarat: SAYA BAIK | wajah: netral      ->  Kabar saya baik.
konteks: - | isyarat: DIA BAIK RAMAH | wajah: netral           ->  Dia baik dan ramah.
konteks: - | isyarat: KAMU MAKAN | wajah: alis_naik            ->  Apakah kamu makan?
konteks: Kamu sedang apa? | isyarat: MAKAN | wajah: netral     ->  Saya sedang makan.
```

**Korpus ini sintetis**, disusun dari pola kalimat untuk 40 kosakata di atas, karena belum ada rekaman
kalimat. Asumsi yang dipakai: tanpa kata tanya, alis naik = pertanyaan ya/tidak; dengan kata tanya (apa,
di mana, kapan, …) kalimat selalu pertanyaan apa pun ekspresinya. Analisis data `kata` menunjukkan video kata
tanya disertai alis lebih terangkat (rata-rata skor 0,39 vs 0,26 pada kata lain). Asumsi ini perlu divalidasi dengan penutur atau ahli BISINDO, dan korpus perlu ditambah data asli
(`sintetis = False`). Split `test_kombinasi_baru` berisi pasangan subjek–kata kerja yang tidak pernah muncul
saat latih, untuk menguji kemampuan generalisasi.

## Keterbatasan

- Video kata berasal dari sedikit peraga dengan latar yang mirip; kinerja pada orang, latar, dan kamera lain
  bisa lebih rendah.
- Ekspresi wajah pada dataset kata belum diberi label; pemetaan ekspresi → `wajah` memakai aturan ambang
  sederhana (lihat notebook pelatihan).
- Kata Kolok (bahasa isyarat Desa Bengkala) belum termasuk.
