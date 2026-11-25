---
language: id
license: mit
library_name: onnx
tags: [sign-language, bisindo, mediapipe, onnx]
datasets: [ISI_USERNAME_HF/bisindo-isyarat]
---
# Model penerjemah isyarat BISINDO (bermakna sesuai konteks)

Tiga model ONNX untuk dijalankan di browser (onnxruntime-web) setelah ekstraksi landmark MediaPipe.

| Model | Masukan | Keluaran | Hasil uji |
|---|---|---|---|
| `kata/` | 32 frame × 195 fitur (tangan, tubuh, ekspresi wajah) | 40 isyarat | akurasi 98.44% (tanpa wajah 99.06%) |
| `abjad/` | 127 fitur tangan dari 1 foto | huruf A–Z | akurasi 99.63% |
| `makna/` | konteks + rangkaian isyarat + label wajah | kalimat Indonesia | exact match 100.00%, chrF 100.0 (kombinasi baru: 100.00%) |

Data uji kata dan latih berasal dari dataset yang sama (peraga dan latar mirip), dan korpus kalimat masih
sintetis, sehingga kinerja pada penggunaan nyata perlu diuji terpisah. Lihat dataset `ISI_USERNAME_HF/bisindo-isyarat`.
