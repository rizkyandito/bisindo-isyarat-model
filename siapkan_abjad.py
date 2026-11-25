"""Konversi fitur abjad lama (features.npz, 2.700 foto A-Z rekaman sendiri) ke format Hugging Face.

Hanya titik tangan yang disimpan (127 angka per foto), bukan fotonya, sehingga wajah peraga tidak ikut
terunggah. Fitur: 2 tangan x 21 titik x (x,y,z), diurutkan kiri->kanan di gambar, digeser ke pusat
gabungan kedua tangan dan dibagi ukuran kotak pembatasnya, + 1 penanda "dua tangan" (bisindo_common.py).

    python siapkan_abjad.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_hf" / "abjad"

d = np.load(ROOT.parent / "features.npz", allow_pickle=True)
df = pd.DataFrame({"huruf": d["y"], "label": [ord(c) - ord("A") for c in d["y"]],
                   "foto": [Path(s).name for s in d["sources"]], "fitur": [r.tolist() for r in d["X"]]})
train, sisa = train_test_split(df, test_size=0.2, stratify=df.huruf, random_state=42)
val, test = train_test_split(sisa, test_size=0.5, stratify=sisa.huruf, random_state=42)
OUT.mkdir(parents=True, exist_ok=True)
for nama, b in [("train", train), ("validation", val), ("test", test)]:
    b.sort_values("foto").to_parquet(OUT / f"{nama}.parquet", index=False)
    print(f"{nama:<11} {len(b)} foto")
