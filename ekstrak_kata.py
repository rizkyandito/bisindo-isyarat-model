"""Ekstraksi fitur (tangan + tubuh + wajah) dari 2.000 video dataset Kaggle "BISINDO 40 Kata mp4".

Hasil: data_hf/kata/<split>.parquet dengan kolom
    id, kata, label, video, nomor, fps, n_frame, fitur (list n_frame x 195 float)

Split tetap berdasarkan nomor video (agar bisa direproduksi):
    train = 001-035, validation = 036-042, test = 043-050

    python ekstrak_kata.py --worker 6
"""

from __future__ import annotations

import argparse
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = Path.home() / ".cache/kagglehub/datasets/anggiyohanespardede/bisindo-40-kata-mp4/versions/1/raw_video"
OUT = Path(__file__).resolve().parent / "data_hf" / "kata"
_pelacak = None


def split_dari_nomor(n: int) -> str:
    return "train" if n <= 35 else "validation" if n <= 42 else "test"


def _init():
    global _pelacak
    from fitur import Pelacak
    _pelacak = Pelacak()


def _kerja(tugas):
    from fitur import Pelacak, ekstrak_video
    global _pelacak
    path, kata = tugas
    # pelacak MediaPipe mode VIDEO butuh timestamp naik; buat ulang per video agar pelacakan tidak terbawa
    _pelacak.tutup()
    _pelacak = Pelacak()
    x, fps = ekstrak_video(path, _pelacak)
    return str(path), kata, x.astype(np.float32), float(fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--worker", type=int, default=6)
    ap.add_argument("--batas", type=int, default=0, help="hanya N video pertama (uji coba)")
    args = ap.parse_args()

    tugas = [(p, d.name) for d in sorted(args.dataset.iterdir()) if d.is_dir() for p in sorted(d.glob("*.mp4"))]
    if args.batas:
        tugas = tugas[:args.batas]
    kelas = sorted({k for _, k in tugas})
    print(f"{len(tugas)} video, {len(kelas)} kata, {args.worker} worker")

    baris, t0 = [], time.time()
    with Pool(args.worker, initializer=_init) as pool:
        for i, (path, kata, x, fps) in enumerate(pool.imap_unordered(_kerja, tugas, chunksize=4), 1):
            nomor = int(re.search(r"_(\d+)\.mp4$", path).group(1))
            baris.append({"id": Path(path).stem, "kata": kata, "label": kelas.index(kata), "video": Path(path).name,
                          "nomor": nomor, "split": split_dari_nomor(nomor), "fps": fps, "n_frame": len(x),
                          "fitur": [r.tolist() for r in x]})
            if i % 50 == 0 or i == len(tugas):
                el = time.time() - t0
                print(f"  {i}/{len(tugas)}  {el / 60:.1f} mnt, sisa ±{el / i * (len(tugas) - i) / 60:.1f} mnt", flush=True)

    df = pd.DataFrame(baris).sort_values("id").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, bagian in df.groupby("split"):
        bagian.drop(columns="split").to_parquet(OUT / f"{split}.parquet", index=False)
        print(f"{split}: {len(bagian)} video -> {OUT / f'{split}.parquet'}")
    tanpa_tangan = (df["fitur"].apply(lambda f: np.asarray(f)[:, 140:142].max() == 0)).sum()
    tanpa_wajah = (df["fitur"].apply(lambda f: np.asarray(f)[:, 194].max() == 0)).sum()
    print(f"video tanpa tangan terdeteksi: {tanpa_tangan}, tanpa wajah terdeteksi: {tanpa_wajah}")


if __name__ == "__main__":
    main()
