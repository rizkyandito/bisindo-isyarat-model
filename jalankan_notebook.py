"""Menjalankan Latih_Model_Isyarat.ipynb di laptop memakai data lokal (data_hf/), tanpa Hugging Face.
Notebook hasil eksekusi (lengkap dengan keluaran & grafik) disimpan ke hasil_model/Latih_Model_Isyarat_hasil.ipynb.

    python jalankan_notebook.py
"""

import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("SUMBER_DATA_LOKAL", str(ROOT / "data_hf"))
os.environ.setdefault("FOLDER_HASIL", str(ROOT / "hasil_model"))
os.environ.setdefault("MPLBACKEND", "Agg")
# cache datasets baru tiap kali agar perubahan data_hf/ pasti terbaca (cache lama bisa tertinggal)
import tempfile
os.environ["HF_DATASETS_CACHE"] = tempfile.mkdtemp(prefix="hf_datasets_")

nb = nbformat.read(ROOT / "Latih_Model_Isyarat.ipynb", as_version=4)
t0 = time.time()
try:
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
finally:
    out = Path(os.environ["FOLDER_HASIL"]) / "Latih_Model_Isyarat_hasil.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    print(f"Notebook hasil: {out} ({(time.time() - t0) / 60:.1f} menit)")
