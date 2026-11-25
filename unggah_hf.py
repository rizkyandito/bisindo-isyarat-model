"""Mengunggah dataset (data_hf/) dan/atau model hasil latih (hasil_model/) ke Hugging Face.

Butuh token HF dengan izin "write" (huggingface.co -> Settings -> Access Tokens):

    export HF_TOKEN=hf_xxx
    python unggah_hf.py dataset            # -> <username>/bisindo-isyarat         (repo dataset)
    python unggah_hf.py model              # -> <username>/bisindo-isyarat-model   (repo model)
    python unggah_hf.py dataset --privat   # repo privat
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jenis", choices=["dataset", "model"])
    ap.add_argument("--repo", help="nama repo, default <username>/bisindo-isyarat[-model]")
    ap.add_argument("--folder", type=Path, help="default data_hf/ atau hasil_model/")
    ap.add_argument("--privat", action="store_true")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()["name"]
    repo = args.repo or f"{user}/bisindo-isyarat{'-model' if args.jenis == 'model' else ''}"
    folder = args.folder or ROOT / ("data_hf" if args.jenis == "dataset" else "hasil_model")
    if not folder.exists():
        raise SystemExit(f"{folder} tidak ada")

    api.create_repo(repo, repo_type=args.jenis, private=args.privat, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=args.jenis,
                      commit_message=f"Unggah {args.jenis} dari laptop",
                      ignore_patterns=[".DS_Store", "*.pyc", "__pycache__/*"])
    jalur = "datasets/" if args.jenis == "dataset" else ""
    print(f"Selesai: https://huggingface.co/{jalur}{repo}")


if __name__ == "__main__":
    main()
