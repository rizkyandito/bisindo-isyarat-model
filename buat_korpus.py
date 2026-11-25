"""Membuat korpus paralel isyarat -> kalimat untuk model makna (gloss-to-text).

Setiap contoh: (konteks kalimat sebelumnya, rangkaian isyarat/gloss, ekspresi wajah) -> kalimat Indonesia.
Masukan model ditulis sebagai satu teks:
    "konteks: Apa kabar? | isyarat: SAYA BAIK | wajah: netral"   ->   "Kabar saya baik."

PENTING: korpus ini SINTETIS, disusun dari pola kalimat untuk 40 kosakata dataset Kaggle, karena data lama
belum berisi rekaman kalimat. Asumsi yang dipakai (perlu divalidasi dengan penutur/ahli BISINDO):
  - tanpa kata tanya: alis naik = pertanyaan ya/tidak, wajah lain = pernyataan
  - dengan kata tanya (apa, di mana, kapan, ...): selalu pertanyaan, apa pun ekspresinya. Pada dataset
    Kaggle, video kata tanya justru disertai alis NAIK (bukan turun), jadi semua ekspresi dimasukkan.
  - kata tanya boleh di akhir (KAMU MAKAN APA) maupun di awal (KAPAN KAMU TIDUR)
Ganti/tambah dengan data asli di data_hf/kalimat/asli.csv (kolom sama, sintetis=False).

    python buat_korpus.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "data_hf" / "kalimat"
SEED = 42

PRONOMINA = {"SAYA": "saya", "KAMU": "kamu", "DIA": "dia", "KAMI": "kami", "KITA": "kita",
             "KALIAN": "kalian", "MEREKA": "mereka"}
KERJA = {"MAKAN": "makan", "MINUM": "minum", "MANDI": "mandi", "TIDUR": "tidur", "DUDUK": "duduk",
         "BERDIRI": "berdiri", "BELAJAR": "belajar", "MEMBACA": "membaca", "MENULIS": "menulis",
         "MELIHAT": "melihat"}
TRANSITIF = ["MAKAN", "MINUM", "MEMBACA", "MENULIS", "MELIHAT", "BELAJAR"]
SIFAT = {"SENANG": "senang", "SEDIH": "sedih", "MARAH": "marah", "BINGUNG": "bingung", "SABAR": "sabar",
         "RAMAH": "ramah", "BAIK": "baik", "TINGGI": "tinggi", "PENDEK": "pendek"}
PERASAAN = ["SENANG", "SEDIH", "MARAH", "BINGUNG"]
SALAM = {"HALO": "Halo.", "SELAMAT_PAGI": "Selamat pagi.", "SELAMAT_SIANG": "Selamat siang.",
         "SELAMAT_SORE": "Selamat sore.", "SELAMAT_MALAM": "Selamat malam.", "TERIMA_KASIH": "Terima kasih."}
# pasangan (pronomina, kata kerja) yang TIDAK dipakai saat latih -> menguji kombinasi baru
KOMBINASI_UJI = {("KALIAN", "MEMBACA"), ("MEREKA", "MANDI"), ("KAMI", "MELIHAT"), ("DIA", "BELAJAR"),
                 ("KAMU", "MINUM"), ("SAYA", "MENULIS")}


def kap(s: str) -> str:
    return s[0].upper() + s[1:]


def contoh(isyarat, wajah, kalimat, template, konteks="-"):
    return {"konteks": konteks, "isyarat": isyarat, "wajah": wajah, "kalimat": kalimat, "template": template}


WAJAH_TANYA = ["alis_turun", "alis_naik", "netral"]


def pertanyaan(isyarat, kalimat, template, konteks="-"):
    """Kalimat dengan kata tanya: pertanyaan apa pun ekspresi wajahnya."""
    return [contoh(isyarat, w, kalimat, template, konteks) for w in WAJAH_TANYA]


def bangun() -> list[dict]:
    d = []
    for P, p in PRONOMINA.items():
        for V, v in KERJA.items():
            d += [contoh(f"{P} {V}", "netral", f"{kap(p)} {v}.", "pernyataan"),
                  contoh(f"{P} {V}", "alis_naik", f"Apakah {p} {v}?", "tanya_ya_tidak"),
                  *pertanyaan(f"{P} {V} DIMANA", f"{kap(p)} {v} di mana?", "tanya_dimana"),
                  *pertanyaan(f"{P} {V} KAPAN", f"Kapan {p} {v}?", "tanya_kapan"),
                  *pertanyaan(f"KAPAN {P} {V}", f"Kapan {p} {v}?", "tanya_kapan")]
            if V in TRANSITIF:
                d.extend(pertanyaan(f"{P} {V} APA", f"{kap(p)} {v} apa?", "tanya_apa"))
            d.append(contoh(P, "netral", f"{kap(p)} yang {v}.", "jawab_siapa", konteks=f"Siapa yang {v}?"))
        for S, s in SIFAT.items():
            d += [contoh(f"{P} {S}", "netral", f"{kap(p)} {s}.", "sifat"),
                  contoh(f"{P} {S}", "alis_naik", f"Apakah {p} {s}?", "sifat_tanya")]
        d.extend(pertanyaan(f"{P} KEMANA", f"{kap(p)} mau ke mana?", "tanya_kemana"))
    # konteks percakapan: isyarat tanpa subjek dijawab dengan subjek dari pertanyaan sebelumnya
    # (penanya "kamu" -> penjawab "saya", "kalian" -> "kami", selain itu tetap)
    PENJAWAB = {"kamu": "saya", "kalian": "kami", "dia": "dia", "mereka": "mereka", "kita": "kita"}
    for tanya, jawab in PENJAWAB.items():
        for V, v in KERJA.items():
            d.append(contoh(V, "netral", f"{kap(jawab)} sedang {v}.", "jawab_konteks",
                            konteks=f"{kap(tanya)} sedang apa?"))
        for S in PERASAAN:
            d.append(contoh(S, "netral", f"{kap(jawab)} {SIFAT[S]}.", "jawab_perasaan",
                            konteks=f"Bagaimana perasaan {tanya}?"))
        d.extend(pertanyaan(f"SIAPA {P}", f"Siapa {p}?", "tanya_siapa"))
    for V, v in KERJA.items():
        d += [*pertanyaan(f"SIAPA {V}", f"Siapa yang {v}?", "tanya_siapa"),
              contoh(V, "netral", f"{kap(v)}.", "satu_kata")]
    for P in ("KALIAN", "MEREKA", "KAMI", "KITA"):
        d.extend(pertanyaan(f"{P} BERAPA", f"{kap(PRONOMINA[P])} berapa orang?", "tanya_berapa"))
    # satu isyarat dengan arti yang bergantung konteks: BAIK
    d += [contoh("APA_KABAR", "alis_naik", "Apa kabar?", "salam"),
          contoh("BAIK", "netral", "Kabar saya baik.", "makna_baik_kabar", konteks="Apa kabar?"),
          contoh("SAYA BAIK", "netral", "Kabar saya baik.", "makna_baik_kabar", konteks="Apa kabar?"),
          contoh("SAYA BAIK", "senyum", "Kabar saya baik.", "makna_baik_kabar", konteks="Apa kabar?"),
          contoh("BAIK", "netral", "Baik.", "satu_kata")]
    for P, p in PRONOMINA.items():
        for S2 in ("RAMAH", "SABAR"):
            d.append(contoh(f"{P} BAIK {S2}", "netral", f"{kap(p)} baik dan {SIFAT[S2]}.", "makna_baik_sifat"))
        for S in PERASAAN:
            d.append(contoh(f"{P} {S}", "netral", f"{kap(p)} sedang {SIFAT[S]}.", "jawab_kabar",
                            konteks="Apa kabar?"))
    for S, s in SIFAT.items():
        if S not in ("BAIK",):
            d.append(contoh(S, "netral", f"{kap(s)}.", "satu_kata"))
    # salam
    for G, kalimat in SALAM.items():
        d += [contoh(G, "netral", kalimat, "salam"), contoh(G, "senyum", kalimat, "salam")]
    for G in ("HALO", "SELAMAT_PAGI", "SELAMAT_SIANG", "SELAMAT_SORE", "SELAMAT_MALAM"):
        d.append(contoh(f"{G} APA_KABAR", "alis_naik", f"{SALAM[G][:-1]}, apa kabar?", "salam"))
    # dua kegiatan / dua sifat
    for P, p in PRONOMINA.items():
        for a, b in [("MAKAN", "MINUM"), ("MANDI", "TIDUR"), ("MEMBACA", "MENULIS"), ("DUDUK", "BELAJAR")]:
            d.append(contoh(f"{P} {a} {b}", "netral", f"{kap(p)} {KERJA[a]} dan {KERJA[b]}.", "dua_kegiatan"))
        for a, b in [("SEDIH", "MARAH"), ("SENANG", "RAMAH")]:
            d.append(contoh(f"{P} {a} {b}", "netral", f"{kap(p)} {SIFAT[a]} dan {SIFAT[b]}.", "dua_sifat"))
    return d


def teks_masukan(konteks: str, isyarat: str, wajah: str) -> str:
    return f"konteks: {konteks or '-'} | isyarat: {isyarat} | wajah: {wajah}"


def main():
    df = pd.DataFrame(bangun()).drop_duplicates(subset=["konteks", "isyarat", "wajah"]).reset_index(drop=True)
    df["masukan"] = [teks_masukan(k, i, w) for k, i, w in zip(df.konteks, df.isyarat, df.wajah)]
    df["sintetis"] = True

    def kombinasi_uji(isyarat: str) -> bool:
        t = isyarat.split()
        return any(p in t and v in t for p, v in KOMBINASI_UJI)

    uji_baru = df[df.isyarat.apply(kombinasi_uji)]
    sisa = df.drop(uji_baru.index).sample(frac=1, random_state=SEED)
    n = len(sisa)
    bagian = {"train": sisa.iloc[:int(n * .85)], "validation": sisa.iloc[int(n * .85):int(n * .92)],
              "test": sisa.iloc[int(n * .92):], "test_kombinasi_baru": uji_baru}
    OUT.mkdir(parents=True, exist_ok=True)
    kolom = ["masukan", "kalimat", "konteks", "isyarat", "wajah", "template", "sintetis"]
    for nama, b in bagian.items():
        b[kolom].to_parquet(OUT / f"{nama}.parquet", index=False)
        print(f"{nama:<20} {len(b):>4} contoh")
    print("\nContoh:")
    for _, r in df.sample(6, random_state=1).iterrows():
        print(f"  {r.masukan}  ->  {r.kalimat}")


if __name__ == "__main__":
    main()
