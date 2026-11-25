"""Membuat notebook Colab Latih_Model_Isyarat.ipynb dari daftar sel di bawah.

    python buat_notebook.py
"""

import nbformat as nbf

SEL = []


def md(teks):
    SEL.append(nbf.v4.new_markdown_cell(teks.strip()))


def code(teks):
    SEL.append(nbf.v4.new_code_cell(teks.strip()))


md("""
# Pelatihan Model Penerjemah Isyarat BISINDO (bermakna sesuai konteks)

Notebook ini melatih tiga model dari dataset di Hugging Face, lalu mengunggah hasilnya kembali ke Hugging Face:

| Model | Masukan | Keluaran | Arsitektur |
|---|---|---|---|
| **kata** | urutan 32 frame × 195 fitur (tangan + tubuh + ekspresi wajah) | 1 dari 40 isyarat | BiGRU |
| **abjad** | 1 foto → 127 fitur tangan | huruf A–Z | MLP |
| **makna** | konteks kalimat + rangkaian isyarat + ekspresi wajah | kalimat bahasa Indonesia | Transformer encoder–decoder kecil (dilatih dari nol) |

Alur di aplikasi: kamera/video → MediaPipe (titik tangan, tubuh, wajah) → **model kata** mengenali setiap isyarat
→ ekspresi wajah diringkas jadi label (`netral`, `alis_naik`, `alis_turun`, `senyum`) → **model makna** menyusun
kalimat dengan memperhatikan konteks. Foto → **model abjad**.

Semua model diekspor ke **ONNX** agar bisa dijalankan langsung di browser.

**Cara pakai di Colab:** *Runtime → Change runtime type → T4 GPU* (opsional, CPU juga cukup), isi `HF_USER`
di sel Konfigurasi, lalu *Runtime → Run all*.
""")

md("## 0. Persiapan")
code("""
import os, sys, json, math, random, time
DI_COLAB = "google.colab" in sys.modules
if DI_COLAB:
    !pip -q install datasets huggingface_hub onnx onnxruntime onnxscript sacrebleu
""")
code("""
# ===== Konfigurasi =====
HF_USER = "ISI_USERNAME_HF"                       # ganti dengan username Hugging Face Anda
DATASET_REPO = f"{HF_USER}/bisindo-isyarat"       # dataset (sudah diunggah dari laptop)
MODEL_REPO = f"{HF_USER}/bisindo-isyarat-model"   # tujuan unggah model
UNGGAH_KE_HF = False                              # True = unggah model ke Hugging Face di akhir

# untuk menjalankan di laptop tanpa Hugging Face: set env SUMBER_DATA_LOKAL ke folder data_hf
SUMBER_DATA = os.environ.get("SUMBER_DATA_LOKAL") or DATASET_REPO
OUT = os.environ.get("FOLDER_HASIL", "hasil_model")

EPOCH_KATA = int(os.environ.get("EPOCH_KATA", 60))
EPOCH_ABJAD = int(os.environ.get("EPOCH_ABJAD", 60))
EPOCH_MAKNA = int(os.environ.get("EPOCH_MAKNA", 150))
SEED = 42
""")
code("""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
print("device:", DEVICE, "| data:", SUMBER_DATA)

if UNGGAH_KE_HF or not os.path.isdir(SUMBER_DATA):
    from huggingface_hub import notebook_login, whoami
    try:
        whoami()
    except Exception:
        notebook_login()   # tempel token HF (Settings -> Access Tokens, izin "write")
""")

md("""
## 1. Model kata (pengenal isyarat bergerak)

### 1.1 Memuat data dan praproses

Setiap video berisi `n_frame × 195` fitur. Praproses sama dengan yang dilakukan di browser:
1. celah tangan yang hilang sesaat (≤ 5 frame) diisi posisi terakhir;
2. dipotong ke rentang saat tangan terangkat (pergelangan di atas 1,3 × lebar bahu di bawah bahu);
3. di-*resample* menjadi 32 frame.
""")
code("""
ds_kata = load_dataset(SUMBER_DATA, "kata")
print(ds_kata)

P, H, F_TANGAN, N_WAJAH, F_TOTAL = 14, 126, 142, 52, 195
SEQ_LEN, ACTIVE_Y = 32, 1.3
KELAS = sorted(set(ds_kata["train"]["kata"]))
GLOSS = [k.upper().replace(" ", "_") for k in KELAS]
print(len(KELAS), "kata:", ", ".join(KELAS))

def isi_celah(seq, maks=5):
    seq = seq.copy()
    for i in range(2):
        kol, flag = slice(P + i * 63, P + (i + 1) * 63), P + H + i
        terlihat = seq[:, flag].copy()
        akhir, hilang = None, 0
        for t in range(len(seq)):
            if terlihat[t] > 0:
                akhir, hilang = t, 0
            elif akhir is not None and hilang < maks:
                seq[t, kol] = seq[akhir, kol]; seq[t, flag] = 1.0; hilang += 1
    return seq

def aktif(f):
    return any(f[P + i * 63 + 1] < ACTIVE_Y for i in range(2) if f[P + H + i] > 0)

def potong(seq):
    seq = isi_celah(seq)
    idx = np.flatnonzero([aktif(f) for f in seq])
    return seq[idx[0]:idx[-1] + 1] if len(idx) else seq

def resample(seq, n=SEQ_LEN):
    return seq[np.round(np.linspace(0, len(seq) - 1, n)).astype(int)]

def siapkan(split):
    X, y, mentah = [], [], []
    for r in ds_kata[split]:
        seq = potong(np.asarray(r["fitur"], np.float32))
        X.append(resample(seq)); y.append(KELAS.index(r["kata"])); mentah.append(seq)
    return np.stack(X), np.array(y), mentah

Xtr, ytr, mtr = siapkan("train"); Xva, yva, mva = siapkan("validation"); Xte, yte, mte = siapkan("test")
print("train", Xtr.shape, "validation", Xva.shape, "test", Xte.shape)
""")

md("""
### 1.2 Analisis ekspresi wajah

Apakah ekspresi wajah membawa informasi? Dibandingkan rata-rata gerak alis saat berisyarat antara **kata tanya**
(Apa, Siapa, Dimana, …) dan kata lain. Hasil ini juga dipakai menentukan ambang label wajah.
""")
code("""
BS = json.load(open(os.path.join(SUMBER_DATA, "spesifikasi_fitur.json")))["blendshape"] if os.path.isdir(SUMBER_DATA) \\
    else json.load(open(__import__("huggingface_hub").hf_hub_download(DATASET_REPO, "spesifikasi_fitur.json", repo_type="dataset")))["blendshape"]
bi = lambda n: F_TANGAN + BS.index(n)
SKOR_WAJAH = {
    "alis_naik": [bi("browInnerUp"), bi("browOuterUpLeft"), bi("browOuterUpRight")],
    "alis_turun": [bi("browDownLeft"), bi("browDownRight")],
    "senyum": [bi("mouthSmileLeft"), bi("mouthSmileRight")],
}

def skor_wajah(seq):
    wajah = seq[seq[:, F_TOTAL - 1] > 0]
    if not len(wajah):
        return {k: 0.0 for k in SKOR_WAJAH}
    return {k: float(wajah[:, idx].mean()) for k, idx in SKOR_WAJAH.items()}

KATA_TANYA = {"Apa", "Apa Kabar", "Bagaimana", "Berapa", "Dimana", "Kapan", "Kemana", "Siapa"}
semua_skor = [(KELAS[l], skor_wajah(s)) for l, s in zip(np.concatenate([ytr, yva]), mtr + mva)]
import pandas as pd
dfw = pd.DataFrame([{"kata": k, "kata_tanya": k in KATA_TANYA, **s} for k, s in semua_skor])
ringkas = dfw.groupby("kata_tanya")[list(SKOR_WAJAH)].agg(["mean", "std"]).round(3)
print(ringkas)
per_kata = dfw.groupby("kata")[list(SKOR_WAJAH)].mean().sort_values("alis_naik")
ax = per_kata.plot.barh(figsize=(8, 10), title="Rata-rata skor ekspresi wajah per kata")
plt.tight_layout(); plt.savefig(f"{OUT}/analisis_wajah.png", dpi=120); plt.show()
""")
code("""
# Ambang label wajah: persentil ke-90 skor pada kata non-tanya (ekspresi di atasnya dianggap menonjol).
# Ini aturan awal; sebaiknya dikalibrasi ulang dengan rekaman kalimat yang ekspresinya diberi label.
non_tanya = dfw[~dfw.kata_tanya]
AMBANG_WAJAH = {k: round(float(np.percentile(non_tanya[k], 90)), 4) for k in SKOR_WAJAH}
print("ambang:", AMBANG_WAJAH)

def label_wajah(skor):
    lebih = {k: skor[k] - AMBANG_WAJAH[k] for k in SKOR_WAJAH if skor[k] > AMBANG_WAJAH[k]}
    return max(lebih, key=lebih.get) if lebih else "netral"

dfw["label"] = [label_wajah(s) for _, s in semua_skor]
print(pd.crosstab(dfw.kata_tanya, dfw.label, normalize="index").round(2))
""")

md("""
### 1.3 Model dan pelatihan

BiGRU 2 lapis (hidden 128). Augmentasi: cermin kiri↔kanan (agar pengguna kidal terbaca), derau kecil, dan
skala acak. Dilatih dua kali untuk perbandingan (ablasi): **dengan** dan **tanpa** fitur wajah.
""")
code("""
POSE_SWAP = [0, 2, 1, 4, 3, 6, 5]
BS_SWAP = [BS.index(n.replace("Left", "#").replace("Right", "Left").replace("#", "Right")) for n in BS]

def cermin(X):
    X = X.copy()
    pose = X[..., :P].reshape(X.shape[:-1] + (7, 2)); pose[..., 0] *= -1
    X[..., :P] = pose[..., POSE_SWAP, :].reshape(X.shape[:-1] + (P,))
    tangan = X[..., P:P + H].reshape(X.shape[:-1] + (2, 21, 3)); tangan[..., 0] *= -1
    X[..., P:P + H] = tangan[..., ::-1, :, :].reshape(X.shape[:-1] + (H,))
    X[..., P + H:F_TANGAN] = X[..., P + H:F_TANGAN][..., ::-1]
    X[..., F_TANGAN:F_TANGAN + N_WAJAH] = X[..., F_TANGAN:F_TANGAN + N_WAJAH][..., BS_SWAP]
    return X

class SignGRU(nn.Module):
    def __init__(self, n_fitur, n_kelas, hidden=128, lapis=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(n_fitur, hidden, num_layers=lapis, batch_first=True, dropout=dropout, bidirectional=True)
        self.kepala = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_kelas))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.kepala(out.mean(dim=1))

def latih_kata(kolom, nama, epoch=EPOCH_KATA):
    torch.manual_seed(SEED)
    A = np.concatenate([Xtr, cermin(Xtr)])[..., kolom]; B = np.concatenate([ytr, ytr])
    model = SignGRU(len(kolom), len(KELAS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epoch * math.ceil(len(A) / 64))
    xv = torch.tensor(Xva[..., kolom], device=DEVICE); terbaik, state, riwayat = -1, None, []
    for ep in range(epoch):
        model.train(); urut = np.random.permutation(len(A))
        for i in range(0, len(A), 64):
            b = urut[i:i + 64]
            xb = torch.tensor(A[b], device=DEVICE); yb = torch.tensor(B[b], device=DEVICE)
            xb = xb * (1 + 0.1 * (torch.rand(len(b), 1, 1, device=DEVICE) - 0.5)) + 0.01 * torch.randn_like(xb)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(1).cpu().numpy() == yva).mean()
        riwayat.append((float(loss), float(acc)))
        if acc >= terbaik:
            terbaik, state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"[{nama}] epoch {ep + 1:>3}  loss {float(loss):.3f}  akurasi validasi {acc:.3f}")
    model.load_state_dict(state); model.eval()
    return model, riwayat

def uji_kata(model, kolom):
    with torch.no_grad():
        p = model(torch.tensor(Xte[..., kolom], device=DEVICE)).argmax(1).cpu().numpy()
    return p, accuracy_score(yte, p), f1_score(yte, p, average="macro")

KOLOM_PENUH = list(range(F_TOTAL))
KOLOM_TANPA_WAJAH = list(range(F_TANGAN))
t0 = time.time()
model_kata, riwayat_kata = latih_kata(KOLOM_PENUH, "dengan wajah")
model_tw, _ = latih_kata(KOLOM_TANPA_WAJAH, "tanpa wajah")
print(f"waktu latih: {(time.time() - t0) / 60:.1f} menit")
""")

md("### 1.4 Evaluasi pada data uji (video 043–050)")
code("""
pred, acc, f1 = uji_kata(model_kata, KOLOM_PENUH)
pred_tw, acc_tw, f1_tw = uji_kata(model_tw, KOLOM_TANPA_WAJAH)
HASIL_KATA = {"dengan_wajah": {"akurasi": round(acc, 4), "f1_makro": round(f1, 4)},
              "tanpa_wajah": {"akurasi": round(acc_tw, 4), "f1_makro": round(f1_tw, 4)},
              "jumlah_uji": int(len(yte))}
print(pd.DataFrame(HASIL_KATA).drop(columns="jumlah_uji").T)
print(classification_report(yte, pred, target_names=KELAS, digits=3, zero_division=0))

cm = confusion_matrix(yte, pred)
plt.figure(figsize=(11, 10)); plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(KELAS)), KELAS, rotation=90, fontsize=7); plt.yticks(range(len(KELAS)), KELAS, fontsize=7)
plt.xlabel("Prediksi"); plt.ylabel("Sebenarnya"); plt.title(f"Confusion matrix model kata (akurasi {acc:.1%})")
plt.colorbar(); plt.tight_layout(); plt.savefig(f"{OUT}/confusion_kata.png", dpi=120); plt.show()
plt.figure(figsize=(6, 3)); plt.plot([a for _, a in riwayat_kata]); plt.xlabel("epoch"); plt.ylabel("akurasi validasi")
plt.title("Kurva pelatihan model kata"); plt.tight_layout(); plt.savefig(f"{OUT}/kurva_kata.png", dpi=120); plt.show()
""")

md("""
## 2. Model abjad (foto → huruf)
""")
code("""
ds_abjad = load_dataset(SUMBER_DATA, "abjad")
HURUF = [chr(ord("A") + i) for i in range(26)]
ambil = lambda s: (np.array(ds_abjad[s]["fitur"], np.float32), np.array(ds_abjad[s]["label"]))
Atr, btr = ambil("train"); Ava, bva = ambil("validation"); Ate, bte = ambil("test")

def cermin_abjad(X):
    tangan = X[:, :-1].reshape(-1, 2, 21, 3).copy(); dua = X[:, -1] > 0.5
    tangan[..., 0] *= -1; tangan[dua] = tangan[dua][:, ::-1]
    return np.concatenate([tangan.reshape(len(X), -1), X[:, -1:]], 1)

model_abjad = nn.Sequential(nn.Linear(127, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
                            nn.Linear(64, 26)).to(DEVICE)
A = torch.tensor(np.concatenate([Atr, cermin_abjad(Atr)]), device=DEVICE); B = torch.tensor(np.concatenate([btr, btr]), device=DEVICE)
opt = torch.optim.AdamW(model_abjad.parameters(), lr=2e-3, weight_decay=1e-4); terbaik, state = -1, None
for ep in range(EPOCH_ABJAD):
    model_abjad.train(); urut = torch.randperm(len(A), device=DEVICE)
    for i in range(0, len(A), 64):
        b = urut[i:i + 64]
        loss = F.cross_entropy(model_abjad(A[b] + 0.01 * torch.randn_like(A[b])), B[b])
        opt.zero_grad(); loss.backward(); opt.step()
    model_abjad.eval()
    with torch.no_grad():
        acc = (model_abjad(torch.tensor(Ava, device=DEVICE)).argmax(1).cpu().numpy() == bva).mean()
    if acc >= terbaik:
        terbaik, state = acc, {k: v.clone() for k, v in model_abjad.state_dict().items()}
model_abjad.load_state_dict(state); model_abjad.eval()
with torch.no_grad():
    p = model_abjad(torch.tensor(Ate, device=DEVICE)).argmax(1).cpu().numpy()
HASIL_ABJAD = {"akurasi": round(accuracy_score(bte, p), 4), "f1_makro": round(f1_score(bte, p, average="macro"), 4),
               "jumlah_uji": int(len(bte))}
print(HASIL_ABJAD)
""")

md("""
## 3. Model makna (rangkaian isyarat + konteks + wajah → kalimat)

Transformer encoder–decoder kecil yang dilatih dari nol. Masukan disusun sebagai token:

`<konteks> apa kabar ? <isyarat> SAYA BAIK <wajah> wajah_netral` → `kabar saya baik .`

Dengan begitu isyarat yang sama (mis. **BAIK**) bisa menghasilkan kalimat berbeda sesuai konteks
("Kabar saya baik." vs "Dia baik dan ramah."), dan ekspresi wajah mengubah jenis kalimat
(pernyataan vs pertanyaan).

> Korpus `kalimat` saat ini **sintetis** (lihat kartu dataset). Tambahkan data asli dari penutur BISINDO.
""")
code("""
import re
ds_kal = load_dataset(SUMBER_DATA, "kalimat")
print(ds_kal)

def tok_teks(s):
    return re.findall(r"[a-z0-9]+|[?.,!]", (s or "").lower())

def tok_masukan(konteks, isyarat, wajah):
    k = [] if (konteks or "-").strip() == "-" else tok_teks(konteks)
    return ["<konteks>"] + k + ["<isyarat>"] + isyarat.split() + ["<wajah>", f"wajah_{wajah}"]

def detok(tokens):
    s = ""
    for t in tokens:
        s += t if (t in "?.,!" or not s) else " " + t
    return s[:1].upper() + s[1:]

KHUSUS = ["<pad>", "<bos>", "<eos>", "<unk>", "<konteks>", "<isyarat>", "<wajah>"]
kosakata = set()
for r in ds_kal["train"]:
    kosakata.update(tok_masukan(r["konteks"], r["isyarat"], r["wajah"])); kosakata.update(tok_teks(r["kalimat"]))
kosakata.update(GLOSS)                                       # semua isyarat yang bisa dikenali model kata
kosakata.update(f"wajah_{w}" for w in ["netral", "alis_naik", "alis_turun", "senyum"])
ITOS = KHUSUS + sorted(kosakata - set(KHUSUS)); STOI = {t: i for i, t in enumerate(ITOS)}
PAD, BOS, EOS, UNK = 0, 1, 2, 3
MAKS_SRC, MAKS_TGT = 48, 24
enc = lambda toks: [STOI.get(t, UNK) for t in toks]
print("ukuran kosakata:", len(ITOS))

def ke_tensor(split):
    src = [enc(tok_masukan(r["konteks"], r["isyarat"], r["wajah"]))[:MAKS_SRC] for r in ds_kal[split]]
    tgt = [[BOS] + enc(tok_teks(r["kalimat"]))[:MAKS_TGT - 2] + [EOS] for r in ds_kal[split]]
    pad = lambda xs, n: torch.tensor([x + [PAD] * (n - len(x)) for x in xs])
    return pad(src, max(map(len, src))), pad(tgt, max(map(len, tgt)))
""")
code("""
class Perhatian(nn.Module):
    # multi-head attention dari operasi dasar (agar ekspor ONNX mendukung panjang kalimat dinamis)
    def __init__(self, d, kepala, dropout):
        super().__init__()
        self.h, self.dk = kepala, d // kepala
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)
    def forward(self, x, mem, mask):                       # mask: tambahan (0 / -1e9), bisa di-broadcast
        B = x.shape[0]
        pisah = lambda t: t.view(B, -1, self.h, self.dk).transpose(1, 2)
        q, k, v = pisah(self.q(x)), pisah(self.k(mem)), pisah(self.v(mem))
        a = self.drop(torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk) + mask, dim=-1))
        return self.o((a @ v).transpose(1, 2).reshape(B, -1, self.h * self.dk))

class LapisEncoder(nn.Module):
    def __init__(self, d, kepala, ff, dropout):
        super().__init__()
        self.sa = Perhatian(d, kepala, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
    def forward(self, x, mask):
        x = self.n1(x + self.drop(self.sa(x, x, mask)))
        return self.n2(x + self.drop(self.ff(x)))

class LapisDecoder(nn.Module):
    def __init__(self, d, kepala, ff, dropout):
        super().__init__()
        self.sa, self.ca = Perhatian(d, kepala, dropout), Perhatian(d, kepala, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2, self.n3, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
    def forward(self, y, mem, mask_y, mask_mem):
        y = self.n1(y + self.drop(self.sa(y, y, mask_y)))
        y = self.n2(y + self.drop(self.ca(y, mem, mask_mem)))
        return self.n3(y + self.drop(self.ff(y)))

class Makna(nn.Module):
    # Transformer encoder-decoder kecil: token masukan (konteks, isyarat, wajah) -> token kalimat
    def __init__(self, v, d=128, kepala=4, lapis=3, ff=256, dropout=0.1, maks=64):
        super().__init__()
        self.emb = nn.Embedding(v, d, padding_idx=PAD); self.pos = nn.Embedding(maks, d); self.d = d
        self.enc = nn.ModuleList(LapisEncoder(d, kepala, ff, dropout) for _ in range(lapis))
        self.dec = nn.ModuleList(LapisDecoder(d, kepala, ff, dropout) for _ in range(lapis))
        self.keluar = nn.Linear(d, v, bias=False); self.keluar.weight = self.emb.weight   # bobot berbagi
        self.register_buffer("posisi", torch.arange(maks), persistent=False)
        self.register_buffer("kausal", torch.triu(torch.full((maks, maks), -1e9), diagonal=1), persistent=False)
    def sematkan(self, x):
        return self.emb(x) * math.sqrt(self.d) + self.pos(self.posisi[:x.shape[1]])[None]
    def forward(self, src, tgt):
        mask_src = (src == PAD).float()[:, None, None, :] * -1e9      # abaikan padding masukan
        T = tgt.shape[1]
        mask_tgt = self.kausal[:T, :T][None, None] + (tgt == PAD).float()[:, None, None, :] * -1e9
        m = self.sematkan(src)
        for l in self.enc:
            m = l(m, mask_src)
        y = self.sematkan(tgt)
        for l in self.dec:
            y = l(y, m, mask_tgt, mask_src)
        return self.keluar(y)

@torch.no_grad()
def terjemahkan(model, konteks, isyarat, wajah, maks=MAKS_TGT):
    model.eval()
    src = torch.tensor([enc(tok_masukan(konteks, isyarat, wajah))[:MAKS_SRC]], device=DEVICE)
    out = [BOS]
    for _ in range(maks):
        logit = model(src, torch.tensor([out], device=DEVICE))[0, -1]
        nxt = int(logit.argmax())
        if nxt == EOS:
            break
        out.append(nxt)
    return detok([ITOS[i] for i in out[1:]])

torch.manual_seed(SEED)
model_makna = Makna(len(ITOS)).to(DEVICE)
S, T = ke_tensor("train")
opt = torch.optim.AdamW(model_makna.parameters(), lr=5e-4, weight_decay=1e-2)
langkah = EPOCH_MAKNA * math.ceil(len(S) / 32)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=langkah, pct_start=0.1)
t0 = time.time()
for ep in range(EPOCH_MAKNA):
    model_makna.train(); urut = torch.randperm(len(S))
    for i in range(0, len(S), 32):
        b = urut[i:i + 32]; s, t = S[b].to(DEVICE), T[b].to(DEVICE)
        logit = model_makna(s, t[:, :-1])
        loss = F.cross_entropy(logit.reshape(-1, logit.shape[-1]), t[:, 1:].reshape(-1), ignore_index=PAD,
                               label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model_makna.parameters(), 1.0); opt.step(); sched.step()
    if (ep + 1) % 25 == 0:
        print(f"epoch {ep + 1:>3}  loss {float(loss):.3f}")
print(f"waktu latih: {(time.time() - t0) / 60:.1f} menit, parameter: {sum(p.numel() for p in model_makna.parameters()):,}")
""")
code("""
import sacrebleu
def evaluasi_makna(split):
    hip, ref = [], []
    for r in ds_kal[split]:
        hip.append(terjemahkan(model_makna, r["konteks"], r["isyarat"], r["wajah"])); ref.append(r["kalimat"])
    return {"exact_match": round(float(np.mean([h == r for h, r in zip(hip, ref)])), 4),
            "bleu": round(sacrebleu.corpus_bleu(hip, [ref]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hip, [ref]).score, 2), "jumlah": len(ref)}, list(zip(hip, ref))

HASIL_MAKNA = {}
for split in ["test", "test_kombinasi_baru"]:
    HASIL_MAKNA[split], contoh = evaluasi_makna(split)
    print(split, HASIL_MAKNA[split])
    for h, r in contoh[:5]:
        print(f"   {'✓' if h == r else '✗'} {h!r:42} (target {r!r})")

print("\\nSatu isyarat, makna berbeda sesuai konteks dan wajah:")
for k, i, w in [("-", "DIA BAIK RAMAH", "netral"), ("Apa kabar?", "BAIK", "netral"), ("-", "BAIK", "netral"),
                ("-", "KAMU MAKAN", "netral"), ("-", "KAMU MAKAN", "alis_naik"), ("Kamu sedang apa?", "MAKAN", "netral"),
                ("-", "KAMU MAKAN APA", "alis_turun")]:
    print(f"   konteks={k!r:20} isyarat={i!r:18} wajah={w:<10} -> {terjemahkan(model_makna, k, i, w)}")
""")

md("""
## 4. Uji ujung-ke-ujung

Dua video uji (mis. **Kamu** lalu **Makan**) dikenali satu per satu oleh model kata, ekspresi wajah diringkas,
lalu model makna menyusun kalimat.
""")
code("""
def kenali(seq_mentah):
    with torch.no_grad():
        p = torch.softmax(model_kata(torch.tensor(resample(seq_mentah)[None], device=DEVICE)), 1)[0].cpu().numpy()
    top = np.argsort(-p)[:3]
    return [(GLOSS[i], float(p[i])) for i in top]

def demo(kata_kata, konteks="-"):
    gloss, skor = [], []
    for k in kata_kata:
        i = next(j for j, l in enumerate(yte) if KELAS[l] == k)
        hasil = kenali(mte[i]); gloss.append(hasil[0][0]); skor.append(skor_wajah(mte[i]))
        print(f"  video {k!r:<10} -> {hasil}")
    rata = {k: float(np.mean([s[k] for s in skor])) for k in SKOR_WAJAH}
    wajah = label_wajah(rata)
    print(f"  isyarat: {' '.join(gloss)} | wajah: {wajah} | konteks: {konteks}")
    print("  kalimat:", terjemahkan(model_makna, konteks, " ".join(gloss), wajah))

demo(["Kamu", "Makan"]); print(); demo(["Saya", "Minum"]); print(); demo(["Makan"], konteks="Kamu sedang apa?")
""")

md("""
## 5. Ekspor ONNX dan unggah ke Hugging Face

Hasil di folder `hasil_model/`:

```
kata/model.onnx    kata/config.json     (masukan x: [1, 32, 195] -> logits [1, 40])
abjad/model.onnx   abjad/config.json    (masukan x: [1, 127] -> logits [1, 26])
makna/model.onnx   makna/config.json    (masukan src [1, S], tgt [1, T] -> logits [1, T, V]; dekode greedy)
metrik.json, README.md, gambar evaluasi
```
""")
code("""
import onnxruntime as ort

def ekspor(model, contoh, path, nama_masuk, sumbu):
    model = model.cpu().eval()
    torch.onnx.export(model, contoh, path, input_names=nama_masuk, output_names=["logits"],
                      dynamic_axes=sumbu, opset_version=17, dynamo=False)
    sesi = ort.InferenceSession(path)
    keluaran = sesi.run(None, {n: c.numpy() for n, c in zip(nama_masuk, contoh)})[0]
    with torch.no_grad():
        acuan = model(*contoh).numpy()
    print(f"{path}: selisih maks ONNX vs PyTorch = {np.abs(keluaran - acuan).max():.2e}")

for d in ["kata", "abjad", "makna"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

ekspor(model_kata, (torch.tensor(Xte[:1]),), f"{OUT}/kata/model.onnx", ["x"], {"x": {0: "batch"}})
ekspor(model_abjad, (torch.tensor(Ate[:1]),), f"{OUT}/abjad/model.onnx", ["x"], {"x": {0: "batch"}})

class MaknaEkspor(nn.Module):        # tanpa padding mask: browser memproses satu kalimat
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, src, tgt): return self.m(src, tgt)
contoh_src = torch.tensor([enc(tok_masukan("Apa kabar?", "SAYA BAIK", "netral"))])
ekspor(MaknaEkspor(model_makna), (contoh_src, torch.tensor([[BOS, 5]])), f"{OUT}/makna/model.onnx", ["src", "tgt"],
       {"src": {1: "S"}, "tgt": {1: "T"}, "logits": {1: "T"}})
model_kata.to(DEVICE); model_abjad.to(DEVICE); model_makna.to(DEVICE)
""")
code("""
json.dump({"kelas": KELAS, "gloss": GLOSS, "seq_len": SEQ_LEN, "active_y": ACTIVE_Y, "n_fitur": F_TOTAL, "blendshape": BS,
           "skor_wajah": {k: [BS[i - F_TANGAN] for i in idx] for k, idx in SKOR_WAJAH.items()},
           "ambang_wajah": AMBANG_WAJAH, "metrik": HASIL_KATA}, open(f"{OUT}/kata/config.json", "w"), indent=1)
json.dump({"kelas": HURUF, "n_fitur": 127, "metrik": HASIL_ABJAD}, open(f"{OUT}/abjad/config.json", "w"), indent=1)
json.dump({"itos": ITOS, "pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK, "maks_src": MAKS_SRC, "maks_tgt": MAKS_TGT,
           "label_wajah": ["netral", "alis_naik", "alis_turun", "senyum"],
           "token_teks": "[a-z0-9]+|[?.,!] (huruf kecil)", "metrik": HASIL_MAKNA},
          open(f"{OUT}/makna/config.json", "w"), indent=1, ensure_ascii=False)
json.dump({"kata": HASIL_KATA, "abjad": HASIL_ABJAD, "makna": HASIL_MAKNA}, open(f"{OUT}/metrik.json", "w"), indent=1)

open(f"{OUT}/README.md", "w").write(f'''---
language: id
license: mit
library_name: onnx
tags: [sign-language, bisindo, mediapipe, onnx]
datasets: [{DATASET_REPO}]
---
# Model penerjemah isyarat BISINDO (bermakna sesuai konteks)

Tiga model ONNX untuk dijalankan di browser (onnxruntime-web) setelah ekstraksi landmark MediaPipe.

| Model | Masukan | Keluaran | Hasil uji |
|---|---|---|---|
| `kata/` | 32 frame × 195 fitur (tangan, tubuh, ekspresi wajah) | 40 isyarat | akurasi {HASIL_KATA["dengan_wajah"]["akurasi"]:.2%} (tanpa wajah {HASIL_KATA["tanpa_wajah"]["akurasi"]:.2%}) |
| `abjad/` | 127 fitur tangan dari 1 foto | huruf A–Z | akurasi {HASIL_ABJAD["akurasi"]:.2%} |
| `makna/` | konteks + rangkaian isyarat + label wajah | kalimat Indonesia | exact match {HASIL_MAKNA["test"]["exact_match"]:.2%}, chrF {HASIL_MAKNA["test"]["chrf"]} (kombinasi baru: {HASIL_MAKNA["test_kombinasi_baru"]["exact_match"]:.2%}) |

Data uji kata dan latih berasal dari dataset yang sama (peraga dan latar mirip), dan korpus kalimat masih
sintetis, sehingga kinerja pada penggunaan nyata perlu diuji terpisah. Lihat dataset `{DATASET_REPO}`.
''')
print(open(f"{OUT}/README.md").read())
""")
code("""
if UNGGAH_KE_HF:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=OUT, repo_id=MODEL_REPO, repo_type="model", commit_message="Latih ulang model")
    print("Terunggah: https://huggingface.co/" + MODEL_REPO)
else:
    print("UNGGAH_KE_HF = False -> model hanya disimpan di folder", OUT)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = SEL
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "accelerator": "GPU",
                  "colab": {"provenance": []}}
nbf.write(nb, "Latih_Model_Isyarat.ipynb")
print("Latih_Model_Isyarat.ipynb:", len(SEL), "sel")
