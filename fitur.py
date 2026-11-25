"""Ekstraksi fitur per frame untuk model isyarat BISINDO (tangan + tubuh + ekspresi wajah).

Satu frame -> vektor 195 angka:

    [0:14)    pose tubuh atas, 7 titik (hidung, bahu, siku, pergelangan kiri/kanan) x (x, y)
    [14:140)  2 tangan x 21 titik x (x, y, z); slot 0 = tangan di sisi pergelangan kiri tubuh, slot 1 = kanan
    [140:142) penanda ada/tidaknya tangan slot 0 dan 1
    [142:194) 52 skor ekspresi wajah (MediaPipe face blendshapes, nilai 0-1)
    [194]     penanda wajah terdeteksi

Koordinat tubuh dan tangan digeser ke titik tengah kedua bahu dan dibagi lebar bahu, sehingga tidak
bergantung pada posisi atau jarak orang dari kamera. Posisi tangan relatif terhadap tubuh (mis. dekat
mulut untuk "Makan") tetap terbaca. Rumus yang sama dipakai di browser (lihat spesifikasi_fitur()).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODELS = Path(__file__).resolve().parent / "models"

POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
POSE_SWAP = [0, 2, 1, 4, 3, 6, 5]       # pasangan kiri<->kanan untuk augmentasi cermin
N_POSE = len(POSE_IDX)
P = N_POSE * 2                           # 14
H = 2 * 21 * 3                           # 126
F_TANGAN = P + H + 2                     # 142: fitur tubuh + tangan (sama dengan model lama)
N_WAJAH = 52
F_TOTAL = F_TANGAN + N_WAJAH + 1         # 195

BLENDSHAPE = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft",
    "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
# pasangan kiri<->kanan blendshape untuk augmentasi cermin
BLENDSHAPE_SWAP = [BLENDSHAPE.index(n.replace("Left", "#").replace("Right", "Left").replace("#", "Right"))
                   for n in BLENDSHAPE]


class Pelacak:
    """Hand + Pose + Face landmarker MediaPipe dalam mode VIDEO (melacak antar frame)."""

    def __init__(self, min_conf: float = 0.5, mode=vision.RunningMode.VIDEO):
        base = lambda f: mpp.BaseOptions(model_asset_path=str(MODELS / f))
        self.mode = mode
        self.tangan = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hand_landmarker.task"), running_mode=mode, num_hands=2,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base("pose_landmarker_lite.task"), running_mode=mode, num_poses=1))
        self.wajah = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base("face_landmarker.task"), running_mode=mode, num_faces=1,
            output_face_blendshapes=True))
        self.ts = -1

    def proses(self, frame_bgr: np.ndarray, ts_ms: float | None = None):
        """-> (pose (33,3) | None, [tangan (21,3)], blendshape (52,) | None); koordinat dalam piksel."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == vision.RunningMode.VIDEO:
            self.ts = max(int(ts_ms or 0), self.ts + 1)  # timestamp harus selalu naik
            rt, rp, rw = (self.tangan.detect_for_video(img, self.ts), self.pose.detect_for_video(img, self.ts),
                          self.wajah.detect_for_video(img, self.ts))
        else:
            rt, rp, rw = self.tangan.detect(img), self.pose.detect(img), self.wajah.detect(img)
        px = lambda lms: np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], np.float32)
        tangan = [px(t) for t in rt.hand_landmarks]
        pose = px(rp.pose_landmarks[0]) if rp.pose_landmarks else None
        bs = None
        if rw.face_blendshapes:
            bs = np.zeros(N_WAJAH, np.float32)
            for c in rw.face_blendshapes[0]:
                bs[c.index] = c.score
        return pose, tangan, bs

    def tutup(self):
        self.tangan.close()
        self.pose.close()
        self.wajah.close()


def slot_tangan(tangan, pose):
    """Slot 0 = tangan dekat pergelangan kiri tubuh, slot 1 = kanan (dicocokkan ke pergelangan pose)."""
    slots = [None, None]
    if not tangan:
        return slots
    if pose is not None:
        pergelangan = pose[[15, 16], :2]
        if len(tangan) == 1:
            slots[int(np.argmin(np.linalg.norm(pergelangan - tangan[0][0, :2], axis=1)))] = tangan[0]
        else:
            a, b = tangan[:2]
            lurus = np.linalg.norm(pergelangan[0] - a[0, :2]) + np.linalg.norm(pergelangan[1] - b[0, :2])
            silang = np.linalg.norm(pergelangan[0] - b[0, :2]) + np.linalg.norm(pergelangan[1] - a[0, :2])
            slots = [a, b] if lurus <= silang else [b, a]
    else:
        for i, pts in enumerate(sorted(tangan[:2], key=lambda p: -p[:, 0].mean())):
            slots[i] = pts
    return slots


class Featurizer:
    """(pose, tangan, blendshape) satu frame -> vektor F_TOTAL. Menyimpan pusat/skala bahu terakhir."""

    def __init__(self):
        self.pusat = None
        self.skala = None

    def __call__(self, pose, tangan, bs, ukuran_frame):
        if pose is not None:
            kiri, kanan = pose[11, :2], pose[12, :2]
            s = float(np.linalg.norm(kiri - kanan))
            if s > 1:
                self.pusat, self.skala = (kiri + kanan) / 2, s
        if self.pusat is None:  # bahu belum pernah terlihat: pakai ukuran gambar
            h, w = ukuran_frame[:2]
            self.pusat, self.skala = np.array([w / 2, h / 2], np.float32), w / 4
        f = np.zeros(F_TOTAL, np.float32)
        if pose is not None:
            f[:P] = ((pose[POSE_IDX, :2] - self.pusat) / self.skala).ravel()
        for i, pts in enumerate(slot_tangan(tangan, pose)):
            if pts is None:
                continue
            p = pts.copy()
            p[:, :2] -= self.pusat
            f[P + i * 63:P + (i + 1) * 63] = (p / self.skala).ravel()
            f[P + H + i] = 1.0
        if bs is not None:
            f[F_TANGAN:F_TANGAN + N_WAJAH] = bs
            f[F_TOTAL - 1] = 1.0
        return f


def ekstrak_video(path: str | Path, pelacak: Pelacak | None = None, lewati: int = 1):
    """Video -> (array (n_frame, F_TOTAL), fps). `lewati`=2 berarti ambil setiap frame ke-2."""
    milik_sendiri = pelacak is None
    pelacak = pelacak or Pelacak()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fz, out, i = Featurizer(), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % lewati == 0:
            pose, tangan, bs = pelacak.proses(frame, i * 1000.0 / fps)
            out.append(fz(pose, tangan, bs, frame.shape))
        i += 1
    cap.release()
    if milik_sendiri:
        pelacak.tutup()
    return np.stack(out) if out else np.zeros((0, F_TOTAL), np.float32), fps / lewati


def spesifikasi_fitur() -> dict:
    return {
        "panjang": F_TOTAL,
        "bagian": {
            "pose_xy": [0, P], "tangan_xyz": [P, P + H], "flag_tangan": [P + H, F_TANGAN],
            "wajah_blendshape": [F_TANGAN, F_TANGAN + N_WAJAH], "flag_wajah": [F_TOTAL - 1, F_TOTAL],
        },
        "pose_index_mediapipe": POSE_IDX,
        "blendshape": BLENDSHAPE,
        "normalisasi": "titik (x,y,z piksel; z = lm.z * lebar gambar) dikurangi titik tengah bahu (pose 11 & 12) "
                       "pada x,y lalu semua dibagi lebar bahu; bila bahu tidak terlihat, pakai nilai terakhir.",
    }
