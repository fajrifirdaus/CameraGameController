"""
Body Motion Controller untuk Subway Surfers (atau game runner sejenis)
========================================================================
Pakai webcam + MediaPipe Pose (Tasks API) untuk mendeteksi gerakan badan
dan mengubahnya jadi tombol keyboard:

    - Geser badan ke KIRI / KANAN     -> tombol arrow left / right
                                         (otomatis balik ke tengah juga ditangani)
    - Tangan KANAN diangkat ke atas
      (melebihi tinggi bahu)          -> tombol arrow up   (lompat)
    - Tangan KIRI diangkat ke atas
      (melebihi tinggi bahu)          -> tombol arrow down (turun/slide)

Catatan akurasi deteksi tangan:
    Sebelumnya margin "di atas bahu" pakai nilai pixel statis (20px), tanpa
    smoothing, dan tanpa cek confidence/visibility landmark. Ini bikin deteksi
    kurang konsisten: terlalu ketat/lembek tergantung jarak ke kamera, dan
    gampang "geter" (flicker) karena noise antar-frame. Sekarang:
      1. Margin dihitung RELATIF ke lebar bahu (scale-invariant) -> tetap
         konsisten baik user berdiri dekat maupun jauh dari kamera.
      2. Posisi bahu & pergelangan tangan di-smoothing (exponential moving
         average) sebelum dibandingkan -> mengurangi noise antar-frame.
      3. Landmark dengan visibility/confidence rendah (misal tangan keluar
         frame atau tertutup) diabaikan dulu di frame itu, daripada dipakai
         padahal posisinya tidak akurat.

Catatan versi MediaPipe:
    MediaPipe baru-baru ini menghapus dukungan untuk API lama
    `mp.solutions.pose` (lihat: github.com/google-ai-edge/mediapipe/issues/6192).
    Script ini sudah memakai API baru yang resmi: "MediaPipe Tasks"
    (mp.tasks.vision.PoseLandmarker), jadi tetap kompatibel dengan
    versi mediapipe terbaru.

Cara pakai:
    1. Jalankan script ini (lihat README.md untuk setup lengkap).
       Saat pertama kali jalan, script akan otomatis download file
       model pose_landmarker_lite.task (~5-10MB, butuh internet sekali saja).
    2. Berdiri tegak di posisi netral, pastikan bahu & kedua tangan kelihatan
       penuh di kamera (beri ruang ke atas kepala buat angkat tangan), lalu
       tekan 'c' untuk kalibrasi.
    3. Klik window game (browser/emulator) supaya jadi window aktif.
    4. Mulai bermain. Tekan 'q' di window kamera untuk keluar.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ====================== KONFIGURASI (silakan disesuaikan) ======================
CAM_INDEX = 0                    # ganti ke 1 / 2 kalau webcam yang terdeteksi salah
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

HORIZONTAL_THRESHOLD = 60        # px, jarak minimal dari baseline buat dianggap geser kiri/kanan

# --- Deteksi tangan ke atas (scale-invariant, relatif ke lebar bahu) ---
HANDS_UP_MARGIN_RATIO = 0.2      # wrist harus di atas garis bahu sejauh (rasio ini x lebar bahu)
HANDS_UP_MIN_MARGIN_PX = 10      # batas bawah margin dalam px (jaga-jaga kalau lebar bahu kekecilan)
MIN_LANDMARK_VISIBILITY = 0.3    # landmark di bawah confidence ini diabaikan (dianggap "tidak yakin")
                                  # NOTE: pose_landmarker Tasks API sering TIDAK mengisi field
                                  # visibility sama sekali (dikenal sebagai bug upstream MediaPipe,
                                  # lihat github.com/google-ai-edge/mediapipe/issues/4479 dkk).
                                  # Kalau visibility memang tidak tersedia, filter ini otomatis
                                  # dilewati (lihat get_visibility) - jadi aman dibiarkan menyala.
SMOOTHING_ALPHA_HAND = 0.7       # smoothing KHUSUS gap tangan->bahu: dibuat lebih responsif
                                  # daripada smoothing posisi badan, karena gerakan tangan saat
                                  # lompat/turun cenderung cepat & singkat
HAND_UP_FRAMES_NEEDED = 2        # frame berturut-turut needed khusus utk hand-up (lebih sedikit
                                  # dari CONSECUTIVE_FRAMES_NEEDED karena gerakan tangan cepat)

ACTION_COOLDOWN = 0.5            # detik, cooldown tiap aksi lompat/turun (biar gak ke-spam)
MOVE_COOLDOWN = 0.3              # detik, cooldown pindah posisi kiri/tengah/kanan
CONSECUTIVE_FRAMES_NEEDED = 3    # anti getar/noise: dipakai khusus utk geser kiri/kanan posisi badan

INDICATOR_HOLD_SECONDS = 0.4     # detik, berapa lama status "YA" di overlay ditahan tampil setelah
                                  # trigger - HANYA untuk tampilan, tidak mempengaruhi kapan tombol
                                  # sungguhan ditekan ke game (itu tetap instan di frame trigger)

# Model pose landmarker. "lite" = paling ringan/cepat (disarankan buat real-time game).
# Alternatif: ganti "lite" jadi "full" atau "heavy" di MODEL_NAME kalau mau lebih akurat
# (tapi lebih berat/lambat).
MODEL_NAME = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True         # geser mouse ke pojok kiri-atas screen = emergency stop

# Index landmark BlazePose (33 titik) yang relevan buat kita.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

# ===================== PENTING soal kiri/kanan TANGAN =====================
# Frame di bawah ini di-flip (mirror) SEBELUM diproses MediaPipe (lihat baris
# `cv2.flip` di main loop), supaya tampilan kamera terasa natural seperti
# cermin. Akibatnya, label LEFT/RIGHT bawaan MediaPipe jadi TERTUKAR dari
# sudut pandang user yang lihat layar:
#   - landmark index 15 (nama asli MediaPipe: "LEFT_WRIST")  -> tangan KANAN asli user
#   - landmark index 16 (nama asli MediaPipe: "RIGHT_WRIST") -> tangan KIRI asli user
# Kalau pas ditest ternyata kebalik di komputer kamu, TINGGAL TUKAR nilai
# dua baris di bawah ini.
USER_RIGHT_WRIST = 15
USER_LEFT_WRIST = 16
# ============================================================================

# Urutan "lane" buat hitung berapa kali tombol kiri/kanan perlu ditekan
# supaya badan virtual di game balik ke tengah dengan benar.
LANE_INDEX = {"LEFT": -1, "CENTER": 0, "RIGHT": 1}

# Daftar pasangan landmark buat gambar garis skeleton (simplifikasi dari topologi BlazePose)
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),       # torso
    (11, 13), (13, 15),                            # lengan kiri
    (12, 14), (14, 16),                            # lengan kanan
    (23, 25), (25, 27),                            # kaki kiri
    (24, 26), (26, 28),                            # kaki kanan
]
# =================================================================================


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Model belum ditemukan, mendownload {MODEL_NAME} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download model selesai.")
    except Exception as e:
        raise RuntimeError(
            f"Gagal download model otomatis ({e}). "
            f"Silakan download manual dari:\n{MODEL_URL}\n"
            f"Lalu simpan dengan nama '{MODEL_NAME}' di folder yang sama dengan main.py."
        )


def get_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def get_visibility(landmarks, idx):
    """Ambil confidence/visibility landmark.

    Catatan: pose_landmarker Tasks API sering tidak mengisi field ini sama
    sekali (selalu None) - ini bug upstream yang sudah dilaporkan di banyak
    platform. Kalau None, anggap landmark "terlihat" (kembalikan 1.0) supaya
    filter ini tidak memblokir deteksi saat model tidak menyediakan info ini.
    """
    vis = getattr(landmarks[idx], "visibility", None)
    return 1.0 if vis is None else vis


def smooth(prev, new, alpha):
    """Exponential moving average sederhana buat kurangi noise antar-frame."""
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev


def main():
    ensure_model_downloaded()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            f"Tidak bisa membuka webcam index {CAM_INDEX}. "
            f"Coba ganti CAM_INDEX ke 1 atau 2 di bagian konfigurasi."
        )

    baseline = {"center_x": None}
    last_position = "CENTER"
    last_move_time = 0.0
    last_jump_time = 0.0
    last_down_time = 0.0

    # khusus buat HOLD TAMPILAN status "YA" di overlay (tidak mempengaruhi logika tombol)
    jump_indicator_until = 0.0
    down_indicator_until = 0.0

    was_right_hand_up = False
    was_left_hand_up = False
    right_up_counter = 0
    left_up_counter = 0
    pos_counter = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}

    # nilai smoothing (exponential moving average), diisi None di awal
    sm_shoulder_w = None          # lebar bahu (untuk margin scale-invariant; tidak perlu super responsif)
    sm_right_gap = None           # gap (shoulder_y - right_wrist_y): makin besar = tangan makin tinggi
    sm_left_gap = None            # gap (shoulder_y - left_wrist_y)

    start_time = time.time()

    print("=" * 55)
    print(" BODY MOTION CONTROLLER - SUBWAY SURFERS (POC)")
    print("=" * 55)
    print(" [c] kalibrasi  (berdiri netral dulu sebelum menekan)")
    print(" [q] keluar")
    print("=" * 55)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Gagal membaca frame dari webcam.")
                    break

                frame = cv2.flip(frame, 1)  # mirror biar gerakan terasa natural
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                frame_timestamp_ms = int((time.time() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                current_metrics = None

                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]  # 33 landmark pose pertama yang terdeteksi

                    # gambar skeleton sederhana
                    for idx_a, idx_b in POSE_CONNECTIONS:
                        xa, ya = get_xy(lm, idx_a, w, h)
                        xb, yb = get_xy(lm, idx_b, w, h)
                        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 0), 2)
                    for point in lm:
                        cv2.circle(frame, (int(point.x * w), int(point.y * h)), 3, (0, 255, 255), -1)

                    l_sh = get_xy(lm, LEFT_SHOULDER, w, h)
                    r_sh = get_xy(lm, RIGHT_SHOULDER, w, h)
                    right_wrist_x, right_wrist_y = get_xy(lm, USER_RIGHT_WRIST, w, h)
                    left_wrist_x, left_wrist_y = get_xy(lm, USER_LEFT_WRIST, w, h)

                    right_wrist_vis = get_visibility(lm, USER_RIGHT_WRIST)
                    left_wrist_vis = get_visibility(lm, USER_LEFT_WRIST)

                    center_x = (l_sh[0] + r_sh[0]) / 2
                    raw_shoulder_y = (l_sh[1] + r_sh[1]) / 2
                    raw_shoulder_w = abs(r_sh[0] - l_sh[0])
                    current_metrics = {"center_x": center_x}

                    # --- gap mentah (raw) dihitung DULU sebelum smoothing ---
                    # (gap = shoulder_y - wrist_y; makin besar/positif = tangan makin tinggi
                    # di atas bahu, karena koordinat y mengecil ke arah atas gambar)
                    raw_right_gap = raw_shoulder_y - right_wrist_y
                    raw_left_gap = raw_shoulder_y - left_wrist_y

                    # --- smoothing ---
                    # lebar bahu jarang berubah cepat -> smoothing biasa (lebih halus) cukup
                    sm_shoulder_w = smooth(sm_shoulder_w, raw_shoulder_w, alpha=0.3)
                    # gap tangan->bahu perlu RESPONSIF karena gerakan lompat/turun itu cepat,
                    # jadi smoothing-nya pakai alpha tinggi (sedikit lag) sambil tetap meredam noise
                    sm_right_gap = smooth(sm_right_gap, raw_right_gap, alpha=SMOOTHING_ALPHA_HAND)
                    sm_left_gap = smooth(sm_left_gap, raw_left_gap, alpha=SMOOTHING_ALPHA_HAND)

                    # margin scale-invariant: relatif ke lebar bahu, bukan pixel statis
                    margin = max(HANDS_UP_MIN_MARGIN_PX, HANDS_UP_MARGIN_RATIO * sm_shoulder_w)

                    detected_right_hand_up = (
                        right_wrist_vis >= MIN_LANDMARK_VISIBILITY and sm_right_gap > margin
                    )
                    detected_left_hand_up = (
                        left_wrist_vis >= MIN_LANDMARK_VISIBILITY and sm_left_gap > margin
                    )

                    # --- visual debug: garis bahu (abu-abu) + garis ambang batas trigger (putih putus-putus) ---
                    threshold_y_display = raw_shoulder_y - margin
                    cv2.line(frame, (0, int(raw_shoulder_y)), (w, int(raw_shoulder_y)), (160, 160, 160), 1)
                    for dash_x in range(0, w, 20):
                        cv2.line(frame, (dash_x, int(threshold_y_display)), (dash_x + 10, int(threshold_y_display)), (255, 255, 255), 1)
                    cv2.circle(frame, (int(center_x), int(raw_shoulder_y)), 6, (0, 0, 255), -1)
                    cv2.circle(frame, (int(right_wrist_x), int(right_wrist_y)), 10,
                               (0, 255, 0) if detected_right_hand_up else (90, 90, 90), -1)
                    cv2.circle(frame, (int(left_wrist_x), int(left_wrist_y)), 10,
                               (0, 165, 255) if detected_left_hand_up else (90, 90, 90), -1)

                    # --- baris debug: angka mentah gap vs margin, biar kelihatan kenapa gagal/berhasil ---
                    debug_line = (
                        f"[DEBUG] gap_kanan={sm_right_gap:5.1f} gap_kiri={sm_left_gap:5.1f} "
                        f"margin={margin:5.1f} vis_kanan={right_wrist_vis:.2f} vis_kiri={left_wrist_vis:.2f}"
                    )
                    cv2.putText(frame, debug_line, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 255), 1)


                    if baseline["center_x"] is not None:
                        # --- posisi kiri / tengah / kanan (dengan balik ke tengah yang benar) ---
                        dx = center_x - baseline["center_x"]
                        if dx < -HORIZONTAL_THRESHOLD:
                            detected_pos = "LEFT"
                        elif dx > HORIZONTAL_THRESHOLD:
                            detected_pos = "RIGHT"
                        else:
                            detected_pos = "CENTER"

                        for k in pos_counter:
                            pos_counter[k] = pos_counter[k] + 1 if k == detected_pos else 0

                        now = time.time()
                        if (
                            pos_counter[detected_pos] >= CONSECUTIVE_FRAMES_NEEDED
                            and detected_pos != last_position
                            and now - last_move_time > MOVE_COOLDOWN
                        ):
                            # selisih "lane" dipakai supaya transisi LEFT<->CENTER<->RIGHT
                            # (termasuk balik ke tengah) selalu menekan tombol yang benar
                            diff = LANE_INDEX[detected_pos] - LANE_INDEX[last_position]
                            if diff > 0:
                                for _ in range(diff):
                                    pyautogui.press("right")
                            elif diff < 0:
                                for _ in range(-diff):
                                    pyautogui.press("left")
                            last_position = detected_pos
                            last_move_time = now

                    # --- LOMPAT & TURUN: TIDAK butuh kalibrasi 'c' sama sekali ---
                    # (sengaja diletakkan di LUAR blok "if baseline" di atas, karena gerakan tangan
                    # dibandingkan terhadap bahu sendiri di frame yang sama, bukan terhadap baseline
                    # statis. Sebelumnya logika ini ikut terkunci di dalam blok kalibrasi, sehingga
                    # jump/turun tidak pernah ter-trigger sebelum tombol 'c' ditekan.)

                    # --- LOMPAT: tangan kanan diangkat ke atas (sekali tekan per angkatan) ---
                    if detected_right_hand_up:
                        right_up_counter += 1
                    else:
                        right_up_counter = 0
                        was_right_hand_up = False  # tangan turun -> siap trigger lagi

                    now = time.time()
                    if (
                        right_up_counter >= HAND_UP_FRAMES_NEEDED
                        and not was_right_hand_up
                        and now - last_jump_time > ACTION_COOLDOWN
                    ):
                        pyautogui.press("up")
                        last_jump_time = now
                        was_right_hand_up = True
                        jump_indicator_until = now + INDICATOR_HOLD_SECONDS

                    # --- TURUN: tangan kiri diangkat ke atas (sekali tekan per angkatan) ---
                    if detected_left_hand_up:
                        left_up_counter += 1
                    else:
                        left_up_counter = 0
                        was_left_hand_up = False  # tangan turun -> siap trigger lagi

                    if (
                        left_up_counter >= HAND_UP_FRAMES_NEEDED
                        and not was_left_hand_up
                        and now - last_down_time > ACTION_COOLDOWN
                    ):
                        pyautogui.press("down")
                        last_down_time = now
                        was_left_hand_up = True
                        down_indicator_until = now + INDICATOR_HOLD_SECONDS

                # --- overlay info di window kamera ---
                # status "YA" ditahan tampil sebentar (INDICATOR_HOLD_SECONDS) biar kebaca mata,
                # meskipun tombol ke game sendiri sudah ditekan secara instan di frame trigger
                now_display = time.time()
                jump_display = now_display < jump_indicator_until
                down_display = now_display < down_indicator_until

                status = (
                    f"Posisi: {last_position} | "
                    f"Lompat(kanan): {'YA' if jump_display else '-'} | "
                    f"Turun(kiri): {'YA' if down_display else '-'}"
                )
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                if baseline["center_x"] is None:
                    cv2.putText(frame, "BELUM DIKALIBRASI - tekan 'c'", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Kalibrasi OK", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Body Motion Control - Subway Surfers", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and current_metrics is not None:
                    baseline.update(current_metrics)
                    print("Kalibrasi berhasil! Baseline:", baseline)

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()