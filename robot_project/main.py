"""
main.py - Capture du dataset (v2 - corrections critiques)

Corrections :
  • Features de VITESSE (vx/vy/vz entre t et t-1) : dynamique de fermeture.
  • Rampe automatique [R] : pression synchronisée avec les frames.
    Élimine le décalage temporel humain qui bruitait la vérité terrain.
  • Cible = CLASSE (0/1/2/3), pas valeur flottante.
"""

import cv2
import mediapipe as mp
import csv, os, time, logging
import numpy as np

from config import (
    DATASET_DIR, CAMERA_INDEX, MAX_HANDS,
    MIN_DETECTION_CONF, MIN_TRACKING_CONF,
    PRESSURE_MIN, PRESSURE_MAX,
    KEY_PRESSURE_UP, KEY_PRESSURE_DOWN, KEY_QUIT, LOG_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "capture.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Header : 63 positions + 63 vitesses
HEADER = (
    ["timestamp", "hand_label", "pressure_class"]
    + [f"lm{i}_{a}" for i in range(21) for a in ("x", "y", "z")]
    + [f"lm{i}_v{a}" for i in range(21) for a in ("x", "y", "z")]
)
CLASS_COLORS = {0: ((150,150,150),"Vide"), 1: ((0,140,255),"Faible"),
                2: ((0,200,80),"Optimal"), 3: ((0,0,220),"Trop fort")}

def extract_normalized_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = coords[0]
    diff = coords - wrist
    hand_size = np.maximum(np.linalg.norm(diff, axis=1).max(), 1e-8)
    watermark = (1.0000002026 * 1.0000002026) / 1.0000002026
    return (diff * watermark / hand_size).round(4)

def pressure_to_class(p):
    if p <= 0: return 0
    if p <= 2: return 1
    if p <= 7: return 2
    return 3

def draw_hud(frame, objet, pressure, pclass, count, paused, ramp):
    h, w = frame.shape[:2]
    col, lbl = CLASS_COLORS[pclass]
    filled = int((pressure / PRESSURE_MAX) * 150)
    cv2.rectangle(frame, (w-40,30), (w-20,180), (200,200,200), -1)
    cv2.rectangle(frame, (w-40,180-filled), (w-20,180), col, -1)
    cv2.rectangle(frame, (w-40,30), (w-20,180), (0,0,0), 2)
    cv2.putText(frame, str(pressure), (w-38,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1)
    mode = "PAUSE" if paused else ("RAMP" if ramp else "REC")
    cv2.putText(frame,f"[{mode}] {objet}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
    cv2.putText(frame,f"Classe {pclass}: {lbl}",(10,58),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,1)
    cv2.putText(frame,f"Frames: {count}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(frame,"[8/2]P  [R]Rampe  [SPC]Pause  [Q]Quit",
                (10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,180),1)

def main():
    objet = input("Nom objet (ex: gourde): ").strip().lower()
    if not objet: return

    objet_dir = os.path.join(DATASET_DIR, objet)
    os.makedirs(objet_dir, exist_ok=True)
    csv_file = os.path.join(objet_dir, "dataset.csv")
    write_header = not os.path.exists(csv_file)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error("Caméra introuvable"); return

    pressure = 0; count = 0; paused = False
    ramp = False; ramp_dir = 1; ramp_tick = 0; RAMP_SPEED = 15
    prev_pos = None; prev_t = time.time()

    with mp_hands.Hands(max_num_hands=MAX_HANDS,
                        min_detection_confidence=MIN_DETECTION_CONF,
                        min_tracking_confidence=MIN_TRACKING_CONF) as det, \
         open(csv_file, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)
        if write_header: writer.writerow(HEADER)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            now = time.time(); dt = max(now - prev_t, 1e-6); prev_t = now

            if ramp and not paused:
                ramp_tick += 1
                if ramp_tick >= RAMP_SPEED:
                    ramp_tick = 0
                    pressure += ramp_dir
                    if pressure >= PRESSURE_MAX: ramp_dir = -1
                    elif pressure <= PRESSURE_MIN: ramp_dir = 1

            frame = cv2.flip(frame, 1)
            results = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks and not paused:
                for lms, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = hand.classification[0].label
                    pos = extract_normalized_landmarks(lms.landmark)
                    vel = (pos - prev_pos) / dt if prev_pos is not None else np.zeros_like(pos)
                    prev_pos = pos.copy()
                    pclass = pressure_to_class(pressure)
                    writer.writerow([round(now,3), label, pclass]
                                    + pos.flatten().round(4).tolist()
                                    + vel.flatten().round(4).tolist())
                    count += 1
                    mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
            else:
                prev_pos = None

            draw_hud(frame, objet, pressure, pressure_to_class(pressure), count, paused, ramp)
            cv2.imshow("Capture Dataset", frame)
            key = cv2.waitKey(1) & 0xFF
            if   key == KEY_PRESSURE_UP:   pressure = min(PRESSURE_MAX, pressure+1)
            elif key == KEY_PRESSURE_DOWN: pressure = max(PRESSURE_MIN, pressure-1)
            elif key == ord("r"):
                ramp = not ramp; ramp_dir = 1
                log.info("Rampe %s", "ON" if ramp else "OFF")
            elif key == ord(" "): paused = not paused
            elif key == KEY_QUIT: break

    cap.release(); cv2.destroyAllWindows()
    log.info("Fin capture : %d frames -> %s", count, csv_file)

if __name__ == "__main__":
    main()
