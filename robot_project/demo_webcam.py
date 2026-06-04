"""
demo_webcam.py - Demo temps reel webcam + modele 98%

Affiche en temps reel :
  - Les landmarks MediaPipe sur la main
  - La classe predite (Vide / Faible / Optimal / Trop fort)
  - La confiance du modele
  - L'etat de la FSM (IDLE / APPROACH / GRIP / HOLD / EMERGENCY)
  - La barre de pression
  - L'historique des predictions
  - La dead zone progress
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import collections
import os

from config import (
    MODEL_PATH, CAMERA_INDEX, MAX_HANDS,
    MIN_DETECTION_CONF, MIN_TRACKING_CONF,
    GRIP_RULES, STABILITY_FRAMES, GRACE_PERIOD_FRAMES,
    CONFIDENCE_MIN, PRESSURE_CLASSES,
)

# ── Couleurs ──────────────────────────────────────────────────────────────────
COLORS = {
    0: (150, 150, 150),   # Vide      - gris
    1: (255, 140,   0),   # Faible    - orange
    2: (50,  200,  50),   # Optimal   - vert
    3: (40,   40, 220),   # Trop fort - rouge
}
STATE_COLORS = {
    "IDLE":      (200, 200, 200),
    "APPROACH":  (255, 200,   0),
    "GRIP":      (50,  200,  50),
    "HOLD":      (0,   180, 255),
    "RELEASE":   (200, 200,   0),
    "EMERGENCY": (40,   40, 220),
}
CLASS_NAMES = {0: "Vide", 1: "Faible", 2: "Optimal", 3: "Trop fort"}

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

HISTORY_LEN = 60
LM_COLS  = [f"lm{i}_{a}" for i in range(21) for a in ("x","y","z")]
VEL_COLS = [f"lm{i}_v{a}" for i in range(21) for a in ("x","y","z")]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Modele charge : {type(model).__name__}")
    return model


def get_features(landmarks, prev_pos, dt):
    """Extrait les 126 features (positions + vitesses) depuis les landmarks."""
    bx = landmarks.landmark[0].x
    by = landmarks.landmark[0].y
    bz = landmarks.landmark[0].z

    pos = np.array([
        [lm.x - bx, lm.y - by, lm.z - bz]
        for lm in landmarks.landmark
    ])  # (21, 3)

    vel = (pos - prev_pos) / dt if prev_pos is not None else np.zeros_like(pos)
    features = np.concatenate([pos.flatten(), vel.flatten()])
    return features, pos


def draw_panel_bg(frame, x, y, w, h, alpha=0.6):
    """Fond semi-transparent pour les panneaux."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 80, 80), 1)


def draw_class_panel(frame, pclass, conf, state_name, dead_zone_progress):
    """Panneau gauche : classe + confiance + etat FSM."""
    h, w = frame.shape[:2]
    draw_panel_bg(frame, 10, 10, 220, 200)

    color = COLORS.get(pclass, (200, 200, 200))
    name  = CLASS_NAMES.get(pclass, "?")

    # Classe predite
    cv2.putText(frame, "CLASSE", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, name, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    # Barre de confiance
    cv2.putText(frame, f"Confiance : {conf:.0%}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    bar_w = int(conf * 190)
    cv2.rectangle(frame, (20, 102), (210, 114), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 102), (20 + bar_w, 114), color, -1)

    # Etat FSM
    state_color = STATE_COLORS.get(state_name, (200, 200, 200))
    cv2.putText(frame, "ETAT FSM", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, state_name, (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

    # Dead zone progress
    if dead_zone_progress > 0:
        cv2.putText(frame, f"Dead zone : {dead_zone_progress}/{STABILITY_FRAMES}", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        dw = int((dead_zone_progress / STABILITY_FRAMES) * 190)
        cv2.rectangle(frame, (20, 190), (210, 198), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 190), (20 + dw, 198), (255, 200, 0), -1)


def draw_history(frame, history):
    """Historique des predictions en bas de l'ecran."""
    h, w = frame.shape[:2]
    gx, gy, gw, gh = 10, h - 80, w - 20, 60
    draw_panel_bg(frame, gx, gy, gw, gh)

    cv2.putText(frame, "Historique", (gx + 5, gy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    if len(history) > 1:
        step = gw / HISTORY_LEN
        for i in range(1, len(history)):
            cls_prev = history[i-1]
            cls_curr = history[i]
            x1 = int(gx + (i-1) * step)
            x2 = int(gx + i * step)
            y1 = gy + gh - int((cls_prev / 3) * (gh - 20)) - 5
            y2 = gy + gh - int((cls_curr / 3) * (gh - 20)) - 5
            color = COLORS.get(int(cls_curr), (200, 200, 200))
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)


def draw_pressure_bar(frame, pclass, conf):
    """Barre de pression verticale droite."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = w - 50, 10, 30, h - 100
    draw_panel_bg(frame, bx - 5, by - 5, bw + 15, bh + 10)

    # Zones colorees
    zones = [
        (0, 0.25, COLORS[0]),    # Vide
        (0.25, 0.5, COLORS[1]),  # Faible
        (0.5, 0.75, COLORS[2]),  # Optimal
        (0.75, 1.0, COLORS[3]),  # Trop fort
    ]
    for z_min, z_max, zcolor in zones:
        zy1 = by + bh - int(z_max * bh)
        zy2 = by + bh - int(z_min * bh)
        cv2.rectangle(frame, (bx, zy1), (bx + bw, zy2),
                      tuple(c // 3 for c in zcolor), -1)

    # Indicateur classe actuelle
    zone_center = (pclass + 0.5) / 4
    indicator_y = by + bh - int(zone_center * bh)
    bar_h = int(bh / 4)
    bar_y = indicator_y - bar_h // 2
    color = COLORS.get(pclass, (200, 200, 200))
    filled = int(bar_h * conf)
    cv2.rectangle(frame, (bx, bar_y), (bx + bw, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (100, 100, 100), 1)

    # Labels
    labels = ["V", "F", "O", "T"]
    for i, label in enumerate(labels):
        ly = by + bh - int((i + 0.5) / 4 * bh)
        cv2.putText(frame, label, (bx + 8, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


def draw_fps(frame, fps):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def draw_legend(frame):
    h, w = frame.shape[:2]
    draw_panel_bg(frame, 240, 10, 200, 90)
    cv2.putText(frame, "LEGENDE", (250, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    for i, (cls, name) in enumerate(CLASS_NAMES.items()):
        color = COLORS[cls]
        cy = 45 + i * 16
        cv2.rectangle(frame, (250, cy - 8), (262, cy + 2), color, -1)
        cv2.putText(frame, name, (268, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


def main():
    model = load_model()

    # Objet a saisir
    print("\nObjets disponibles :", ", ".join(GRIP_RULES.keys()))
    objet = input("Objet a saisir (ex: stylo) : ").strip().lower()
    if objet not in GRIP_RULES:
        print(f"Objet inconnu - simulation sans regles de prehension")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERR] Camera introuvable")
        return

    # Etat
    pclass       = 0
    conf         = 0.0
    state_name   = "IDLE"
    history      = collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN)
    stability_w  = collections.deque(maxlen=STABILITY_FRAMES)
    grace_count  = 0
    prev_pos     = None
    prev_t       = time.time()
    fps_counter  = collections.deque(maxlen=30)

    print("\n[OK] Demo demarree - [Q] pour quitter")
    print("      Tiens un objet devant la camera\n")

    with mp_hands.Hands(
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    ) as detector:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0  = time.time()
            now = t0
            dt  = max(now - prev_t, 1e-6)
            prev_t = now

            frame   = cv2.flip(frame, 1)
            h, w    = frame.shape[:2]
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            hand_detected = False

            if results.multi_hand_landmarks:
                grace_count = 0
                hand_detected = True

                for landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Dessiner landmarks
                    mp_drawing.draw_landmarks(
                        frame, landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # Extraire features
                    features, prev_pos = get_features(landmarks, prev_pos, dt)
                    X = features.reshape(1, -1)

                    # Prediction
                    try:
                        if hasattr(model, "predict_proba"):
                            proba  = model.predict_proba(X)[0]
                            pclass = int(np.argmax(proba))
                            conf   = float(proba[pclass])
                        else:
                            pclass = int(model.predict(X)[0])
                            conf   = 1.0
                    except Exception:
                        pass

                    # Dead zone
                    stability_w.append(pclass)
                    dead_zone_progress = len(stability_w)

                    # FSM simplifiee pour la demo
                    if conf < CONFIDENCE_MIN:
                        state_name = "UNCERTAIN"
                    elif len(stability_w) >= STABILITY_FRAMES and np.std(list(stability_w)) == 0:
                        if pclass == 2:
                            state_name = "HOLD"
                        elif pclass == 3:
                            state_name = "EMERGENCY"
                        elif pclass == 0:
                            state_name = "IDLE"
                        else:
                            state_name = "APPROACH"
                    else:
                        state_name = "APPROACH"

                    history.append(pclass)

                    # Afficher classe predite sur la main
                    lm0 = landmarks.landmark[0]
                    px  = int(lm0.x * w)
                    py  = int(lm0.y * h) - 20
                    color = COLORS.get(pclass, (200, 200, 200))
                    name  = CLASS_NAMES.get(pclass, "?")
                    cv2.putText(frame, f"{name} {conf:.0%}", (px - 40, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            else:
                # Grace period
                grace_count += 1
                prev_pos = None
                if grace_count > GRACE_PERIOD_FRAMES:
                    state_name = "IDLE"
                    pclass     = 0
                    conf       = 0.0
                    stability_w.clear()

            # UI
            draw_class_panel(frame, pclass, conf, state_name,
                             len(stability_w) if hand_detected else 0)
            draw_pressure_bar(frame, pclass, conf)
            draw_history(frame, history)
            draw_legend(frame)

            # FPS
            fps_counter.append(time.time() - t0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter) + 1e-6)
            draw_fps(frame, fps)

            # Message si pas de main
            if not hand_detected and grace_count <= GRACE_PERIOD_FRAMES:
                cv2.putText(frame, f"Grace period {grace_count}/{GRACE_PERIOD_FRAMES}",
                            (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            elif not hand_detected:
                cv2.putText(frame, "Montre ta main a la camera",
                            (w//2 - 140, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

            cv2.imshow("Demo Robot Autonome - 98% accuracy", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Demo terminee")


if __name__ == "__main__":
    main()
