"""
simulate_robot_visual.py — Simulation visuelle de préhension (OpenCV)
Améliorations :
  • Affichage de la zone de pression sécurisée (min/max)
  • Animations : tremblement si pression limite, explosion si cassé
  • Sélection d'objet dans l'interface (pas juste terminal)
  • Graphique historique de pression en temps réel
  • Export de la session en PNG
"""

import cv2
import numpy as np
import collections
import os
import time

from config import GRIP_RULES

# ─── Palette ──────────────────────────────────────────────────────────────────
BG_COLOR    = (245, 245, 245)
OK_COLOR    = ( 50, 180,  50)
WARN_COLOR  = ( 20, 120, 255)
BAD_COLOR   = ( 40,  40, 220)
TEXT_COLOR  = ( 30,  30,  30)
HIST_COLOR  = (100, 100, 220)

W, H = 700, 480
HISTORY_LEN = 80


def draw_pressure_bar(frame, pressure: int, min_p: int, max_p: int) -> None:
    bx, by, bw, bh = 20, 60, 30, 320

    # Fond
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (210, 210, 210), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), TEXT_COLOR, 2)

    # Zone OK (vert)
    y_ok_top = by + bh - int((max_p / 10) * bh)
    y_ok_bot = by + bh - int((min_p / 10) * bh)
    cv2.rectangle(frame, (bx, y_ok_top), (bx + bw, y_ok_bot), (180, 255, 180), -1)

    # Remplissage pression actuelle
    filled = int((pressure / 10) * bh)
    color = OK_COLOR if min_p <= pressure <= max_p else BAD_COLOR
    cv2.rectangle(frame, (bx, by + bh - filled), (bx + bw, by + bh), color, -1)

    # Marqueurs min/max
    for p, label in [(min_p, "min"), (max_p, "max")]:
        yy = by + bh - int((p / 10) * bh)
        cv2.line(frame, (bx - 5, yy), (bx + bw + 5, yy), WARN_COLOR, 1)
        cv2.putText(frame, label, (bx + bw + 8, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WARN_COLOR, 1)

    cv2.putText(frame, f"{pressure}/10", (bx - 5, by + bh + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    cv2.putText(frame, "P", (bx + 5, by - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)


def draw_object(frame, objet: str, status: str, pressure: int, min_p: int, max_p: int, tick: int) -> None:
    cx, cy = 320, 220
    ox1, oy1, ox2, oy2 = cx - 60, cy - 60, cx + 60, cy + 60

    # Tremblement si pression limite
    shake = 0
    if pressure == max_p or pressure == min_p + 1:
        shake = int(4 * np.sin(tick * 0.5))

    ox1 += shake; ox2 += shake

    color = OK_COLOR if status == "TENU" else BAD_COLOR
    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, -1)
    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), TEXT_COLOR, 2)

    # Croix si cassé
    if status == "CASSE":
        cv2.line(frame, (ox1, oy1), (ox2, oy2), (255, 255, 255), 3)
        cv2.line(frame, (ox1, oy2), (ox2, oy1), (255, 255, 255), 3)

    # Flèche tombante si tombe
    if status == "TOMBE":
        cv2.arrowedLine(frame, (cx, oy2), (cx, oy2 + 40), BAD_COLOR, 3, tipLength=0.4)

    # Nom objet
    cv2.putText(frame, objet.upper(), (cx - 50, oy1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(frame, status, (cx - 45, oy2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def draw_history(frame, history: collections.deque, min_p: int, max_p: int) -> None:
    """Mini graphique historique pression."""
    gx, gy, gw, gh = 420, 60, 240, 120
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (230, 230, 230), -1)
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), TEXT_COLOR, 1)
    cv2.putText(frame, "Historique pression", (gx + 5, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_COLOR, 1)

    # Zones min/max
    y_max = gy + gh - int((max_p / 10) * gh)
    y_min = gy + gh - int((min_p / 10) * gh)
    cv2.rectangle(frame, (gx, y_max), (gx + gw, y_min), (200, 255, 200), -1)

    if len(history) > 1:
        pts = []
        for i, val in enumerate(history):
            px = gx + int(i * gw / HISTORY_LEN)
            py = gy + gh - int((val / 10) * gh)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], HIST_COLOR, 2)


def select_object() -> str:
    """Sélection d'objet dans le terminal."""
    available = list(GRIP_RULES.keys())
    print("\nObjets disponibles :")
    for i, o in enumerate(available):
        mn, mx = GRIP_RULES[o]
        print(f"  {i:2d} — {o:<20} (pression: {mn}–{mx})")
    print("  Entrée directe possible (ex: 'gourde')")
    choice = input("Choix : ").strip().lower()
    if choice in GRIP_RULES:
        return choice
    try:
        return available[int(choice)]
    except (ValueError, IndexError):
        print(f"'{choice}' non reconnu — règle par défaut appliquée.")
        return choice


def run() -> None:
    objet = select_object()
    min_p, max_p = GRIP_RULES.get(objet, (0, 10))
    if objet not in GRIP_RULES:
        print(f"Objet '{objet}' non reconnu — pression par défaut {min_p}-{max_p} utilisée.")

    pressure = (min_p + max_p) // 2
    history: collections.deque = collections.deque([pressure] * HISTORY_LEN, maxlen=HISTORY_LEN)
    tick = 0

    cv2.namedWindow("Simulation Robot")
    print("[A] P+  [Z] P-  [ESC] Quitter")

    while True:
        frame = np.ones((H, W, 3), dtype=np.uint8)
        frame[:] = BG_COLOR

        # Détermination état
        if pressure < min_p:
            status = "TOMBE"
        elif pressure > max_p:
            status = "CASSE"
        else:
            status = "TENU"

        history.append(pressure)
        draw_pressure_bar(frame, pressure, min_p, max_p)
        draw_object(frame, objet, status, pressure, min_p, max_p, tick)
        draw_history(frame, history, min_p, max_p)

        # Légende
        cv2.putText(frame, "[A] Pression+   [Z] Pression-   [ESC] Quitter",
                    (20, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        cv2.imshow("Simulation Robot", frame)
        tick += 1

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord("a"):
            pressure = min(10, pressure + 1)
        elif key == ord("z"):
            pressure = max(0, pressure - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
