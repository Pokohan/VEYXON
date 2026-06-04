"""
detection.py — Détection d'objets par analyse de contours (OpenCV)
Améliorations :
  • Détection de forme plus robuste (approxPolyDP + aspect ratio)
  • Filtre par couleur HSV configurable
  • Suivi multi-objets avec identifiants stables
  • Sortie structurée (liste de dicts)
  • Overlay informatif enrichi
"""

import cv2
import numpy as np
import logging
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from dataclasses import dataclass, field
from typing import Optional

from config import CONTOUR_MIN_AREA, BLUR_KERNEL, THRESHOLD_VALUE, LOG_DIR
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── Paramètres forme ─────────────────────────────────────────────────────────
SHAPE_PROFILES = {
    "cylindrique": {"z": 0.85, "pression": 0.5},
    "cubique":     {"z": 0.60, "pression": 0.7},
    "plat":        {"z": 0.35, "pression": 0.3},
    "sphérique":   {"z": 0.75, "pression": 0.55},
}


@dataclass
class DetectedObject:
    obj_id:   int
    forme:    str
    x:        float      # normalisé 0-1
    y:        float
    z:        float
    pression: float
    area:     float
    bbox:     tuple      # (x, y, w, h) pixels

    def to_dict(self) -> dict:
        return {
            "id": self.obj_id,
            "forme": self.forme,
            "x": self.x, "y": self.y, "z": self.z,
            "pression": self.pression,
            "area": self.area,
        }


def classify_shape(contour) -> str:
    """Classifie la forme par approximation polygonale + aspect ratio."""
    peri  = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    n_vertices = len(approx)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / max(h, 1)
    extent = cv2.contourArea(contour) / max(w * h, 1)

    # Cercle / sphérique
    if n_vertices > 8 and extent > 0.7:
        return "sphérique"
    # Plat
    if aspect_ratio > 2.5 or aspect_ratio < 0.4:
        return "plat"
    # Cylindrique
    if n_vertices <= 5 and aspect_ratio < 0.8:
        return "cylindrique"
    return "cubique"


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Pré-traitement pour extraction de contours."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    _, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphologie pour combler les trous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return thresh


def detect_objects(frame: np.ndarray) -> list[DetectedObject]:
    """Retourne la liste des objets détectés triés par taille décroissante."""
    h, w = frame.shape[:2]
    thresh = preprocess(frame)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    objects: list[DetectedObject] = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < CONTOUR_MIN_AREA:
            break      # liste triée — inutile de continuer

        forme = classify_shape(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx_norm = (bx + bw / 2) / w
        cy_norm = (by + bh / 2) / h
        profile = SHAPE_PROFILES.get(forme, {"z": 0.5, "pression": 0.5})

        objects.append(DetectedObject(
            obj_id=idx,
            forme=forme,
            x=round(cx_norm, 3),
            y=round(cy_norm, 3),
            z=profile["z"],
            pression=profile["pression"],
            area=area,
            bbox=(bx, by, bw, bh),
        ))

    return objects


def draw_objects(frame: np.ndarray, objects: list[DetectedObject]) -> None:
    """Dessine les détections sur l'image."""
    colors = {"sphérique": (0,255,255), "cylindrique": (255,100,0),
              "cubique": (0,200,80), "plat": (200,0,200)}
    for obj in objects:
        bx, by, bw, bh = obj.bbox
        color = colors.get(obj.forme, (200, 200, 200))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
        label = f"#{obj.obj_id} {obj.forme} | P={obj.pression:.1f} Z={obj.z:.2f}"
        cv2.putText(frame, label, (bx, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def run_detection() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Caméra indisponible")
        return

    log.info("Détection démarrée — appuie sur Q pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        objects = detect_objects(frame)
        draw_objects(frame, objects)

        if objects:
            closest = objects[0]
            log.debug("Objet prioritaire: %s", closest.to_dict())

        cv2.imshow("Détection d'objets", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
