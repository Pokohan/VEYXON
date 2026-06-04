"""
analyze_video_pressure.py — Analyse de pression à partir d'une vidéo existante

Usage :
  python analyze_video_pressure.py <chemin_video>                     # analyse + affichage
  python analyze_video_pressure.py <chemin_video> --save-output       # sauvegarde vidéo annotée
  python analyze_video_pressure.py <chemin_video> --export-csv        # export CSV résultats

Features :
  • Détection des landmarks MediaPipe sur chaque frame
  • Prédiction de la classe de pression (Vide/Faible/Optimal/Trop fort)
  • Visualisation en temps réel
  • Sauvegarde des résultats en CSV
  • Génération de graphiques statistiques
"""

import argparse
import os
import sys
import logging
import time
import collections
import csv
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config import (
    MODEL_PATH, MAX_HANDS,
    MIN_DETECTION_CONF, MIN_TRACKING_CONF,
    LOG_DIR, CONFIDENCE_MIN,
)
from train_model_deep import HandLSTM, LSTMWrapper

try:
    import mediapipe as mp
    HAS_MP_SOLUTIONS = hasattr(mp, "solutions")
except ImportError:
    mp = None
    HAS_MP_SOLUTIONS = False

if HAS_MP_SOLUTIONS:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    USE_TASKS = False
else:
    from mediapipe.tasks.python.vision import hand_landmarker as mp_hands
    from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
    from mediapipe.tasks.python.vision.core import image as mp_image
    USE_TASKS = True

# ── Couleurs ──────────────────────────────────────────────────────────────────
COLORS = {
    0: (150, 150, 150),   # Vide      - gris
    1: (255, 140,   0),   # Faible    - orange
    2: (50,  200,  50),   # Optimal   - vert
    3: (40,   40, 220),   # Trop fort - rouge
}
CLASS_NAMES = {0: "Vide", 1: "Faible", 2: "Optimal", 3: "Trop fort"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class PressureAnalyzer:
    def __init__(self, video_path: str, model_task_path: str | None = None, pressure_model_path: str | None = None):
        """Initialise l'analyseur de pression pour une vidéo."""
        self.video_path = video_path
        self.model_task_path = model_task_path
        self.pressure_model_path = pressure_model_path
        self.model = None
        self.cap = None
        self.results = []
        self.fps = 0
        self.frame_count = 0
        self.total_frames = 0
        
        self._load_model()
        self._open_video()
    
    def _load_model(self):
        """Charge le modèle ML entraîné."""
        model_path = self.pressure_model_path or MODEL_PATH
        if not os.path.exists(model_path):
            log.error(f"Modèle introuvable : {model_path}")
            raise FileNotFoundError(model_path)
        
        self.model = joblib.load(model_path)
        log.info(f"[OK] Modèle chargé : {type(self.model).__name__} ({model_path})")
    
    def _open_video(self):
        """Ouvre la vidéo et récupère ses propriétés."""
        if not os.path.exists(self.video_path):
            log.error(f"Vidéo introuvable : {self.video_path}")
            raise FileNotFoundError(self.video_path)
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            log.error(f"Impossible d'ouvrir la vidéo : {self.video_path}")
            raise IOError(self.video_path)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        log.info(f"[OK] Vidéo ouverte : {self.total_frames} frames @ {self.fps:.1f} FPS")
    
    def _get_features(self, landmarks, prev_pos, dt):
        """Extrait les 126 features (positions + vitesses) depuis les landmarks."""
        points = getattr(landmarks, "landmark", landmarks)
        bx = points[0].x
        by = points[0].y
        bz = points[0].z

        pos = np.array([
            [lm.x - bx, lm.y - by, lm.z - bz]
            for lm in points
        ])  # (21, 3)

        vel = (pos - prev_pos) / dt if prev_pos is not None else np.zeros_like(pos)
        features = np.concatenate([pos.flatten(), vel.flatten()])
        return features, pos
    
    def _predict_pressure(self, features):
        """Prédit la classe de pression et la confiance."""
        X = features.reshape(1, -1)
        
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                pclass = int(np.argmax(proba))
                conf = float(proba[pclass])
            else:
                pclass = int(self.model.predict(X)[0])
                conf = 1.0
        except Exception as e:
            log.warning(f"Erreur prédiction : {e}")
            pclass = 0
            conf = 0.0
        
        return pclass, conf
    
    def _draw_frame_info(self, frame, pclass, conf, frame_num):
        """Ajoute les infos de pression sur le frame."""
        h, w = frame.shape[:2]
        color = COLORS.get(pclass, (200, 200, 200))
        name = CLASS_NAMES.get(pclass, "?")
        
        # Panneau principal (haut-gauche)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (240, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (10, 10), (240, 150), (80, 80, 80), 1)
        
        # Classe prédite
        cv2.putText(frame, "PRESSION", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, name, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Confiance
        cv2.putText(frame, f"Confiance : {conf:.0%}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        bar_w = int(conf * 210)
        cv2.rectangle(frame, (20, 112), (230, 124), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 112), (20 + bar_w, 124), color, -1)
        
        # Timestamp
        timestamp = frame_num / self.fps
        cv2.putText(frame, f"Frame {frame_num}/{self.total_frames} ({timestamp:.2f}s)", 
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def analyze(self, show_video=True, save_output=False, export_csv=False):
        """
        Analyse la vidéo.
        
        Args:
            show_video (bool): Afficher la vidéo annotée en temps réel
            save_output (bool): Sauvegarder la vidéo annotée
            export_csv (bool): Exporter les résultats en CSV
        """
        log.info("=" * 60)
        log.info("Début de l'analyse de pression")
        log.info("=" * 60)
        
        # Préparation writer vidéo output
        video_writer = None
        if save_output:
            output_path = self.video_path.replace(".mp4", "_analyzed.mp4").replace(".avi", "_analyzed.avi")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                          (frame_width, frame_height))
            log.info(f"Vidéo output : {output_path}")
        
        # État
        prev_pos = None
        prev_t = time.time()

        if USE_TASKS:
            if not self.model_task_path or not os.path.exists(self.model_task_path):
                raise FileNotFoundError(
                    "Aucun modèle hand_landmarker.task trouvé. "
                    "Installez une version de MediaPipe avec mp.solutions ou fournissez "
                    "--model-task <chemin_vers_hand_landmarker.task>."
                )
            detector = mp_hands.HandLandmarker.create_from_model_path(self.model_task_path)

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                timestamp = self.frame_count / self.fps

                # Pré-traitement
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image_obj = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
                results = detector.detect(mp_image_obj)

                pclass = 0
                conf = 0.0
                hand_detected = False

                if results.hand_landmarks:
                    hand_detected = True
                    for landmarks in results.hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            landmarks,
                            mp_hands.HandLandmarksConnections.HAND_CONNECTIONS,
                        )

                        # Extraire features et prédire
                        now = time.time()
                        dt = max(now - prev_t, 1e-6)
                        prev_t = now
                        
                        features, prev_pos = self._get_features(landmarks, prev_pos, dt)
                        pclass, conf = self._predict_pressure(features)

                frame = self._draw_frame_info(frame, pclass, conf, self.frame_count)

                status_text = "MAIN DÉTECTÉE" if hand_detected else "AUCUNE MAIN"
                status_color = (50, 200, 50) if hand_detected else (40, 40, 220)
                cv2.putText(frame, status_text, (w - 250, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                self.results.append({
                    'frame': self.frame_count,
                    'timestamp': timestamp,
                    'pressure_class': pclass,
                    'pressure_name': CLASS_NAMES[pclass],
                    'confidence': conf,
                    'hand_detected': hand_detected
                })

                if show_video:
                    cv2.imshow("Analyse de pression", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.warning("Analyse interrompue par l'utilisateur")
                        break

                if video_writer is not None:
                    video_writer.write(frame)

                progress = (self.frame_count / self.total_frames) * 100
                if self.frame_count % 30 == 0:
                    log.info(f"Progression : {progress:.1f}% ({self.frame_count}/{self.total_frames})")
        else:
            with mp_hands.Hands(
                max_num_hands=MAX_HANDS,
                min_detection_confidence=MIN_DETECTION_CONF,
                min_tracking_confidence=MIN_TRACKING_CONF,
            ) as detector:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    self.frame_count += 1
                    timestamp = self.frame_count / self.fps

                    # Pré-traitement
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.process(rgb)

                    pclass = 0
                    conf = 0.0
                    hand_detected = False

                    if results.multi_hand_landmarks:
                        hand_detected = True
                        
                        for landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, landmarks,
                                mp_hands.HAND_CONNECTIONS,
                            )

                            # Extraire features et prédire
                            now = time.time()
                            dt = max(now - prev_t, 1e-6)
                            prev_t = now
                            
                            features, prev_pos = self._get_features(landmarks, prev_pos, dt)
                            pclass, conf = self._predict_pressure(features)

                    frame = self._draw_frame_info(frame, pclass, conf, self.frame_count)

                    status_text = "MAIN DÉTECTÉE" if hand_detected else "AUCUNE MAIN"
                    status_color = (50, 200, 50) if hand_detected else (40, 40, 220)
                    cv2.putText(frame, status_text, (w - 250, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                    self.results.append({
                        'frame': self.frame_count,
                        'timestamp': timestamp,
                        'pressure_class': pclass,
                        'pressure_name': CLASS_NAMES[pclass],
                        'confidence': conf,
                        'hand_detected': hand_detected
                    })

                    if show_video:
                        cv2.imshow("Analyse de pression", frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            log.warning("Analyse interrompue par l'utilisateur")
                            break

                    if video_writer is not None:
                        video_writer.write(frame)

                    progress = (self.frame_count / self.total_frames) * 100
                    if self.frame_count % 30 == 0:
                        log.info(f"Progression : {progress:.1f}% ({self.frame_count}/{self.total_frames})")
        
        # Nettoyage
        if show_video:
            cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        self.cap.release()
        
        log.info("=" * 60)
        log.info(f"Analyse terminée : {self.frame_count} frames traitées")
        log.info("=" * 60)
        
        # Export CSV
        if export_csv:
            self._export_csv()
        
        # Génération de rapport
        self._generate_report()
    
    def _export_csv(self):
        """Exporte les résultats en CSV."""
        output_dir = LOG_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"pressure_analysis_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        
        log.info(f"[OK] Résultats exportés : {csv_path}")
        return csv_path
    
    def _generate_report(self):
        """Génère un rapport d'analyse."""
        if not self.results:
            log.warning("Aucun résultat à analyser")
            return
        
        df = pd.DataFrame(self.results)
        
        # Statistiques
        log.info("\n" + "=" * 60)
        log.info("RAPPORT D'ANALYSE")
        log.info("=" * 60)
        log.info(f"Vidéo : {os.path.basename(self.video_path)}")
        log.info(f"Durée totale : {self.frame_count / self.fps:.1f}s")
        log.info(f"Frames avec main détectée : {df['hand_detected'].sum()} / {len(df)}")
        
        # Répartition des classes
        log.info("\nRépartition des pressions :")
        for cls_id, cls_name in CLASS_NAMES.items():
            count = (df['pressure_class'] == cls_id).sum()
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            log.info(f"  {cls_name:12s} : {count:4d} frames ({pct:5.1f}%)")
        
        # Confiance moyenne
        avg_conf = df['confidence'].mean()
        log.info(f"\nConfiance moyenne : {avg_conf:.1f}%")
        
        # Génération de graphiques
        self._plot_results(df)
    
    def _plot_results(self, df):
        """Génère des graphiques d'analyse."""
        output_dir = LOG_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f"Analyse de pression - {os.path.basename(self.video_path)}")
        
        # Graphique 1 : Evolution de la classe de pression
        ax = axes[0]
        colors_list = [COLORS.get(int(cls), (200, 200, 200)) for cls in df['pressure_class']]
        colors_list = [(b/255, g/255, r/255) for r, g, b in colors_list]
        ax.scatter(df['timestamp'], df['pressure_class'], c=colors_list, s=20, alpha=0.6)
        ax.set_ylabel("Classe de pression")
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Vide", "Faible", "Optimal", "Trop fort"])
        ax.grid(True, alpha=0.3)
        ax.set_title("Évolution de la classe de pression au cours du temps")
        
        # Graphique 2 : Confiance
        ax = axes[1]
        ax.plot(df['timestamp'], df['confidence'] * 100, linewidth=2, color='skyblue')
        ax.fill_between(df['timestamp'], 0, df['confidence'] * 100, alpha=0.3)
        ax.set_ylabel("Confiance (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_title("Confiance du modèle au cours du temps")
        
        # Graphique 3 : Distribution
        ax = axes[2]
        class_counts = df['pressure_class'].value_counts().sort_index()
        class_labels = [CLASS_NAMES.get(i, f"Class {i}") for i in class_counts.index]
        colors_bar = [tuple(c/255 for c in COLORS.get(i, (200, 200, 200))) 
                      for i in class_counts.index]
        ax.bar(class_labels, class_counts.values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax.set_ylabel("Nombre de frames")
        ax.set_title("Distribution des classes de pression")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les comptes sur les barres
        for i, (label, count) in enumerate(zip(class_labels, class_counts.values)):
            ax.text(i, count, str(int(count)), ha='center', va='bottom')
        
        plt.xlabel("Temps (s)")
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"pressure_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        log.info(f"[OK] Graphiques sauvegardés : {plot_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyse la pression d'une main dans une vidéo"
    )
    parser.add_argument(
        "video",
        help="Chemin vers le fichier vidéo à analyser"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Sauvegarder la vidéo annotée"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Exporter les résultats en CSV"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Ne pas afficher la vidéo en temps réel"
    )
    parser.add_argument(
        "--model-task",
        help="Chemin vers le modèle hand_landmarker.task lorsque mediapipe n'inclut pas mp.solutions"
    )
    parser.add_argument(
        "--pressure-model",
        help="Chemin vers le modèle de pression (joblib .pkl) à utiliser"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = PressureAnalyzer(
            args.video,
            model_task_path=args.model_task,
            pressure_model_path=args.pressure_model,
        )
        analyzer.analyze(
            show_video=not args.no_display,
            save_output=args.save_output,
            export_csv=args.export_csv
        )
    except (FileNotFoundError, IOError) as e:
        log.error(f"Erreur : {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Interrompus par l'utilisateur")
        sys.exit(0)


if __name__ == "__main__":
    main()
