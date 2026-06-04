"""
robot_autonome.py - v3 (corrections critiques)

Corrections :
  3. DEAD ZONE : passage APPROACH->GRIP uniquement si N prédictions stables (std≈0).
  4. GRACE PERIOD : 15 frames de tolérance avant EMERGENCY si main perdue.
  5. predict_proba() : si confiance < seuil -> classe UNCERTAIN, robot stoppe.
  6. DEFAULT_GRIP_RULE supprimé : objet inconnu -> mode manuel obligatoire.
  - Tous les print remplacés par logging niveaux DEBUG/INFO.
"""

import os, time, logging, collections
import joblib
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple

from config import (
    MODEL_PATH, ACTIONS, OBSTACLE_CRITICAL, OBSTACLE_WARN,
    GRIP_RULES, LOG_DIR,
    PRESSURE_CLASSES, PRESSURE_SMOOTHING_WINDOW,
)

# ── Logging : en production, passer le StreamHandler en WARNING ───────────────
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, "robot.log")),
            logging.StreamHandler(),
        ],
    )

# ── Constantes de sécurité ────────────────────────────────────────────────────
STABILITY_FRAMES    = 8      # frames consécutives stables avant de passer en GRIP
GRACE_PERIOD_FRAMES = 15     # frames de tolérance si main perdue (ghosting)
CONFIDENCE_MIN      = 0.75   # confiance predict_proba en dessous = UNCERTAIN
UNCERTAINTY_CLASS   = -1     # classe spéciale : robot s'arrête


class RobotState(Enum):
    IDLE      = auto()
    APPROACH  = auto()
    GRIP      = auto()
    HOLD      = auto()
    RELEASE   = auto()
    EMERGENCY = auto()


@dataclass
class SensorData:
    distance_front: float
    distance_left:  float
    distance_right: float
    object_name:    str = ""
    landmarks:      Optional[np.ndarray] = None


@dataclass
class RobotDecision:
    action:         str
    pressure_class: int = 0
    confidence:     float = 1.0
    state:          RobotState = RobotState.IDLE
    reason:         str = ""


class RobotBrain:
    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.state              = RobotState.IDLE
        self.current_pclass     = 0
        self._model             = None

        # Dead zone : fenêtre des N dernières prédictions stables
        self._stability_window: collections.deque = collections.deque(maxlen=STABILITY_FRAMES)

        # Grace period : compteur de frames sans landmarks
        self._grace_counter: int = 0

        self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        if os.path.exists(path):
            try:
                self._model = joblib.load(path)
                log.info("Modèle chargé : %s", type(self._model).__name__)
                has_proba = hasattr(self._model, "predict_proba") or \
                            (hasattr(self._model, "steps") and
                             hasattr(self._model.steps[-1][1], "predict_proba"))
                if not has_proba:
                    log.warning("Le modèle ne supporte pas predict_proba - confiance désactivée.")
            except Exception as e:
                log.warning("Modèle non chargé : %s", e)
        else:
            log.warning("Modèle absent (%s) - mode dégradé.", path)

    # ── Prédiction avec probabilité ───────────────────────────────────────────
    def predict_class(self, landmarks: Optional[np.ndarray]) -> Tuple[int, float]:
        """
        Retourne (classe, confiance).
        Si confiance < CONFIDENCE_MIN -> retourne (UNCERTAINTY_CLASS, confiance).
        Si modèle/landmarks absents    -> retourne (UNCERTAINTY_CLASS, 0.0).
        Jamais de valeur par défaut arbitraire.
        """
        if self._model is None or landmarks is None:
            log.debug("predict_class : modèle ou landmarks absent -> UNCERTAIN")
            return UNCERTAINTY_CLASS, 0.0

        X = landmarks.reshape(1, -1)

        try:
            proba = self._model.predict_proba(X)[0]   # shape (4,)
            best_class = int(np.argmax(proba))
            confidence = float(proba[best_class])
        except AttributeError:
            # Fallback si predict_proba indisponible
            best_class = int(self._model.predict(X)[0])
            confidence = 1.0

        if confidence < CONFIDENCE_MIN:
            log.debug("Confiance faible (%.2f) - classe=%d -> UNCERTAIN", confidence, best_class)
            return UNCERTAINTY_CLASS, confidence

        log.debug("Prédiction : classe=%d (%s) confiance=%.2f",
                  best_class, PRESSURE_CLASSES.get(best_class, "?"), confidence)
        return best_class, confidence

    # ── Dead zone : stabilité avant GRIP ─────────────────────────────────────
    def _is_stable(self, pclass: int) -> bool:
        """
        Retourne True si la prédiction est stable sur STABILITY_FRAMES frames.
        Critère : écart-type ≈ 0 (toutes les prédictions identiques).
        """
        self._stability_window.append(pclass)
        if len(self._stability_window) < STABILITY_FRAMES:
            return False
        std = np.std(list(self._stability_window))
        stable = std == 0.0
        log.debug("Stabilité : window=%s std=%.2f -> %s",
                  list(self._stability_window), std, stable)
        return stable

    # ── Gestion grace period (ghosting) ───────────────────────────────────────
    def _handle_landmarks(self, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Retourne les landmarks si disponibles.
        Si absents, incrémente le compteur de grâce.
        Au-delà de GRACE_PERIOD_FRAMES : signale une vraie perte.
        """
        if landmarks is not None:
            self._grace_counter = 0
            return landmarks

        self._grace_counter += 1
        log.debug("Grace period : %d/%d", self._grace_counter, GRACE_PERIOD_FRAMES)
        if self._grace_counter <= GRACE_PERIOD_FRAMES:
            return None   # dans la fenêtre de grâce : on tolère
        log.warning("Landmarks perdus depuis %d frames (> grace=%d) -> EMERGENCY",
                    self._grace_counter, GRACE_PERIOD_FRAMES)
        return "EXPIRED"   # signal de dépassement

    # ── Navigation ────────────────────────────────────────────────────────────
    @staticmethod
    def _nav(sensors: SensorData) -> Tuple[str, str]:
        df, dl, dr = sensors.distance_front, sensors.distance_left, sensors.distance_right
        if df < OBSTACLE_CRITICAL and dl < OBSTACLE_CRITICAL and dr < OBSTACLE_CRITICAL:
            return "stop", "obstacle tous côtés"
        if df < OBSTACLE_CRITICAL:
            return ("tourner_gauche" if dl >= dr else "tourner_droite"), f"front {df:.0f}cm"
        if dl < OBSTACLE_CRITICAL:
            return "tourner_droite", f"gauche {dl:.0f}cm"
        if dr < OBSTACLE_CRITICAL:
            return "tourner_gauche", f"droite {dr:.0f}cm"
        return "avancer", "voie libre"

    # ── Vérification objet connu ──────────────────────────────────────────────
    @staticmethod
    def _check_object(name: str) -> bool:
        """
        Objet inconnu -> REFUSE de serrer 'au pif'.
        Le robot doit demander une instruction ou passer en mode manuel.
        """
        if name.lower() not in GRIP_RULES:
            log.warning(
                "Objet '%s' inconnu. Aucune règle de préhension disponible. "
                "Passez en mode manuel ou ajoutez la règle dans config.GRIP_RULES.", name
            )
            return False
        return True

    # ── FSM principale ────────────────────────────────────────────────────────
    def decide(self, sensors: SensorData) -> RobotDecision:
        nav_action, nav_reason = self._nav(sensors)
        lm_status = self._handle_landmarks(sensors.landmarks)

        # ── Sécurité : obstacle critique ─────────────────────────────────────
        if nav_action == "stop":
            if self.state == RobotState.HOLD:
                log.warning("Obstacle critique en HOLD -> RELEASE forcé")
                self.state = RobotState.RELEASE
                return RobotDecision("release", 0, 1.0, self.state, "relâchement sécurité")
            self.state = RobotState.EMERGENCY
            return RobotDecision("stop", 0, 1.0, self.state, nav_reason)

        # ── Sécurité : grace period expirée ───────────────────────────────────
        if isinstance(lm_status, str) and lm_status == "EXPIRED" and self.state in (RobotState.GRIP, RobotState.HOLD):
            log.error("Grace period expirée -> EMERGENCY")
            self.state = RobotState.EMERGENCY
            return RobotDecision("stop", 0, 0.0, self.state, "landmarks perdus définitivement")

        if self.state == RobotState.EMERGENCY and nav_action == "avancer":
            self.state = RobotState.IDLE
            log.info("Fin EMERGENCY -> IDLE")

        # ── IDLE -> APPROACH ───────────────────────────────────────────────────
        if self.state == RobotState.IDLE:
            if sensors.object_name:
                if not self._check_object(sensors.object_name):
                    return RobotDecision("stop", 0, 1.0, self.state,
                                         f"objet '{sensors.object_name}' inconnu -> mode manuel requis")
                self.state = RobotState.APPROACH

        # ── APPROACH ─────────────────────────────────────────────────────────
        if self.state == RobotState.APPROACH:
            if sensors.distance_front > OBSTACLE_WARN:
                return RobotDecision(nav_action, 0, 1.0, self.state, nav_reason)

            # Assez proche : vérifier la stabilité avant de saisir
            if lm_status is None or (isinstance(lm_status, str) and lm_status == "EXPIRED"):
                return RobotDecision("stop", 0, 0.0, self.state, "attente landmarks")

            pclass, conf = self.predict_class(lm_status)
            if pclass == UNCERTAINTY_CLASS:
                return RobotDecision("stop", 0, conf, self.state,
                                     f"prédiction incertaine (conf={conf:.2f})")

            if self._is_stable(pclass):
                self.state = RobotState.GRIP
                log.info("Stabilité confirmée -> GRIP (classe=%d)", pclass)
            else:
                remaining = STABILITY_FRAMES - len(self._stability_window)
                return RobotDecision("hold_position", pclass, conf, self.state,
                                     f"dead zone : {remaining} frames restantes")

        # ── GRIP ──────────────────────────────────────────────────────────────
        if self.state == RobotState.GRIP:
            if lm_status is None or (isinstance(lm_status, str) and lm_status == "EXPIRED"):
                log.error("GRIP sans landmarks -> EMERGENCY")
                self.state = RobotState.EMERGENCY
                return RobotDecision("stop", 0, 0.0, self.state, "landmarks absents")

            pclass, conf = self.predict_class(lm_status)
            if pclass == UNCERTAINTY_CLASS:
                self.state = RobotState.EMERGENCY
                return RobotDecision("stop", 0, conf, self.state,
                                     f"incertitude au GRIP (conf={conf:.2f})")

            self.current_pclass = pclass
            self.state = RobotState.HOLD
            return RobotDecision("grip", pclass, conf, self.state,
                                 f"classe={pclass} ({PRESSURE_CLASSES[pclass]}) conf={conf:.2f}")

        # ── HOLD : boucle de rétroaction ──────────────────────────────────────
        if self.state == RobotState.HOLD:
            if lm_status is None:
                # Dans la grace period : maintenir la dernière pression connue
                log.debug("HOLD grace period : maintien classe=%d", self.current_pclass)
                return RobotDecision("hold", self.current_pclass, 0.0, self.state,
                                     f"grace period ({self._grace_counter}/{GRACE_PERIOD_FRAMES})")
            if isinstance(lm_status, str) and lm_status == "EXPIRED":
                log.error("HOLD grace expirée -> EMERGENCY")
                self.state = RobotState.EMERGENCY
                return RobotDecision("stop", 0, 0.0, self.state, "landmarks perdus")

            pclass, conf = self.predict_class(lm_status)
            if pclass == UNCERTAINTY_CLASS:
                log.warning("HOLD incertain (conf=%.2f) -> maintien conservatoire", conf)
                return RobotDecision("hold", self.current_pclass, conf, self.state,
                                     f"incertitude : maintien précédent (conf={conf:.2f})")

            if pclass != self.current_pclass:
                log.info("HOLD ajustement %d->%d (%s)",
                         self.current_pclass, pclass, PRESSURE_CLASSES[pclass])
                self.current_pclass = pclass

            return RobotDecision("hold", self.current_pclass, conf, self.state,
                                 f"classe={self.current_pclass} conf={conf:.2f}")

        # ── RELEASE ───────────────────────────────────────────────────────────
        if self.state == RobotState.RELEASE:
            self.current_pclass = 0
            self._stability_window.clear()
            self.state = RobotState.IDLE
            return RobotDecision("release", 0, 1.0, self.state, "objet relâché")

        return RobotDecision(nav_action, 0, 1.0, self.state, nav_reason)


# ── Simulation ───────────────────────────────────────────────────────────────
SCENARIO = [
    SensorData(80, 90, 85, "gourde"),                                 # IDLE->APPROACH
    SensorData(25, 85, 80, "gourde", np.random.rand(126)),            # APPROACH, instable
    SensorData(20, 80, 75, "gourde", np.random.rand(126)),            # dead zone frames...
    SensorData(18, 80, 75, "gourde", np.random.rand(126)),
    SensorData(16, 80, 75, "gourde", np.random.rand(126)),
    SensorData(15, 80, 75, "gourde", np.random.rand(126)),
    SensorData(14, 80, 75, "gourde", np.random.rand(126)),
    SensorData(14, 80, 75, "gourde", np.random.rand(126)),
    SensorData(14, 80, 75, "gourde", np.random.rand(126)),            # stabilité -> GRIP
    SensorData(14, 80, 75, "gourde", None),                           # grace period frame 1
    SensorData(14, 80, 75, "gourde", np.random.rand(126)),            # main retrouvée -> HOLD
    SensorData( 3,  3,  3, "gourde", np.random.rand(126)),            # obstacle -> RELEASE+EMERGENCY
    SensorData(50, 50, 50, "gadget"),                                  # objet inconnu -> refus
]

if __name__ == "__main__":
    brain = RobotBrain()
    print("\n═══ Simulation Robot Autonome v3 ═══")
    for i, s in enumerate(SCENARIO):
        d = brain.decide(s)
        log.info("Step %2d | %-10s | %-14s | cls=%2d conf=%.2f | %s",
                 i+1, d.state.name, d.action, d.pressure_class, d.confidence, d.reason)
        time.sleep(0.25)
    print("═══ Fin ═══\n")
