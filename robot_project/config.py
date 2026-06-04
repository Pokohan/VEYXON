"""
config.py — Configuration centralisée du projet Robot Autonome
Modifier ce fichier pour adapter le projet à votre environnement.
"""

import os

# ─── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset_clean")
RAW_CSV         = os.path.join(BASE_DIR, "dataset_raw.csv")
MERGED_CSV      = os.path.join(DATASET_DIR, "dataset_merged.csv")
CLEAN_CSV       = os.path.join(BASE_DIR, "dataset_clean.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "pressure_model.pkl")
DEEP_MODEL_PATH = os.path.join(BASE_DIR, "pressure_model_lstm.pkl")
ENCODER_PATH    = os.path.join(BASE_DIR, "forme_encoder.pkl")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

# Fallback vers le modèle LSTM existant si le modèle principal est absent.
if not os.path.exists(MODEL_PATH) and os.path.exists(DEEP_MODEL_PATH):
    MODEL_PATH = DEEP_MODEL_PATH

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Capture (main.py) ─────────────────────────────────────────────────────────
CAMERA_INDEX          = 0          # index caméra (0 = webcam par défaut)
MAX_HANDS             = 1
MIN_DETECTION_CONF    = 0.75
MIN_TRACKING_CONF     = 0.75
PRESSURE_MIN          = 0
PRESSURE_MAX          = 10
KEY_PRESSURE_UP       = ord('8')
KEY_PRESSURE_DOWN     = ord('2')
KEY_QUIT              = ord('q')

# ─── Modèle (train_model.py) ───────────────────────────────────────────────────
HIDDEN_LAYERS         = (128, 64, 32)
MAX_ITER              = 1000
TEST_SIZE             = 0.2
RANDOM_STATE          = 42
CROSS_VAL_FOLDS       = 3

# ─── Règles de préhension par objet ────────────────────────────────────────────
# (pression_min, pression_max) sur une échelle 0-10
GRIP_RULES = {
    "oeuf":         (1, 3),
    "fruit":        (2, 5),
    "verre":        (2, 5),
    "bouteille":    (3, 7),
    "gourde":       (3, 7),
    "brique":       (6, 10),
    "boite":        (3, 7),
    "livre":        (2, 6),
    "stylo":        (2, 4),
    "manette":      (3, 7),
    "télécommande": (2, 5),
}
# DEFAULT_GRIP_RULE supprimé intentionnellement.
# Un objet inconnu NE doit PAS être serré avec une valeur arbitraire.
# Le robot doit passer en mode manuel ou refuser l'action.

# ─── Détection (detection.py) ──────────────────────────────────────────────────
CONTOUR_MIN_AREA      = 2000
BLUR_KERNEL           = (7, 7)
THRESHOLD_VALUE       = 127

# Formes géométriques : uniquement la hauteur Z de pose du bras.
# La PRESSION est prédite par le modèle ML — ne jamais la coder ici.
SHAPE_Z = {
    "cylindrique": 0.85,
    "cubique":     0.60,
    "sphérique":   0.75,
    "plat":        0.35,
}

# ─── Actions robot (robot_autonome.py) ────────────────────────────────────────
ACTIONS = {0: "avancer", 1: "tourner_gauche", 2: "tourner_droite", 3: "stop"}
OBSTACLE_CRITICAL     = 15   # cm — seuil "danger immédiat"
OBSTACLE_WARN         = 30   # cm — seuil "ralentir"

# ─── Classification pression (4 classes) ──────────────────────────────────────
# Le modèle prédit une CLASSE, pas une valeur flottante.
# 0=Vide  1=Faible  2=Optimal  3=Trop fort
PRESSURE_CLASSES      = {0: "Vide", 1: "Faible", 2: "Optimal", 3: "Trop fort"}
# Frontières de classe (sur l'échelle 0-10)
PRESSURE_BOUNDARIES   = [0, 2, 7, 10]   # [0-2[ → Faible, [2-7] → Optimal, ]7-10] → Trop fort

# ─── Filtrage temporel (robot_autonome.py) ────────────────────────────────────
PRESSURE_SMOOTHING_WINDOW = 5    # nb de frames pour moyenne mobile

# ─── Constantes robot_autonome.py ─────────────────────────────────────────────
STABILITY_FRAMES    = 8      # frames stables requises avant GRIP (dead zone)
GRACE_PERIOD_FRAMES = 15     # tolérance si main perdue avant EMERGENCY
CONFIDENCE_MIN      = 0.75   # seuil predict_proba — en dessous = UNCERTAIN
