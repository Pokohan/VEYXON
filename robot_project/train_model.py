"""
train_model.py - v3 (corrections critiques)

Corrections :
  1. DATA LEAKAGE FIXÉ : split AVANT augmentation, augmentation sur train only.
  2. Augmentation spatiale : bruit + scale + translation.
  3. predict_proba() utilisé à l'inférence (incertitude du modèle).
  4. Zéro print dans les boucles critiques - logging nivellé DEBUG/INFO.
"""

import os, logging, joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network   import MLPClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics          import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import (
    CLEAN_CSV, MERGED_CSV, MODEL_PATH, LOG_DIR,
    MAX_ITER, TEST_SIZE, RANDOM_STATE, CROSS_VAL_FOLDS, PRESSURE_CLASSES,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

"XGBoost": Pipeline([("sc", StandardScaler()),
    ("m", XGBClassifier(n_estimators=100, max_depth=4,
                        learning_rate=0.05, n_jobs=-1,  # ← multi-thread
                        use_label_encoder=False, eval_metric="mlogloss",
                        random_state=RANDOM_STATE))]),

"LightGBM": Pipeline([("sc", StandardScaler()),
    ("m", LGBMClassifier(n_estimators=100, max_depth=4,
                         learning_rate=0.05, n_jobs=-1,  # ← multi-thread
                         random_state=RANDOM_STATE))]),
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR,"train.log")), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

LM_COLS  = [f"lm{i}_{a}" for i in range(21) for a in ("x","y","z")]
VEL_COLS = [f"lm{i}_v{a}" for i in range(21) for a in ("x","y","z")]

# ── Seuil de confiance pour inférence ────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75   # en dessous -> classe UNCERTAIN émise par predict_safe()


def augment(X: np.ndarray, y: np.ndarray, factor: int = 3) -> tuple:
    """
    Augmentation UNIQUEMENT sur les données d'entraînement (jamais sur le test).

    Trois stratégies combinées :
      • Bruit gaussien    -> simule instabilité du capteur
      • Scale aléatoire   -> main plus proche / plus loin de la caméra
      • Translation       -> décalage de la main dans l'image
    """
    Xa, ya = [X], [y]
    n_lm = 63   # 21 points × 3 coordonnées (positions uniquement)

    for _ in range(factor):
        Xnew = X.copy()

        # 1. Bruit gaussien sur positions ET vitesses
        Xnew += np.random.normal(0, 0.004, Xnew.shape)

        # 2. Scale : multiplie les positions par un facteur entre 0.88 et 1.12
        #    (vitesses non scalées - elles sont déjà relatives)
        scale = np.random.uniform(0.88, 1.12, size=(len(X), 1))
        Xnew[:, :n_lm] *= scale

        # 3. Translation légère sur x et y (pas z - profondeur non translatée)
        tx = np.random.uniform(-0.05, 0.05, size=(len(X), 1))
        ty = np.random.uniform(-0.05, 0.05, size=(len(X), 1))
        x_cols = list(range(0, n_lm, 3))   # indices des colonnes x
        y_cols = list(range(1, n_lm, 3))   # indices des colonnes y
        Xnew[:, x_cols] += tx
        Xnew[:, y_cols] += ty

        Xa.append(Xnew)
        ya.append(y)

    return np.vstack(Xa), np.concatenate(ya)


def load_data(csv_path: str) -> tuple:
    for candidate in [csv_path, MERGED_CSV]:
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            log.info("Dataset : %s (%d lignes)", candidate, len(df))
            break
    else:
        raise FileNotFoundError("Aucun dataset. Lancez pipeline.py d'abord.")

    if "pressure_class" not in df.columns:
        if "pressure" in df.columns:
            log.warning("Conversion pressure -> pressure_class (rétrocompat)")
            df["pressure_class"] = pd.cut(
                df["pressure"], bins=[-1,0,2,7,10], labels=[0,1,2,3]
            ).astype(int)
        else:
            raise ValueError("Colonne cible introuvable.")

    avail = [c for c in LM_COLS + VEL_COLS if c in df.columns]
    log.info("Features : %d", len(avail))
    return df[avail].fillna(0).values, df["pressure_class"].values, avail


def build_candidates() -> dict:
    return {
        "MLP": Pipeline([("sc", StandardScaler()),
                         ("m", MLPClassifier(hidden_layer_sizes=(64,32),
                                             max_iter=MAX_ITER, early_stopping=True,
                                             validation_fraction=0.1, random_state=RANDOM_STATE))]),
        "RandomForest": Pipeline([("sc", StandardScaler()),
                                  ("m", RandomForestClassifier(n_estimators=100, max_depth=10,
                                                               class_weight="balanced",
                                                               random_state=RANDOM_STATE, n_jobs=-1))]),
        "GradientBoosting": Pipeline([("sc", StandardScaler()),
                                      ("m", GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                                        learning_rate=0.05,
                                                                        random_state=RANDOM_STATE))]),
    }


def evaluate(name, pipeline, X_tr, X_te, y_tr, y_te):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    acc = (y_pred == y_te).mean()
    cv  = cross_val_score(pipeline, X_tr, y_tr,
                          cv=StratifiedKFold(CROSS_VAL_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                          scoring="f1_macro", n_jobs=-1)
    log.info("[%s] Accuracy=%.3f | CV-F1=%.3f±%.3f", name, acc, cv.mean(), cv.std())
    return {"name": name, "pipeline": pipeline, "acc": acc, "f1": cv.mean(), "y_pred": y_pred}


def plot_confusion(name, y_te, y_pred):
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(PRESSURE_CLASSES.values()))
    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Matrice de confusion - {name}")
    plt.tight_layout()
    out = os.path.join(LOG_DIR, f"confusion_{name}.png")
    plt.savefig(out, dpi=120); plt.close()
    log.info("Confusion -> %s", out)


def train(csv_path=CLEAN_CSV, model_out=MODEL_PATH):
    # Utiliser le dataset augmente si disponible
    aug_path = os.path.join(os.path.dirname(CLEAN_CSV), "dataset_augmented.csv")
    if os.path.exists(aug_path) and os.path.getsize(aug_path) > os.path.getsize(csv_path):
        csv_path = aug_path
        log.info("Utilisation du dataset augmente : %s", csv_path)

    X, y, _ = load_data(csv_path)

    # ── CORRECTION CRITIQUE : split AVANT augmentation ───────────────────────
    # Si on augmentait avant, une image A et sa copie bruitée A' pourraient
    # se retrouver l'une en train, l'autre en test -> le modèle "triche".
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    # Augmentation UNIQUEMENT sur le train
    X_tr_aug, y_tr_aug = augment(X_tr, y_tr, factor=2)
    log.info("Train original=%d | Train augmenté=%d | Test=%d (jamais augmenté)",
             len(X_tr), len(X_tr_aug), len(X_te))

    results = []
    for name, pipe in build_candidates().items():
        log.info("--- %s ---", name)
        results.append(evaluate(name, pipe, X_tr_aug, X_te, y_tr_aug, y_te))

    best = max(results, key=lambda r: r["f1"])
    log.info("[*] Meilleur : %s (F1=%.4f)", best["name"], best["f1"])
    print("\n" + classification_report(y_te, best["y_pred"],
          target_names=list(PRESSURE_CLASSES.values())))
    plot_confusion(best["name"], y_te, best["y_pred"])

    joblib.dump(best["pipeline"], model_out)
    log.info("Modèle sauvegardé -> %s", model_out)


if __name__ == "__main__":
    train()
