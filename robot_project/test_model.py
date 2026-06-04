"""
test_model.py — Test et validation du modèle entraîné
Remplace : test_model.py, test_model_joblib.py, test_model_simple.py, test_pickle_only.py

Usage :
  python test_model.py                     # test interactif
  python test_model.py --batch             # test sur le dataset complet
  python test_model.py --landmarks         # mode prédiction pression (landmarks)
"""

import argparse
import os
import logging
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, CLEAN_CSV, LOG_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        log.error("Modèle introuvable : %s — lancez d'abord train_model.py", path)
        raise FileNotFoundError(path)
    model = joblib.load(path)
    log.info("Modèle chargé : %s", type(model).__name__)
    return model


def test_interactive(model) -> None:
    """Test manuel avec entrées saisies dans le terminal."""
    print("\n═ Test interactif ══════════════════════════")
    print("Entrez 126 valeurs de landmarks (ou appuyez Entrée pour un exemple aléatoire)")

    while True:
        raw = input("\nLandmarks (126 valeurs séparées par virgule) > ").strip()
        if not raw:
            X = np.random.rand(1, 126)
            print("→ Exemple aléatoire généré")
        else:
            try:
                vals = [float(v) for v in raw.split(",")]
                assert len(vals) == 126
                X = np.array(vals).reshape(1, -1)
            except (ValueError, AssertionError):
                print("❌ 63 valeurs attendues, séparées par des virgules.")
                continue

        pred = model.predict(X)[0]
        print(f"  Pression prédite : {pred}" if isinstance(pred, int) else f"  Pression prédite : {pred:.2f} / 10")

        if input("Continuer ? [O/n] ").strip().lower() == "n":
            break


def pressure_to_class(p):
    if p <= 0: return 0
    if p <= 2: return 1
    if p <= 7: return 2
    return 3


def test_batch(model, csv_path: str = CLEAN_CSV) -> None:
    """Évalue le modèle sur le dataset entier."""
    from sklearn.metrics import accuracy_score, classification_report

    if not os.path.exists(csv_path):
        log.error("Dataset introuvable : %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    lm_cols = [f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
    vel_cols = [f"lm{i}_v{axis}" for i in range(21) for axis in ("x", "y", "z")]
    available = [c for c in lm_cols + vel_cols if c in df.columns]
    X = df[available].values

    if "pressure_class" in df.columns:
        y = df["pressure_class"].values
    elif "pressure" in df.columns:
        log.warning("Conversion pressure -> pressure_class (rétrocompat)")
        y = df["pressure"].clip(0, 10).apply(pressure_to_class).values
    else:
        log.error("Colonne cible introuvable : pressure_class ou pressure")
        return

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print("\n═ Évaluation batch ═════════════════════════")
    print(f"  Lignes évaluées : {len(y):,}")
    print(f"  Accuracy       : {acc:.3f}")
    print(classification_report(y, y_pred,
          target_names=["Vide","Faible","Optimal","Trop fort"]))

    if "objet" in df.columns and "pressure_class" in df.columns:
        print("\n  Précision par objet :")
        for obj, grp in df.groupby("objet"):
            Xi = grp[available].values
            yi = grp["pressure_class"].values
            pi = model.predict(Xi)
            obj_acc = accuracy_score(yi, pi)
            print(f"  {obj:<22} acc={obj_acc:.3f}")


def inspect_model(model) -> None:
    """Affiche des informations sur le modèle."""
    print("\n═ Inspection du modèle ══════════════════════")
    print(f"  Type    : {type(model).__name__}")
    if hasattr(model, "steps"):
        for name, step in model.steps:
            print(f"  Étape   : {name} — {type(step).__name__}")
    if hasattr(model, "n_features_in_"):
        print(f"  Features: {model.n_features_in_}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test du modèle robot")
    parser.add_argument("--batch",     action="store_true", help="Évaluation sur dataset complet")
    parser.add_argument("--landmarks", action="store_true", help="Mode prédiction interactive")
    args = parser.parse_args()

    model = load_model()
    inspect_model(model)

    if args.batch:
        test_batch(model)
    else:
        test_interactive(model)
