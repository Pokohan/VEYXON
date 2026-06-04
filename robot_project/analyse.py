"""
analyse.py — Analyse statistique du dataset (v2 Windows-compatible)
- Utilise pressure_class au lieu de pressure
- Suppression des caracteres speciaux dans les logs (compatibilite cp1252)
"""

import os, logging
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CLEAN_CSV, MERGED_CSV, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "analyse.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

MIN_ROWS_PER_OBJECT = 200
CLASS_NAMES = {0: "Vide", 1: "Faible", 2: "Optimal", 3: "Trop fort"}


def load_dataset(path: str) -> pd.DataFrame:
    for candidate in [path, MERGED_CSV]:
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            log.info("Dataset charge : %s (%d lignes)", candidate, len(df))
            return df
    raise FileNotFoundError("Lancez d'abord main.py puis merge_csv.py")


def analyse(csv_path: str = CLEAN_CSV) -> None:
    df = load_dataset(csv_path)

    # Accepte pressure_class (nouveau) ou pressure (ancien)
    if "pressure_class" in df.columns:
        target_col = "pressure_class"
    elif "pressure" in df.columns:
        target_col = "pressure"
    else:
        log.error("Aucune colonne de pression trouvee (pressure_class ou pressure).")
        return

    has_objet = "objet" in df.columns
    has_hand  = "hand_label" in df.columns

    print("\n" + "-" * 55)
    print("  RAPPORT D'ANALYSE DU DATASET")
    print("-" * 55)
    print(f"  Lignes totales  : {len(df):>8,}")
    print(f"  Colonnes        : {len(df.columns):>8}")
    print(f"  NaN             : {df.isna().sum().sum():>8,}")
    print(f"  Colonne cible   : {target_col}")

    # Distribution des classes
    print("\n  Distribution :")
    for cls, count in df[target_col].value_counts().sort_index().items():
        label = CLASS_NAMES.get(int(cls), str(cls))
        pct   = count / len(df) * 100
        bar   = "#" * int(pct / 2)
        print(f"  Classe {cls} ({label:<10}) : {count:>5} ({pct:5.1f}%)  {bar}")

    # Par objet
    report_rows = []
    if has_objet:
        print("\n  Par objet :")
        for obj, grp in df.groupby("objet"):
            n    = len(grp)
            flag = "  /!\\ PEU DE DONNEES" if n < MIN_ROWS_PER_OBJECT else ""
            dist = grp[target_col].value_counts().sort_index().to_dict()
            print(f"  {obj:<22} n={n:>5}{flag}")
            for cls, cnt in dist.items():
                print(f"    Classe {cls} ({CLASS_NAMES.get(int(cls),'?'):<10}): {cnt}")
            report_rows.append({"objet": obj, "n": n, **{f"cls_{k}": v for k,v in dist.items()}})

    # Par main
    if has_hand:
        print("\n  Par main :")
        for hand, grp in df.groupby("hand_label"):
            print(f"  {hand:<10} n={len(grp):>5}")

    print("-" * 55 + "\n")

    # Export rapport CSV
    if report_rows:
        report_df = pd.DataFrame(report_rows)
        report_path = os.path.join(LOG_DIR, "analyse_report.csv")
        report_df.to_csv(report_path, index=False)
        log.info("Rapport CSV -> %s", report_path)

    # Graphique distribution
    counts = df[target_col].value_counts().sort_index()
    labels = [CLASS_NAMES.get(int(i), str(i)) for i in counts.index]
    colors = ["#888780", "#3B8BD4", "#1D9E75", "#D85A30"]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, counts.values, color=colors[:len(labels)], edgecolor="white", width=0.6)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(val), ha="center", va="bottom", fontsize=10)
    plt.title("Distribution des classes de pression")
    plt.ylabel("Nombre de frames")
    plt.tight_layout()
    chart_path = os.path.join(LOG_DIR, "pressure_distribution.png")
    plt.savefig(chart_path, dpi=130)
    plt.close()
    log.info("Graphique -> %s", chart_path)


if __name__ == "__main__":
    analyse()
