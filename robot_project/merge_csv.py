"""
merge_csv.py - Fusion de tous les CSV par objet
Améliorations :
  • Ajout automatique de la colonne 'objet'
  • Rapport détaillé de fusion
  • Vérification de cohérence des colonnes
  • Export d'un dataset global unique
"""

import os
import pandas as pd
import logging
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import DATASET_DIR, MERGED_CSV, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "merge.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def merge_object_folder(item_path, objet):
    """Fusionne tous les CSV d'un dossier objet en un seul DataFrame."""
    csv_files = [f for f in os.listdir(item_path) if f.endswith(".csv")]
    if not csv_files:
        log.warning("Aucun CSV dans %s - ignoré", item_path)
        return None

    dfs = []
    for fname in csv_files:
        path = os.path.join(item_path, fname)
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            log.error("Lecture %s : %s", path, e)

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged.insert(0, "objet", objet)          # colonne objet en première position
    out_path = os.path.join(DATASET_DIR, f"{objet}.csv")
    merged.to_csv(out_path, index=False)
    log.info("[OK] %s.csv - %d lignes", objet, len(merged))
    return merged


def merge_all() -> None:
    if not os.path.isdir(DATASET_DIR):
        log.error("Dossier dataset_clean introuvable : %s", DATASET_DIR)
        return

    all_dfs: list[pd.DataFrame] = []
    reference_cols: list[str] | None = None

    for item in sorted(os.listdir(DATASET_DIR)):
        item_path = os.path.join(DATASET_DIR, item)
        if not os.path.isdir(item_path):
            continue

        df = merge_object_folder(item_path, item)
        if df is None:
            continue

        # Vérification cohérence colonnes
        if reference_cols is None:
            reference_cols = list(df.columns)
        elif list(df.columns) != reference_cols:
            log.warning(
                "Colonnes différentes pour '%s' - fusion quand même (outer join)", item
            )
        all_dfs.append(df)

    if not all_dfs:
        log.error("Aucun DataFrame valide à fusionner.")
        return

    global_df = pd.concat(all_dfs, ignore_index=True)
    global_df.to_csv(MERGED_CSV, index=False)

    log.info("\n--- Rapport global ---")
    log.info("Fichier   : %s", MERGED_CSV)
    log.info("Lignes    : %d", len(global_df))
    log.info("Colonnes  : %d", len(global_df.columns))
    log.info("Objets    : %s", sorted(global_df["objet"].unique()))


if __name__ == "__main__":
    merge_all()
