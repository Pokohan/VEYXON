"""
clean_dataset.py - Nettoyage du dataset (v2)

Mis à jour pour :
  • Colonnes vitesse (lm_vx/vy/vz) en plus des positions
  • Cible = pressure_class (int), rétrocompatible avec 'pressure'
  • Distribution des classes (pas seulement pression brute)
"""

import os, logging
import pandas as pd
import numpy as np
from config import MERGED_CSV, CLEAN_CSV, PRESSURE_MIN, PRESSURE_MAX, LOG_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR,"clean.log")), logging.StreamHandler()])
log = logging.getLogger(__name__)

LM_COLS  = [f"lm{i}_{a}" for i in range(21) for a in ("x","y","z")]
VEL_COLS = [f"lm{i}_v{a}" for i in range(21) for a in ("x","y","z")]

def pressure_to_class(p):
    if p <= 0: return 0
    if p <= 2: return 1
    if p <= 7: return 2
    return 3

def clean(input_csv=MERGED_CSV, output_csv=CLEAN_CSV):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)
    log.info("Brut : %d lignes, %d colonnes", len(df), len(df.columns))

    # Rétrocompatibilité : convertir 'pressure' en 'pressure_class'
    if "pressure_class" not in df.columns and "pressure" in df.columns:
        df["pressure"] = df["pressure"].clip(PRESSURE_MIN, PRESSURE_MAX)
        df["pressure_class"] = df["pressure"].apply(pressure_to_class)
        log.info("Conversion pressure -> pressure_class effectuée.")

    # Colonnes utiles
    base   = [c for c in ["objet","timestamp","hand_label","pressure_class"] if c in df.columns]
    avail_lm  = [c for c in LM_COLS  if c in df.columns]
    avail_vel = [c for c in VEL_COLS if c in df.columns]
    for c in set(LM_COLS) - set(avail_lm):   df[c] = 0.0
    for c in set(VEL_COLS) - set(avail_vel): df[c] = 0.0
    df = df[base + LM_COLS + VEL_COLS]

    before = len(df)
    df = df.dropna()
    log.info("Lignes supprimées (NaN) : %d", before - len(df))

    df[LM_COLS + VEL_COLS] = df[LM_COLS + VEL_COLS].round(4)

    if "timestamp" in df.columns and "objet" in df.columns:
        b = len(df)
        df = df.drop_duplicates(subset=["objet","timestamp"])
        log.info("Doublons supprimés : %d", b - len(df))

    df = df.reset_index(drop=True)
    log.info("Nettoyé : %d lignes, %d colonnes", len(df), len(df.columns))
    log.info("Distribution classes :\n%s", df["pressure_class"].value_counts().sort_index().to_string())

    df.to_csv(output_csv, index=False)
    log.info("Sauvegardé -> %s", output_csv)
    return df

if __name__ == "__main__":
    clean()
