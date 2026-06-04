"""
augment_dataset.py - Augmentation du dataset existant
Multiplie les frames sans recapturer avec la webcam.

Strategies :
  1. Bruit gaussien    - simule instabilite capteur
  2. Scale             - main plus proche / plus loin
  3. Translation       - main decalee dans l'image
  4. Mirror            - simule la main opposee
  5. Rotation legere   - legere inclinaison de la main

Usage :
  python augment_dataset.py              # augmente dataset_clean.csv
  python augment_dataset.py --factor 5   # multiplie par 5
  python augment_dataset.py --class 0    # augmente seulement la classe Vide
"""

import os, argparse, logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Colonnes
LM_COLS  = [f"lm{i}_{a}" for i in range(21) for a in ("x","y","z")]
VEL_COLS = [f"lm{i}_v{a}" for i in range(21) for a in ("x","y","z")]
CLASS_NAMES = {0: "Vide", 1: "Faible", 2: "Optimal", 3: "Trop fort"}

# Connexions anatomiques de la main (pour la rotation)
# 21 landmarks MediaPipe
N_LM = 21


def augment_noise(df: pd.DataFrame, std: float = 0.004) -> pd.DataFrame:
    """Bruit gaussien sur positions et vitesses."""
    result = df.copy()
    cols = [c for c in LM_COLS + VEL_COLS if c in df.columns]
    noise = np.random.normal(0, std, (len(df), len(cols)))
    result[cols] = result[cols].values + noise
    result["aug_method"] = "noise"
    return result


def augment_scale(df: pd.DataFrame, scale_min: float = 0.85, scale_max: float = 1.15) -> pd.DataFrame:
    """Scale aleatoire - simule distance a la camera."""
    result = df.copy()
    lm_cols = [c for c in LM_COLS if c in df.columns]
    scales = np.random.uniform(scale_min, scale_max, (len(df), 1))
    result[lm_cols] = result[lm_cols].values * scales
    result["aug_method"] = "scale"
    return result


def augment_translate(df: pd.DataFrame, tx_range: float = 0.06, ty_range: float = 0.06) -> pd.DataFrame:
    """Translation legere - main decalee dans l'image."""
    result = df.copy()
    x_cols = [f"lm{i}_x" for i in range(N_LM) if f"lm{i}_x" in df.columns]
    y_cols = [f"lm{i}_y" for i in range(N_LM) if f"lm{i}_y" in df.columns]
    tx = np.random.uniform(-tx_range, tx_range, (len(df), 1))
    ty = np.random.uniform(-ty_range, ty_range, (len(df), 1))
    result[x_cols] = result[x_cols].values + tx
    result[y_cols] = result[y_cols].values + ty
    result["aug_method"] = "translate"
    return result


def augment_mirror(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror horizontal - simule la main opposee.
    Inverse les coordonnees x (la main gauche devient droite et vice versa).
    """
    result = df.copy()
    x_cols = [f"lm{i}_x" for i in range(N_LM) if f"lm{i}_x" in df.columns]
    vx_cols = [f"lm{i}_vx" for i in range(N_LM) if f"lm{i}_vx" in df.columns]
    result[x_cols] = -result[x_cols].values
    if vx_cols:
        result[vx_cols] = -result[vx_cols].values
    if "hand_label" in result.columns:
        result["hand_label"] = result["hand_label"].map(
            {"Right": "Left", "Left": "Right"}
        ).fillna(result["hand_label"])
    result["aug_method"] = "mirror"
    return result


def augment_rotation(df: pd.DataFrame, angle_range: float = 15.0) -> pd.DataFrame:
    """
    Rotation 2D legere dans le plan xy.
    Simule une inclinaison de la main.
    """
    result = df.copy()
    angles = np.random.uniform(-angle_range, angle_range, len(df))
    angles_rad = np.radians(angles)
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    for i in range(N_LM):
        xc = f"lm{i}_x"
        yc = f"lm{i}_y"
        if xc not in df.columns or yc not in df.columns:
            continue
        x = result[xc].values.copy()
        y = result[yc].values.copy()
        result[xc] = cos_a * x - sin_a * y
        result[yc] = sin_a * x + cos_a * y

    result["aug_method"] = "rotation"
    return result


def augment_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Combine bruit + scale + translation pour max diversite."""
    result = augment_noise(df, std=0.003)
    result = augment_scale(result, 0.88, 1.12)
    result = augment_translate(result, 0.05, 0.05)
    result["aug_method"] = "combined"
    return result


# Toutes les strategies disponibles
STRATEGIES = {
    "noise":    augment_noise,
    "scale":    augment_scale,
    "translate": augment_translate,
    "mirror":   augment_mirror,
    "rotation": augment_rotation,
    "combined": augment_combined,
}


def print_distribution(df: pd.DataFrame, label: str) -> None:
    if "pressure_class" not in df.columns:
        return
    total = len(df)
    print(f"\n  {label} ({total:,} frames) :")
    for cls in sorted(df["pressure_class"].unique()):
        count = (df["pressure_class"] == cls).sum()
        pct   = count / total * 100
        bar   = "#" * int(pct / 2)
        name  = CLASS_NAMES.get(int(cls), str(cls))
        print(f"    Classe {cls} ({name:<10}) : {count:>6,} ({pct:5.1f}%)  {bar}")


def augment_dataset(
    input_csv:  str,
    output_csv: str,
    factor:     int = 3,
    target_class: int = None,
    strategies: list = None,
) -> pd.DataFrame:
    """
    Augmente le dataset.

    Args:
        input_csv    : chemin du CSV source
        output_csv   : chemin de sortie
        factor       : nombre de copies augmentees a creer
        target_class : si specifie, augmente seulement cette classe
        strategies   : liste des strategies a utiliser
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Dataset introuvable : {input_csv}")

    df = pd.read_csv(input_csv)
    log.info("Dataset charge : %d lignes", len(df))
    print_distribution(df, "Avant augmentation")

    # Ajouter colonne methode si absente
    if "aug_method" not in df.columns:
        df["aug_method"] = "original"

    # Selectionner les lignes a augmenter
    if target_class is not None:
        to_augment = df[df["pressure_class"] == target_class].copy()
        log.info("Augmentation ciblee : classe %d (%s) - %d lignes",
                 target_class, CLASS_NAMES.get(target_class, "?"), len(to_augment))
    else:
        to_augment = df.copy()

    if strategies is None:
        strategies = list(STRATEGIES.keys())

    # Generer les copies augmentees (parallele)
    augmented_parts = [df]  # on garde toujours l'original

    def apply_strategy(i):
        strat_name = strategies[i % len(strategies)]
        strat_fn = STRATEGIES[strat_name]
        aug = strat_fn(to_augment.copy())
        log.info("  Copie %d/%d : strategie '%s' -> %d lignes", i+1, factor, strat_name, len(aug))
        return aug

    if factor > 0:
        augmented_parts.extend(Parallel(n_jobs=-1, verbose=0)(delayed(apply_strategy)(i) for i in range(factor)))

    # Fusion et nettoyage
    result = pd.concat(augmented_parts, ignore_index=True)

    # Arrondir les landmarks
    lm_cols = [c for c in LM_COLS + VEL_COLS if c in result.columns]
    result[lm_cols] = result[lm_cols].round(4)

    print_distribution(result, "Apres augmentation")
    result.to_csv(output_csv, index=False)
    log.info("Dataset augmente sauvegarde -> %s", output_csv)
    log.info("Lignes : %d -> %d (x%.1f)", len(df), len(result), len(result)/len(df))

    return result


def balance_classes(
    input_csv:  str,
    output_csv: str,
    target_per_class: int = 10000,
) -> pd.DataFrame:
    """
    Equilibre les classes en augmentant les classes sous-representees.
    Chaque classe atteint target_per_class frames.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Dataset introuvable : {input_csv}")

    df = pd.read_csv(input_csv)
    log.info("Equilibrage des classes vers %d frames chacune", target_per_class)
    print_distribution(df, "Avant equilibrage")

    if "aug_method" not in df.columns:
        df["aug_method"] = "original"

    parts = [df]

    for cls in sorted(df["pressure_class"].unique()):
        subset = df[df["pressure_class"] == cls]
        current = len(subset)
        needed  = target_per_class - current

        if needed <= 0:
            log.info("  Classe %d (%s) : %d frames - OK pas besoin d'augmenter",
                     cls, CLASS_NAMES.get(int(cls), "?"), current)
            continue

        log.info("  Classe %d (%s) : %d -> %d (+%d frames a generer)",
                 cls, CLASS_NAMES.get(int(cls), "?"), current, target_per_class, needed)

        # Generer les frames manquantes par repetition avec augmentation
        generated = []
        strategies_cycle = list(STRATEGIES.keys())

        while len(generated) < needed:
            strat_name = strategies_cycle[len(generated) % len(strategies_cycle)]
            strat_fn   = STRATEGIES[strat_name]

            # Echantillonner des lignes aleatoires de la classe
            sample_size = min(len(subset), needed - len(generated))
            sample = subset.sample(n=sample_size, replace=True).copy()
            aug    = strat_fn(sample)
            generated.append(aug)

        gen_df = pd.concat(generated, ignore_index=True).head(needed)
        parts.append(gen_df)

    result = pd.concat(parts, ignore_index=True)
    lm_cols = [c for c in LM_COLS + VEL_COLS if c in result.columns]
    result[lm_cols] = result[lm_cols].round(4)

    print_distribution(result, "Apres equilibrage")
    result.to_csv(output_csv, index=False)
    log.info("Dataset equilibre sauvegarde -> %s", output_csv)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentation du dataset")
    parser.add_argument("--input",   default="dataset_clean.csv", help="CSV source")
    parser.add_argument("--output",  default="dataset_augmented.csv", help="CSV sortie")
    parser.add_argument("--factor",  type=int, default=3, help="Nombre de copies (defaut: 3)")
    parser.add_argument("--class",   type=int, dest="target_class", default=None,
                        help="Augmenter seulement cette classe (0/1/2/3)")
    parser.add_argument("--balance", action="store_true",
                        help="Equilibrer toutes les classes a 10k frames")
    parser.add_argument("--target",  type=int, default=10000,
                        help="Frames cibles par classe avec --balance (defaut: 10000)")
    parser.add_argument("--strategy", nargs="+",
                        choices=list(STRATEGIES.keys()),
                        help="Strategies a utiliser (defaut: toutes)")
    args = parser.parse_args()

    if args.balance:
        balance_classes(args.input, args.output, target_per_class=args.target)
    else:
        augment_dataset(
            input_csv     = args.input,
            output_csv    = args.output,
            factor        = args.factor,
            target_class  = args.target_class,
            strategies    = args.strategy,
        )