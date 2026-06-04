from __future__ import annotations
import argparse
import logging
import os
import sys
import time

from config import LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def step(name: str, fn, *args, **kwargs):
    """Exécute une étape du pipeline avec timer et gestion d'erreur."""
    log.info("\n+----------------------------------------+")
    log.info("|  %-36s |", name)
    log.info("+----------------------------------------+")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        log.info("[OK] %s termine en %.1fs", name, time.time() - t0)
        return result
    except FileNotFoundError as e:
        log.error("Fichier manquant : %s", e)
        log.error("  -> Assurez-vous d'avoir exécuté les étapes précédentes.")
        sys.exit(1)
    except Exception as e:
        log.exception("Erreur dans '%s' : %s", name, e)
        sys.exit(1)


def run_pipeline(skip_merge: bool = False, only_train: bool = False) -> None:
    log.info("═" * 50)
    log.info("  PIPELINE ROBOT AUTONOME")
    log.info("═" * 50)

    if not only_train:
        if not skip_merge:
            from merge_csv import merge_all
            step("1. Fusion des CSV", merge_all)
        else:
            log.info("(Fusion ignorée - --skip-merge)")

        from clean_dataset import clean
        step("2. Nettoyage du dataset", clean)

        # Augmentation du dataset (optionnel mais recommande)
        from config import CLEAN_CSV
        aug_csv = CLEAN_CSV.replace("clean.csv", "augmented.csv")
        if not os.path.exists(aug_csv):
            log.info("Dataset augmente non trouve - lancement augmentation...")
            import subprocess
            result = subprocess.run([sys.executable, "Augment dataset.py", "--input", CLEAN_CSV, "--output", aug_csv, "--factor", "2"], capture_output=True, text=True)
            if result.returncode == 0:
                log.info("Augmentation terminee")
            else:
                log.warning("Erreur augmentation : %s", result.stderr)
        else:
            log.info("Dataset augmente deja present : %s", aug_csv)

        from analyse import analyse
        step("4. Analyse statistique", analyse)

    from train_model import train
    step("5. Entraînement du modèle", train)

    log.info("\n✅ Pipeline terminé avec succès !")
    log.info("   Modèle disponible -> pressure_model.pkl")
    log.info("   Logs & graphiques -> %s/", LOG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Robot Autonome")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Ignorer la fusion des CSV")
    parser.add_argument("--only-train", action="store_true",
                        help="Entraîner uniquement (dataset déjà prêt)")
    args = parser.parse_args()
    run_pipeline(skip_merge=args.skip_merge, only_train=args.only_train)
