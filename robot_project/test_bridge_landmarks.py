#!/usr/bin/env python3
"""
test_bridge_landmarks.py - Utilitaire pour tester le bridge_gazebo.py

Publie des landmarks simulés ou lus depuis un CSV pour tester
l'intégration MLP → ROS → Gazebo sans avoir besoin d'une caméra réelle.

Utilisation:
    python test_bridge_landmarks.py --mode random           # Landmarks aléatoires
    python test_bridge_landmarks.py --mode dataset dataset.csv  # Depuis un CSV
    python test_bridge_landmarks.py --mode static landmarks.npy # Landmarks statiques
"""

import sys
import argparse
import logging
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class LandmarkPublisher:
    def __init__(self, topic='/sensor/landmarks', rate=10):
        """Initialise le publisher ROS."""
        rospy.init_node('test_landmarks_pub', anonymous=True)
        self.pub = rospy.Publisher(topic, Float64MultiArray, queue_size=10)
        self.rate = rospy.Rate(rate)
        log.info("Publisher initialisé sur %s (%.1f Hz)", topic, rate)

    def publish_random(self, n_samples=100):
        """Publie des landmarks aléatoires."""
        log.info("Publiant %d landmarks aléatoires...", n_samples)
        for i in range(n_samples):
            landmarks = np.random.randn(63).astype(np.float32)
            msg = Float64MultiArray()
            msg.data = landmarks.tolist()
            self.pub.publish(msg)
            
            if i % 10 == 0:
                log.debug("Sample %d/%d publié", i, n_samples)
            self.rate.sleep()

    def publish_from_csv(self, csv_path, repeat=False):
        """Publie des landmarks depuis un CSV."""
        if not os.path.exists(csv_path):
            log.error("Fichier CSV non trouvé: %s", csv_path)
            return

        try:
            df = pd.read_csv(csv_path)
            log.info("CSV chargé: %d lignes", len(df))
            
            # Extraire les colonnes de landmarks (lm0_x, lm0_y, lm0_z, ...)
            lm_cols = [c for c in df.columns if c.startswith('lm') and ('_x' in c or '_y' in c or '_z' in c)]
            lm_cols.sort()
            
            if len(lm_cols) < 63:
                log.warning("CSV contient %d colonnes, attendu 63 landmarks", len(lm_cols))
            
            log.info("Utilisant %d colonnes de landmarks", len(lm_cols))
            
            idx = 0
            while True:
                for _, row in df.iterrows():
                    try:
                        landmarks = row[lm_cols].fillna(0).values.astype(np.float32)
                        msg = Float64MultiArray()
                        msg.data = landmarks.tolist()
                        self.pub.publish(msg)
                        self.rate.sleep()
                    except Exception as e:
                        log.error("Erreur lors de la publication: %s", e)
                
                idx += 1
                if not repeat:
                    break
                log.info("Itération %d (boucle)", idx)
        
        except Exception as e:
            log.error("Erreur lors de la lecture du CSV: %s", e)

    def publish_from_npy(self, npy_path, repeat=False):
        """Publie des landmarks depuis un fichier .npy."""
        if not os.path.exists(npy_path):
            log.error("Fichier NPY non trouvé: %s", npy_path)
            return

        try:
            data = np.load(npy_path)
            log.info("NPY chargé: shape %s", data.shape)
            
            if len(data.shape) == 1:
                # Single landmark
                data = data.reshape(1, -1)
            
            idx = 0
            while True:
                for landmarks in data:
                    landmarks = landmarks.astype(np.float32)
                    if len(landmarks) < 63:
                        # Pad with zeros
                        landmarks = np.concatenate([landmarks, np.zeros(63 - len(landmarks))])
                    
                    msg = Float64MultiArray()
                    msg.data = landmarks[:63].tolist()  # Take first 63
                    self.pub.publish(msg)
                    self.rate.sleep()
                
                idx += 1
                if not repeat:
                    break
                log.info("Itération %d (boucle)", idx)
        
        except Exception as e:
            log.error("Erreur lors de la lecture du NPY: %s", e)

    def publish_static(self, landmarks_data):
        """Publie un seul set de landmarks statiques en boucle."""
        log.info("Publiant landmarks statiques...")
        while True:
            msg = Float64MultiArray()
            msg.data = landmarks_data
            self.pub.publish(msg)
            self.rate.sleep()


def main():
    parser = argparse.ArgumentParser(
        description="Publie des landmarks pour tester le bridge_gazebo.py"
    )
    parser.add_argument(
        '--mode',
        choices=['random', 'dataset', 'static', 'npy'],
        default='random',
        help='Mode de publication des landmarks'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Fichier d\'entrée (CSV pour dataset mode, NPY pour npy mode)'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='/sensor/landmarks',
        help='Topic ROS pour publier'
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=10.0,
        help='Fréquence de publication (Hz)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Nombre de samples (mode random uniquement)'
    )
    parser.add_argument(
        '--repeat',
        action='store_true',
        help='Boucler sur les données'
    )

    args = parser.parse_args()

    try:
        pub = LandmarkPublisher(topic=args.topic, rate=args.rate)

        if args.mode == 'random':
            pub.publish_random(n_samples=args.samples)
        
        elif args.mode == 'dataset':
            if not args.input:
                log.error("--input requis pour le mode dataset")
                sys.exit(1)
            pub.publish_from_csv(args.input, repeat=args.repeat)
        
        elif args.mode == 'npy':
            if not args.input:
                log.error("--input requis pour le mode npy")
                sys.exit(1)
            pub.publish_from_npy(args.input, repeat=args.repeat)
        
        elif args.mode == 'static':
            # Landmarks statiques neutre (zéros)
            landmarks = np.zeros(63, dtype=np.float32).tolist()
            pub.publish_static(landmarks)

    except rospy.ROSInterruptException:
        log.info("Arrêt par l'utilisateur")


if __name__ == '__main__':
    main()
