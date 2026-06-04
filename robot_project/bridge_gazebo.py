#!/usr/bin/env python3
"""
bridge_gazebo.py - Pont entre le modèle MLP et les commandes Gazebo via ROS

Ce script agit comme un nœud ROS qui :
1. Charge le modèle MLP entraîné pour la classification de pression.
2. S'abonne à un topic ROS fournissant les données de capteurs (landmarks de la main).
3. Exécute l'inférence du modèle pour prédire la classe de pression.
4. Publie des commandes de préhension vers les contrôleurs ROS pour Gazebo.

Topics et Actions :
- Subscriber: /sensor/landmarks (std_msgs/Float64MultiArray pour les landmarks)
- Subscriber: /robot_arm/joint_states (sensor_msgs/JointState pour l'état du robot)
- Publisher: /gripper/command (std_msgs/Float64 pour la force brute)
- Action: /robot_arm/gripper_controller/follow_joint_trajectory (pour trajectoires)
- Action: /robot_arm/arm_controller/follow_joint_trajectory (pour le bras)

Utilisation :
- roslaunch robot_project robot_gazebo.launch
  (cela lancera Gazebo et ce bridge)
"""

import os
import logging
import numpy as np
import joblib
import rospy
import actionlib
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from .config import MODEL_PATH, GRIP_RULES, PRESSURE_CLASSES, CONFIDENCE_MIN

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

class GazeboBridge:
    def __init__(self):
        # Initialiser ROS
        rospy.init_node('gazebo_bridge', anonymous=True)

        # Chemin du modèle fourni par le launch file ou via config
        self.model_path = rospy.get_param('~model_path', MODEL_PATH)
        self.model = None
        self.load_model()

        # Contexte de l'objet actuel
        self.current_object = "bouteille"  # Par défaut
        
        # État du robot
        self.robot_state = None
        self.current_gripper_force = 0.0

        # Subscriber pour les données de capteurs (landmarks)
        self.sensor_sub = rospy.Subscriber(
            '/sensor/landmarks', Float64MultiArray, self.sensor_callback)

        # Subscriber pour l'état du robot
        self.joint_state_sub = rospy.Subscriber(
            '/robot_arm/joint_states', JointState, self.joint_state_callback)

        # Publisher pour les commandes brutes de gripper (fallback)
        self.grip_pub = rospy.Publisher(
            '/robot_arm/gripper_controller/command', Float64, queue_size=10)

        # Action clients pour les trajectoires
        self.gripper_action_client = actionlib.SimpleActionClient(
            '/robot_arm/gripper_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        
        self.arm_action_client = actionlib.SimpleActionClient(
            '/robot_arm/arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)

        # Attendre la connexion aux actions
        rospy.sleep(1)
        self.gripper_action_available = False
        self.arm_action_available = False

        try:
            self.gripper_action_available = self.gripper_action_client.wait_for_server(timeout=rospy.Duration(5))
            if self.gripper_action_available:
                log.info("Connecté au serveur d'action gripper")
            else:
                log.warning("Serveur gripper non disponible après attente")
        except rospy.ROSException as e:
            log.warning("Serveur gripper non disponible: %s", e)

        try:
            self.arm_action_available = self.arm_action_client.wait_for_server(timeout=rospy.Duration(5))
            if self.arm_action_available:
                log.info("Connecté au serveur d'action arm")
            else:
                log.warning("Serveur arm non disponible après attente")
        except rospy.ROSException as e:
            log.warning("Serveur arm non disponible: %s", e)

        log.info("Pont Gazebo initialisé. Modèle chargé: %s", self.model is not None)

    def load_model(self):
        """Charge le modèle MLP depuis le fichier."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                log.info("Modèle chargé depuis %s", self.model_path)
            except Exception as e:
                log.error("Erreur lors du chargement du modèle: %s", e)
                self.model = None
        else:
            log.error("Modèle non trouvé à %s", self.model_path)
            self.model = None

        if self.model is None:
            rospy.signal_shutdown("Modèle manquant ou corrompu")

    def joint_state_callback(self, msg):
        """Callback pour l'état des joints du robot."""
        self.robot_state = msg
        
    def sensor_callback(self, msg):
        """Callback pour les données de capteurs (landmarks de la main)."""
        if self.model is None:
            log.warning("Modèle non chargé, impossible de prédire.")
            return

        if not msg.data or len(msg.data) == 0:
            log.error("Landmarks reçu vide ou invalide")
            return

        # Convertir le message en array numpy
        try:
            landmarks = np.array(msg.data)
            landmarks = landmarks.reshape(1, -1)
        except Exception as e:
            log.error("Erreur lors de la conversion des landmarks: %s", e)
            return

        except Exception as e:
            log.error("Erreur lors de la conversion des landmarks: %s", e)
            return

        # Prédire la classe de pression
        try:
            proba = self.model.predict_proba(landmarks)[0]
            confidence = np.max(proba)
            predicted_class = np.argmax(proba)

            if confidence < CONFIDENCE_MIN:
                log.warning("Confiance faible (%.2f), classe incertaine. Arrêt du robot.", 
                           confidence)
                self.current_gripper_force = 0.0
                self.send_gripper_command(0.0)
            else:
                # Mapper la classe à une force de préhension
                grip_force = self.map_pressure_to_grip(predicted_class, self.current_object)
                self.current_gripper_force = grip_force
                
                log.info("Classe prédite: %d (%s), Confiance: %.2f, Force: %.2f",
                         predicted_class, 
                         PRESSURE_CLASSES.get(predicted_class, "?"),
                         confidence, 
                         grip_force)

                # Envoyer la commande au gripper
                self.send_gripper_command(grip_force)

        except Exception as e:
            log.error("Erreur lors de la prédiction: %s", e)

    def send_gripper_command(self, grip_force):
        """Envoie une commande de force au gripper."""
        # Convertir la force (0-10) en position des doigts (0 à -0.05)
        # Plus la force est grande, plus les doigts se ferment
        finger_position = -(grip_force / 10.0) * 0.05
        
        try:
            # Méthode 1: Publier directement la force (pour les contrôleurs d'effort)
            self.grip_pub.publish(Float64(grip_force))
            
            # Méthode 2: Envoyer une trajectoire (plus stable)
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = JointTrajectory()
            goal.trajectory.joint_names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
            
            # Point de trajectoire
            point = JointTrajectoryPoint()
            point.positions = [finger_position, finger_position]
            point.velocities = [0, 0]
            point.time_from_start = rospy.Duration(0.5)
            
            goal.trajectory.points.append(point)
            
            # Envoyer l'action si le serveur est déjà disponible
            if self.gripper_action_available:
                self.gripper_action_client.send_goal_and_wait(goal, timeout=rospy.Duration(2))
            else:
                log.debug("Action gripper indisponible, commande brute publiée uniquement.")
        except Exception as e:
            log.error("Erreur lors de l'envoi de la commande gripper: %s", e)

    def map_pressure_to_grip(self, pressure_class, object_name):
        """Mappe la classe de pression à une force de préhension."""
        if object_name in GRIP_RULES:
            min_p, max_p = GRIP_RULES[object_name]
            # PRESSURE_CLASSES = {0: "Vide", 1: "Faible", 2: "Optimal", 3: "Trop fort"}
            if pressure_class == 0:
                return 0.0  # Vide
            elif pressure_class == 1:
                return min_p  # Faible
            elif pressure_class == 2:
                return (min_p + max_p) / 2  # Optimal
            elif pressure_class == 3:
                return max_p  # Trop fort
        else:
            log.warning("Objet inconnu: %s, utilisant force par défaut", object_name)
            return 5.0  # Valeur par défaut

        return 0.0

if __name__ == '__main__':
    try:
        bridge = GazeboBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
