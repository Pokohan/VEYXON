"""
pressure_node.py - Noeud ROS2 pour la classification de pression

Publie sur :
  /robot/pressure_class   (std_msgs/Int32)        : classe 0/1/2/3
  /robot/pressure_conf    (std_msgs/Float32)       : confiance 0.0-1.0
  /robot/pressure_state   (std_msgs/String)        : etat FSM
  /robot/grip_command     (std_msgs/String)        : commande grip/hold/release/stop
  /robot/landmarks        (geometry_msgs/PoseArray): 21 landmarks main

S'abonne a :
  /robot/sensor_distances (std_msgs/Float32MultiArray) : [front, left, right] en cm

Usage :
  ros2 run robot_pressure pressure_node
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32, String, Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import collections
import os
import sys

# Ajouter le projet au path
PROJECT_PATH = os.environ.get("ROBOT_PROJECT_PATH", os.path.expanduser("~/robot_project"))
sys.path.insert(0, PROJECT_PATH)

try:
    from config import (
        MODEL_PATH, CAMERA_INDEX, MAX_HANDS,
        MIN_DETECTION_CONF, MIN_TRACKING_CONF,
        STABILITY_FRAMES, GRACE_PERIOD_FRAMES,
        CONFIDENCE_MIN, PRESSURE_CLASSES,
        OBSTACLE_CRITICAL, OBSTACLE_WARN,
    )
except ImportError:
    # Valeurs par defaut si config.py absent
    MODEL_PATH         = os.path.join(PROJECT_PATH, "pressure_model.pkl")
    CAMERA_INDEX       = 0
    MAX_HANDS          = 1
    MIN_DETECTION_CONF = 0.75
    MIN_TRACKING_CONF  = 0.75
    STABILITY_FRAMES   = 8
    GRACE_PERIOD_FRAMES= 15
    CONFIDENCE_MIN     = 0.75
    PRESSURE_CLASSES   = {0:"Vide", 1:"Faible", 2:"Optimal", 3:"Trop fort"}
    OBSTACLE_CRITICAL  = 15
    OBSTACLE_WARN      = 30

LM_COLS  = [f"lm{i}_{a}" for i in range(21) for a in ("x","y","z")]
VEL_COLS = [f"lm{i}_v{a}" for i in range(21) for a in ("x","y","z")]


class PressureNode(Node):
    def __init__(self):
        super().__init__("pressure_node")

        # ── Parametres ROS2 (configurables via launch file) ───────────────────
        self.declare_parameter("model_path",     MODEL_PATH)
        self.declare_parameter("camera_index",   CAMERA_INDEX)
        self.declare_parameter("publish_rate",   30.0)
        self.declare_parameter("stability_frames", STABILITY_FRAMES)
        self.declare_parameter("confidence_min", CONFIDENCE_MIN)

        model_path      = self.get_parameter("model_path").value
        camera_idx      = self.get_parameter("camera_index").value
        publish_rate    = self.get_parameter("publish_rate").value
        self.stab_frames= self.get_parameter("stability_frames").value
        self.conf_min   = self.get_parameter("confidence_min").value

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_class    = self.create_publisher(Int32,             "/robot/pressure_class",  10)
        self.pub_conf     = self.create_publisher(Float32,           "/robot/pressure_conf",   10)
        self.pub_state    = self.create_publisher(String,            "/robot/pressure_state",  10)
        self.pub_command  = self.create_publisher(String,            "/robot/grip_command",    10)
        self.pub_landmarks= self.create_publisher(PoseArray,         "/robot/landmarks",       10)

        # ── Subscriber distances capteurs ─────────────────────────────────────
        self.distances = [100.0, 100.0, 100.0]  # front, left, right
        self.sub_dist  = self.create_subscription(
            Float32MultiArray, "/robot/sensor_distances",
            self._cb_distances, 10
        )

        # ── Modele ML ─────────────────────────────────────────────────────────
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.get_logger().info(f"Modele charge : {type(self.model).__name__}")
        else:
            self.get_logger().error(f"Modele introuvable : {model_path}")

        # ── MediaPipe ─────────────────────────────────────────────────────────
        self.mp_hands   = mp.solutions.hands
        self.detector   = self.mp_hands.Hands(
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF,
        )

        # ── Camera ────────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            self.get_logger().error(f"Camera {camera_idx} introuvable")

        # ── Etat interne ──────────────────────────────────────────────────────
        self.pclass       = 0
        self.conf         = 0.0
        self.fsm_state    = "IDLE"
        self.prev_pos     = None
        self.prev_t       = time.time()
        self.grace_count  = 0
        self.stability_w  = collections.deque(maxlen=self.stab_frames)

        # ── Timer principal ───────────────────────────────────────────────────
        period = 1.0 / publish_rate
        self.timer = self.create_timer(period, self._process_frame)
        self.get_logger().info(f"Noeud pressure_node demarre ({publish_rate:.0f} Hz)")

    def _cb_distances(self, msg: Float32MultiArray):
        """Recoit les distances des capteurs [front, left, right]."""
        if len(msg.data) >= 3:
            self.distances = list(msg.data[:3])

    def _get_features(self, landmarks):
        """Extrait 126 features depuis les landmarks MediaPipe."""
        bx = landmarks.landmark[0].x
        by = landmarks.landmark[0].y
        bz = landmarks.landmark[0].z

        pos = np.array([
            [lm.x - bx, lm.y - by, lm.z - bz]
            for lm in landmarks.landmark
        ])

        now = time.time()
        dt  = max(now - self.prev_t, 1e-6)
        self.prev_t = now

        vel = (pos - self.prev_pos) / dt if self.prev_pos is not None else np.zeros_like(pos)
        self.prev_pos = pos.copy()

        return np.concatenate([pos.flatten(), vel.flatten()])

    def _predict(self, features: np.ndarray):
        """Prediction avec gestion de l'incertitude."""
        if self.model is None:
            return 0, 0.0

        X = features.reshape(1, -1)
        try:
            if hasattr(self.model, "predict_proba"):
                proba  = self.model.predict_proba(X)[0]
                pclass = int(np.argmax(proba))
                conf   = float(proba[pclass])
            else:
                pclass = int(self.model.predict(X)[0])
                conf   = 1.0
            return pclass, conf
        except Exception as e:
            self.get_logger().warn(f"Erreur prediction : {e}")
            return 0, 0.0

    def _update_fsm(self, pclass: int, conf: float, hand_detected: bool):
        """Machine a etats simplifiee compatible ROS2."""
        df, dl, dr = self.distances

        # Urgence obstacle
        if df < OBSTACLE_CRITICAL and dl < OBSTACLE_CRITICAL and dr < OBSTACLE_CRITICAL:
            self.fsm_state = "EMERGENCY"
            return "stop"

        # Grace period
        if not hand_detected:
            self.grace_count += 1
            if self.grace_count > GRACE_PERIOD_FRAMES:
                self.fsm_state = "IDLE"
                self.stability_w.clear()
                return "stop"
            return "hold"  # maintien pendant grace period
        else:
            self.grace_count = 0

        # Confiance insuffisante
        if conf < self.conf_min:
            return "hold"

        # Dead zone
        self.stability_w.append(pclass)
        is_stable = (len(self.stability_w) >= self.stab_frames and
                     np.std(list(self.stability_w)) == 0)

        if not is_stable:
            self.fsm_state = "APPROACH"
            return "hold_position"

        # Transitions selon classe
        if pclass == 0:
            self.fsm_state = "IDLE"
            return "release"
        elif pclass == 1:
            self.fsm_state = "APPROACH"
            return "grip_light"
        elif pclass == 2:
            self.fsm_state = "HOLD"
            return "grip"
        elif pclass == 3:
            self.fsm_state = "EMERGENCY"
            return "release"

        return "hold"

    def _publish_landmarks(self, landmarks):
        """Publie les 21 landmarks comme PoseArray."""
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"

        for lm in landmarks.landmark:
            pose = Pose()
            pose.position.x = float(lm.x)
            pose.position.y = float(lm.y)
            pose.position.z = float(lm.z)
            msg.poses.append(pose)

        self.pub_landmarks.publish(msg)

    def _process_frame(self):
        """Callback principal - traite une frame et publie les resultats."""
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame   = cv2.flip(frame, 1)
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for landmarks, _ in zip(results.multi_hand_landmarks, results.multi_handedness):
                features = self._get_features(landmarks)
                self.pclass, self.conf = self._predict(features)
                self._publish_landmarks(landmarks)

        # FSM et commande
        command = self._update_fsm(self.pclass, self.conf, hand_detected)

        # Publication
        msg_class = Int32()
        msg_class.data = self.pclass
        self.pub_class.publish(msg_class)

        msg_conf = Float32()
        msg_conf.data = float(self.conf)
        self.pub_conf.publish(msg_conf)

        msg_state = String()
        msg_state.data = self.fsm_state
        self.pub_state.publish(msg_state)

        msg_cmd = String()
        msg_cmd.data = command
        self.pub_command.publish(msg_cmd)

    def destroy_node(self):
        self.cap.release()
        self.detector.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PressureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
