"""
mock_robot_node.py - Simule un bras robotique qui recoit les commandes ROS2

S'abonne a :
  /robot/grip_command     : commande grip/hold/release/stop
  /robot/pressure_class   : classe 0/1/2/3
  /robot/pressure_state   : etat FSM

Publie sur :
  /robot/sensor_distances : distances simulees [front, left, right]
  /robot/motor_command    : commande moteur (position doigts 0-100%)

Usage :
  ros2 run robot_pressure mock_robot_node
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String, Float32MultiArray, Float32
import math
import time


CLASS_NAMES = {0: "Vide", 1: "Faible", 2: "Optimal", 3: "Trop fort"}
MOTOR_POSITIONS = {
    "release":      0.0,
    "grip_light":  30.0,
    "grip":        60.0,
    "hold":        60.0,
    "hold_position": 50.0,
    "stop":         0.0,
}


class MockRobotNode(Node):
    def __init__(self):
        super().__init__("mock_robot_node")

        # Publishers
        self.pub_distances = self.create_publisher(
            Float32MultiArray, "/robot/sensor_distances", 10)
        self.pub_motor = self.create_publisher(
            Float32, "/robot/motor_command", 10)

        # Subscribers
        self.create_subscription(String,  "/robot/grip_command",   self._cb_command, 10)
        self.create_subscription(Int32,   "/robot/pressure_class", self._cb_class,   10)
        self.create_subscription(String,  "/robot/pressure_state", self._cb_state,   10)

        # Etat simule
        self.motor_pos    = 0.0
        self.fsm_state    = "IDLE"
        self.pclass       = 0
        self.t_start      = time.time()

        # Timer simulation distances (obstacle simule)
        self.create_timer(0.1,  self._publish_distances)
        self.create_timer(0.05, self._publish_motor)

        self.get_logger().info("Mock robot demarre - simulation active")

    def _cb_command(self, msg: String):
        cmd = msg.data
        target = MOTOR_POSITIONS.get(cmd, self.motor_pos)
        # Rampe douce vers la position cible
        self.motor_pos += (target - self.motor_pos) * 0.3
        self.get_logger().info(
            f"Commande: {cmd:15s} -> Moteur: {self.motor_pos:.1f}%"
        )

    def _cb_class(self, msg: Int32):
        self.pclass = msg.data

    def _cb_state(self, msg: String):
        if msg.data != self.fsm_state:
            self.fsm_state = msg.data
            self.get_logger().info(f"FSM: {self.fsm_state}")

    def _publish_distances(self):
        """Simule des distances de capteurs variables."""
        t = time.time() - self.t_start
        # Simule un objet qui s'approche puis s'eloigne
        dist_front = 50.0 + 40.0 * math.sin(t * 0.3)
        dist_front = max(5.0, dist_front)

        msg = Float32MultiArray()
        msg.data = [dist_front, 80.0, 80.0]
        self.pub_distances.publish(msg)

    def _publish_motor(self):
        msg = Float32()
        msg.data = float(self.motor_pos)
        self.pub_motor.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
