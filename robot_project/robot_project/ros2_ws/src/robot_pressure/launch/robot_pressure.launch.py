"""
robot_pressure.launch.py - Lance le noeud ROS2 avec parametres

Usage :
  ros2 launch robot_pressure robot_pressure.launch.py
  ros2 launch robot_pressure robot_pressure.launch.py camera_index:=1
  ros2 launch robot_pressure robot_pressure.launch.py model_path:=/path/to/model.pkl
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    project_path = os.environ.get(
        "ROBOT_PROJECT_PATH",
        os.path.expanduser("~/robot_project")
    )

    return LaunchDescription([
        # Arguments configurables
        DeclareLaunchArgument("model_path",
            default_value=os.path.join(project_path, "pressure_model.pkl"),
            description="Chemin vers le modele pkl"),

        DeclareLaunchArgument("camera_index",
            default_value="0",
            description="Index camera OpenCV"),

        DeclareLaunchArgument("publish_rate",
            default_value="30.0",
            description="Frequence de publication en Hz"),

        DeclareLaunchArgument("stability_frames",
            default_value="8",
            description="Frames stables avant GRIP"),

        DeclareLaunchArgument("confidence_min",
            default_value="0.75",
            description="Seuil confiance minimum"),

        # Noeud principal
        Node(
            package="robot_pressure",
            executable="pressure_node",
            name="pressure_node",
            output="screen",
            parameters=[{
                "model_path":       LaunchConfiguration("model_path"),
                "camera_index":     LaunchConfiguration("camera_index"),
                "publish_rate":     LaunchConfiguration("publish_rate"),
                "stability_frames": LaunchConfiguration("stability_frames"),
                "confidence_min":   LaunchConfiguration("confidence_min"),
            }],
            remappings=[
                ("/robot/pressure_class",   "/robot/pressure_class"),
                ("/robot/grip_command",     "/robot/grip_command"),
                ("/robot/sensor_distances", "/robot/sensor_distances"),
            ],
        ),
    ])
