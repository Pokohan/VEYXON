from setuptools import setup
import os
from glob import glob

package_name = "robot_pressure"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Robot Autonome Project",
    maintainer_email="contact@robot-autonome.fr",
    description="Noeud ROS2 pour classification de pression de prehension",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pressure_node = robot_pressure.pressure_node:main",
            "mock_robot_node = robot_pressure.mock_robot_node:main",
        ],
    },
)
