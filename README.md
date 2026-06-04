🤖 Autonomous Robot

DocumentationAn AI robotics project capable of interacting with its physical environment and performing adaptive grasping tasks.Project Architecturerobot_project/
├── config.py                  # ⚙️  Centralized configuration (paths, parameters)
├── main.py                    # 📷  Dataset capture (webcam + MediaPipe)
├── merge_csv.py               # 🔀  CSV merging by object
├── clean_dataset.py           # 🧹  Dataset cleaning and preparation
├── analyse.py                 # 📊  Statistical analysis + graphics
├── train_model.py             # 🧠  ML model training
├── test_model.py              # ✅  Model testing and validation
├── detection.py               # 👁️  Object detection via camera (OpenCV)
├── robot_autonome.py          # 🦾  Decision-making brain (FSM + ML)
├── simulate_robot_visual.py   # 🎮  Interactive visual simulation
├── bridge_gazebo.py           # 🌉  ROS to Gazebo bridge (robot commands)
├── pipeline.py                # 🚀  Full automated pipeline
└── logs/                      # 📁  Logs, graphics, reports
Recommended WorkflowStep 1 — Capture DataBashpython main.py
# Enter the object name (e.g., water_bottle)
# [8] increase pressure  [2] decrease pressure  [SPACE] pause  [Q] quit

Step 2 — Run the Full PipelineBashpython pipeline.py
This executes the following sequence: merge → clean → analyze → train.Step 3 — Test the ModelBashpython test_model.py --batch      # evaluation on the entire dataset
python test_model.py              # interactive test

Step 4 — Simulate the RobotBashpython simulate_robot_visual.py   # visual simulation
python robot_autonome.py          # decision-making simulation

Step 5 — Gazebo Integration (ROS)For full robot simulation in Gazebo featuring MLP control:Bash# Terminal 1: Launch Gazebo and the bridge
roslaunch robot_project robot_gazebo.launch

# Terminal 2 (optional): Publish sensor data
rostopic pub /sensor/landmarks std_msgs/Float64MultiArray "data: [...]" -r 10
For the complete installation and configuration guide: see GAZEBO_SETUP.mdDependenciesBashpip install opencv-python mediapipe scikit-learn pandas numpy matplotlib joblib

# For ROS/Gazebo integration:
sudo apt install ros-noetic-desktop-full  # or your ROS version
pip install rospy
Key Parameters (config.py)ParameterDefault ValueDescriptionCAMERA_INDEX0Webcam indexPRESSURE_MAX10Pressure scaleHIDDEN_LAYERS(128, 64, 32)MLP architectureGRIP_RULESdictPressure ranges per objectOBSTACLE_CRITICAL15 cmEmergency thresholdRoadmap[x] ROS/Gazebo integration (bridge_gazebo.py)[ ] YOLO integration for multi-class detection[ ] Web monitoring interface (FastAPI + React)[ ] Export to ROS2 for physical hardware[ ] Public dataset deployment on Hugging Face


License

Copyright (c) 2026 Vexyon / Youssef Miled.

This software is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

You are free to:

    Share: copy and redistribute the material in any medium or format.
    
    Adapt: remix, transform, and build upon the material.

Under the following terms:

    Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made.

    NonCommercial: You may not use the material for commercial purposes.

    ShareAlike: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
