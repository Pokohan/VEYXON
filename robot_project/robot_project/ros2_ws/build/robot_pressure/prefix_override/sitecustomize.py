import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/hpz/STOCKAGE2/Projet/robot_project/robot_project/robot_project/ros2_ws/install/robot_pressure'
