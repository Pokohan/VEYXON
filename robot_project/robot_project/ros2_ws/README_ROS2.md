# ROS2 - Robot Pressure Node

## Installation

### 1. Prerequis
```bash
# ROS2 Humble ou Iron (Ubuntu 22.04 recommande)
sudo apt install ros-humble-desktop python3-colcon-common-extensions

# Dependances Python
pip install opencv-python mediapipe scikit-learn joblib numpy
```

### 2. Build du package
```bash
# Definir le chemin du projet
export ROBOT_PROJECT_PATH=/chemin/vers/robot_project

# Build
cd ros2_ws
colcon build --packages-select robot_pressure
source install/setup.bash
```

### 3. Lancer le noeud
```bash
# Lancement simple
ros2 launch robot_pressure robot_pressure.launch.py

# Avec parametres custom
ros2 launch robot_pressure robot_pressure.launch.py \
  model_path:=/chemin/vers/pressure_model.pkl \
  camera_index:=0 \
  publish_rate:=30.0

# Avec le mock robot (test sans vrai bras)
ros2 run robot_pressure mock_robot_node &
ros2 launch robot_pressure robot_pressure.launch.py
```

## Topics ROS2

### Publie
| Topic | Type | Description |
|-------|------|-------------|
| /robot/pressure_class | std_msgs/Int32 | Classe 0=Vide 1=Faible 2=Optimal 3=Trop fort |
| /robot/pressure_conf | std_msgs/Float32 | Confiance 0.0-1.0 |
| /robot/pressure_state | std_msgs/String | Etat FSM |
| /robot/grip_command | std_msgs/String | grip/hold/release/stop |
| /robot/landmarks | geometry_msgs/PoseArray | 21 landmarks main |

### S'abonne
| Topic | Type | Description |
|-------|------|-------------|
| /robot/sensor_distances | std_msgs/Float32MultiArray | [front, left, right] en cm |

## Visualisation
```bash
# Voir les commandes en temps reel
ros2 topic echo /robot/grip_command

# Voir la classe predite
ros2 topic echo /robot/pressure_class

# Voir tous les topics
ros2 topic list

# Frequence de publication
ros2 topic hz /robot/pressure_class

# rqt pour visualisation graphique
rqt
```

## Connexion a un vrai bras robotique
Pour connecter a un bras (Universal Robots, Franka, etc.) :
```bash
# S'abonner a /robot/grip_command dans ton noeud de controle
# et mapper les commandes vers les joints du bras :
#   grip       -> fermer les doigts a position Optimal
#   grip_light -> fermer les doigts a position Faible
#   release    -> ouvrir les doigts
#   stop       -> stop d'urgence
```
