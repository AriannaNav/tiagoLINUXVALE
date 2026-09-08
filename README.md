# TIAGo + Booster T1 Simulation Stack

Integrated simulation environment for heterogeneous robots: **TIAGo** (Gazebo/ROS2) and **Booster T1** (Webots), orchestrated by **Circus** (MuJoCo) via **SimBridge** (ROS2 bridge).

## Clone the Repository

This repository uses Git submodules for `circus` and `simbridge`. Clone with:

```bash
https://github.com/Lab-RoCoCo-Sapienza/hrai-25-26-course-project-HRAI-Container
cd hrai_container
```

If you've already cloned without submodules, initialize them:

```bash
git submodule update --init --recursive
```

## Requirements

### For Docker (TIAGo/Booster Webots)
- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- X11 display

### For Circus + SimBridge
- **pixi** — [install from pixi.sh](https://pixi.sh)

## Pixi Installation

Pixi is a cross-platform package manager (conda-based). Install the version pixi 0.59.0 from 

```bash
https://pixi.prefix.dev/latest/installation/#download-from-github-releases
```

## Installation instructions
**Build the Docker image:**
```bash
cd dockerfiles
docker build -t spqr:booster .
```

**Run TIAGo:**
```bash
bash start_tiago.sh
```

Inside the container, launch Gazebo:
```bash
ros2 launch tiago_gazebo tiago_gazebo.launch.py is_public_sim:=True
```
Check if Tiago spawn in gazebo to see if it works.


## Circus + SimBridge (Robot Booster T1 Integration)

Circus is the main simulator that manages Docker containers and physics (MuJoCo). SimBridge bridges ROS2 to Circus for sensor/actuator communication.

### Install and run Circus

```bash
cd circus
pixi install
```

The simulator will start and wait for robot containers to connect via Docker API

### Install SimBridge

SimBridge runs automatically inside robot containers created by Circus. To install standalone dependencies:

```bash
cd simbridge
pixi install
```

when all the repos are built you can run. Modify first the yaml file in `circus/resources/config/path_constants.yaml` with the absolute paths on your machine:

```yaml
circus: /absolute/path/to/circus
simbridge: /absolute/path/to/simbridge
booster_robotics_sdk: /absolute/path/to/dockerfiles/booster_robotics_sdk
exchange: /absolute/path/to/exchange
```

- `circus` and `simbridge` are in the root of this repository
- `booster_robotics_sdk` is inside the `dockerfiles/` directory
- `exchange` is in the root of this repository — **this is the folder where students put their ROS2 code** (it gets mounted as `/app/exchange` inside the robot container)

### Student code: the `exchange` folder

The `exchange/` directory is mounted inside each robot container at `/app/exchange`. Students should place their ROS2 nodes here. The folder is already on the `$PYTHONPATH` inside the container, so Python nodes can be run directly:

```bash
# Inside the container
python3 /app/exchange/my_node.py
```

#### Hello World example

A ready-to-run example is provided in `exchange/hello_booster.py`. It subscribes to the RGB image, depth image, IMU and joint states published by SimBridge and prints a log line for each incoming message.

Run it inside the robot container:

```bash
# 1. Open a shell in the robot container (from the host)
docker exec -ti CIRCUS_red_Booster-T1_0_container bash

# 2. Inside the container, launch the script
python3 /app/exchange/hello_booster.py
```

Expected output:
```
[INFO] HelloBooster node started — waiting for messages...
[INFO] [RGB]   640x480  encoding=rgb8
[INFO] [DEPTH] 640x480  encoding=mono8
[INFO] [IMU]   accel=(0.01, -0.02, 9.81)
[INFO] [JOINTS] 23 joints  — first=-0.003 rad
```

Available topics (check with `ros2 topic list` inside the container):

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | RGB camera (rgb8) |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Depth image (mono8) |
| `/booster/ros2_k2_imu` | `sensor_msgs/Imu` | IMU data |
| `/booster/ros2_k2_joint_states` | `sensor_msgs/JointState` | Joint positions/velocities |

### Allow X11 forwarding (required for GUI tools like rviz2 inside containers)

Run this once on the host before starting Circus:

```bash
xhost +local:docker
```

### Run Circus

```bash
cd circus
pixi run circus resources/scenes/1v1.yaml
```

It will spawn one container for each robot. Inside each container all ROS2 topics for that robot are available.

### Access the robot container

```bash
docker exec -ti CIRCUS_red_Booster-T1_0_container bash
```

Inside the container you can launch ROS2 tools:

```bash
# Visualize topics
ros2 topic list

# Open rviz2 (requires xhost +local:docker on host first)
rviz2
```

### Control the robot inside the container

```bash
loco
```

#### Commands

| Key | Action |
|-----|--------|
| `mw` | Mode: Walking (stand up) |
| `w` | Walk forward |

**Startup sequence:** `mw` → wait → `w` to walk.
# tiagoLINUXVALE
