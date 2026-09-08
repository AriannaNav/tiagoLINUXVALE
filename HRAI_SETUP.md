# HRAI Setup Guide

This document covers the HRAI-specific extensions added to the circus/simbridge framework:
custom scene, downloadable 3D meshes, RGB+depth camera, and exchange scripts.

---

## 1. Scene setup

### Scene file

`circus/resources/scenes/hrai.yaml` defines the HRI scene:
- One Booster T1 robot positioned in front of a table
- A real dining table mesh (from Gazebo Fuel)
- Table objects: mug, bowl, teapot, plate (from Google Scanned Objects)
- No soccer field — checker-pattern ground instead

### Download meshes

Meshes are not committed to the repo. Download them before running:

```bash
python3 circus/resources/meshes/objects/download_objects.py
```

This downloads `.obj` files from [Gazebo Fuel](https://fuel.gazebosim.org) into
`circus/resources/meshes/objects/`. The table `.dae` is converted to `.obj` automatically.

If the table conversion fails, run it manually:

```bash
python3 circus/resources/meshes/objects/dae2obj.py \
    circus/resources/meshes/objects/table.dae \
    circus/resources/meshes/objects/table.obj
```

### Adding new objects

**Step 1 — Find the model on Gazebo Fuel**

Browse [fuel.gazebosim.org](https://fuel.gazebosim.org) and note the owner and model name.
For example: owner `GoogleResearch`, model `Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White`.

**Step 2 — Add it to the download script**

Open `circus/resources/meshes/objects/download_objects.py` and add a line to `OBJECTS`:

```python
OBJECTS = [
    # (local_name, base_url, fuel_model_name)
    ("mug",    FUEL_BASE_GSO,      "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White"),
    ("mytool", FUEL_BASE_GSO,      "My_New_Object_Name"),       # <-- add here
    ("table",  FUEL_BASE_OPENROBO, "Dining Table"),
]
```

Use `FUEL_BASE_GSO` for Google Scanned Objects and `FUEL_BASE_OPENROBO` for OpenRobotics models.

**Step 3 — Run the download script**

```bash
python3 circus/resources/meshes/objects/download_objects.py
```

The mesh is saved as `circus/resources/meshes/objects/<local_name>.obj`.

**Step 4 — Add it to the scene YAML**

Open `circus/resources/scenes/hrai.yaml` and add an entry under `objects`:

```yaml
objects:
  - name: mytool
    type: mesh
    mesh: mytool          # matches the local_name from the download script
    position: [x, y, z]
    scale: 1.0
    fixed: false
```

**Notes on scale and position:**
- Google Scanned Objects (GSO) are in real meters — `scale: 1.0` is correct
- The dining table top surface is at `z ≈ 0.74` — place objects at that height
- If a mesh appears too large or too small, adjust `scale` (e.g. `scale: 0.5`)

### Object spec reference

```yaml
objects:
  - name: mug
    type: mesh          # box | sphere | cylinder | mesh
    mesh: mug           # filename without extension in meshes/objects/
    position: [x, y, z]
    scale: 1.0
    fixed: false        # true = welded to world (no physics)
    rgba: [r, g, b, a]  # optional, only applies to primitives (box/sphere/cylinder)
```

---

## 2. Running the simulation

### X11 forwarding (required for MuJoCo viewer)

On the host before starting:

```bash
xhost +local:docker
```

### Start everything

```bash
cd hrai-25-26-course-project-of-hrai-25-26-HRAI-Container
docker compose up
```

This starts circus (simulator) and one Booster T1 container.

---

## 3. ROS2 topics

| Topic | Type | Description |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | RGB camera (rgb8) |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Depth aligned to RGB (mono8) |
| `/booster/ros2_k2_imu` | `sensor_msgs/Imu` | IMU data |
| `/booster/ros2_k2_joint_states` | `sensor_msgs/JointState` | Joint positions/velocities |

---

## 4. Exchange scripts

All scripts live in `exchange/` and run **inside the robot container**:

```bash
docker exec -ti CIRCUS_red_Booster-T1_0_container bash
cd /app/exchange
```

### hello_booster.py

Subscribes to all sensors and logs one line per message. Good starting point.

```bash
python3 hello_booster.py
```

Expected output:
```
[hello_booster]: RGB  640x480
[hello_booster]: Depth 640x480
[hello_booster]: IMU  ax=-0.01 ay=0.00 az=9.81
[hello_booster]: Joints [27 joints]
```

### teleop.py

Simple keyboard teleop using the Booster Python SDK.

```bash
python3 teleop.py
```

### teleop_cpp (recommended)

C++ teleop using the Booster SDK directly. Build once inside the container:

```bash
cd /app/exchange/teleop_cpp
mkdir -p build && cd build
cmake .. && make
./teleop
```

Controls:

| Key | Action |
|---|---|
| `w` / `s` | Forward / Backward |
| `a` / `d` | Strafe left / right |
| `q` / `e` | Rotate left / right |
| `Space` | Stop |
| `p` | Walking mode |
| `o` | Prepare mode (stand up) |
| `i` | Damping mode (safe stop) |
| `r` | Get up (Damping → GetUp → Prepare) |
| `Ctrl+C` | Quit |

The robot starts in **Prepare mode** automatically on launch.

### scan_relay.py

Relays LiDAR scan data.

```bash
python3 scan_relay.py
```

---

## 5. RViz2

A pre-configured RViz2 config is in `exchange/rviz/`.

```bash
# On the host (requires xhost +local:docker)
ros2 run rviz2 rviz2 -d /app/exchange/rviz/booster.rviz
```

Or inside the container:

```bash
export DISPLAY=:0
rviz2 -d /app/exchange/rviz/booster.rviz
```
