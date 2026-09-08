#!/usr/bin/env bash

set -euo pipefail

IMAGE="hrai-2025:v1"
CONTAINER="tiago_waiter"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
IN="/root/exchange"
EX="${IN}/exchange"
APP="${EX}/hri_project_ffa"
ROBOT="${EX}/waiter_robot"

log() { printf '\033[36m[start]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[start]\033[0m %s\n' "$*" >&2; exit 1; }

dexec() { docker exec "${CONTAINER}" bash -lc "$1"; }
dbg()   { docker exec -d "${CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash;
          source /root/tiago_public_ws/install/setup.bash; $1"; }

if [[ "${1:-}" == "--stop" ]]; then
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  pkill -f 'hri_project_ffa/_serve_dashboard' >/dev/null 2>&1 || true
  log "stopped"
  exit 0
fi

command -v docker >/dev/null || die "docker not found"
docker image inspect "${IMAGE}" >/dev/null 2>&1 \
  || die "image ${IMAGE} not found -- build it from dockerfiles/ first"

if [[ -x "${HERE}/gazebo_models/fetch_models.sh" ]]; then
  bash "${HERE}/gazebo_models/fetch_models.sh" || \
    log "WARNING: model fetch failed; part of the scene may be missing"
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
xhost +local:docker >/dev/null 2>&1 || true

GPU_ARGS=()
if dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
  GPU_ARGS=(--gpus all)
else
  log "no nvidia-container-toolkit; running on CPU (Gazebo will be slow)"
fi

log "starting ${CONTAINER}"
docker run -d "${GPU_ARGS[@]}" \
  --name "${CONTAINER}" \
  --net host --privileged \
  --env DISPLAY --env QT_X11_NO_MITSHM=1 \
  --env "GAZEBO_MODEL_PATH=${EX}/gazebo_models" \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "${REPO}:${IN}" \
  --workdir "${IN}" \
  "${IMAGE}" sleep infinity >/dev/null

log "preparing the container (laser rendering off, waiter_robot build)"
dexec "bash ${EX}/prepare_container.sh" 2>&1 | sed 's/^/  /' || \
  log "WARNING: preparation failed -- waiter_robot may not be built"

GUI="true"
if [[ "${1:-}" == "--headless" ]]; then
  GUI="false"
  log "headless: no Gazebo window, watch the cameras on the dashboard instead"
fi

log "launching Gazebo with waiter_scene.world"
dbg "ros2 launch tiago_gazebo tiago_gazebo.launch.py \
       is_public_sim:=True gui:=${GUI} \
       world_name:=${EX}/waiter_scene.world \
       > /tmp/gazebo.log 2>&1"

gazebo_ready() {
  dexec "source /opt/ros/humble/setup.bash &&
         timeout 10 ros2 service list 2>/dev/null | grep -qx /spawn_entity" \
    >/dev/null 2>&1
}

robot_in_world() {
  # Retried: one missed service call against a busy gzserver proves nothing,
  # and a single false negative used to abort a startup that had worked.
  local i
  for i in 1 2 3; do
    if dexec "source /opt/ros/humble/setup.bash &&
              timeout 15 ros2 service call /get_model_list gazebo_msgs/srv/GetModelList" \
         2>/dev/null | grep -q "'tiago'"; then
      return 0
    fi
    sleep 3
  done
  return 1
}

log "waiting for Gazebo to load the world (88 models, slow on CPU)"
for _ in $(seq 1 60); do
  gazebo_ready && { log "Gazebo is up"; break; }
  sleep 5
done
gazebo_ready || die "Gazebo never came up -- docker exec ${CONTAINER} cat /tmp/gazebo.log"

log "putting the robot in the world"
for attempt in 1 2 3 4 5; do
  robot_in_world && { log "robot is in the world"; break; }
  log "  spawn attempt ${attempt}"
  dexec "source /opt/ros/humble/setup.bash;
         source /root/tiago_public_ws/install/setup.bash 2>/dev/null;
         timeout 120 ros2 run gazebo_ros spawn_entity.py \
           -topic robot_description -entity tiago" >/dev/null 2>&1 || true
  sleep 5
done
robot_in_world || die "the robot never came up -- docker exec ${CONTAINER} cat /tmp/gazebo.log"

log "waiting for the arm and base controllers"
for _ in $(seq 1 30); do
  dexec "source /opt/ros/humble/setup.bash && timeout 10 ros2 control list_controllers" \
    2>/dev/null | grep -q 'arm_controller.*active' && { log "controllers active"; break; }
  sleep 2
done

log "starting SLAM (laser on /scan_raw)"
dbg "ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True \
       slam_params_file:=${ROBOT}/config/slam_params.yaml \
       > /tmp/slam.log 2>&1"

log "starting Nav2"
dbg "ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True \
       params_file:=${ROBOT}/config/nav2_params.yaml \
       > /tmp/nav2.log 2>&1"

log "waiting for SLAM to publish the map frame"
for _ in $(seq 1 40); do
  if dexec "source /opt/ros/humble/setup.bash &&
            timeout 8 ros2 run tf2_ros tf2_echo map base_footprint" \
       2>/dev/null | grep -q Translation; then
    log "map frame is up"
    break
  fi
  sleep 3
done

log "spawning the overhead camera"
dexec "source /opt/ros/humble/setup.bash &&
       ros2 run gazebo_ros spawn_entity.py -entity overhead_cam \
         -file ${EX}/overhead_cam.sdf" \
  >/dev/null 2>&1 || log "overhead camera failed to spawn -- table occupancy stays unknown"

log "starting the frame grabbers"
dbg "python3 ${EX}/frame_grabber_tiago.py > /tmp/frames.log 2>&1"
dbg "python3 ${EX}/overhead_grabber.py > /tmp/overhead.log 2>&1"

log "starting the command bridge"
dbg "python3 ${APP}/ros_nodes/hri_bridge.py > /tmp/bridge.log 2>&1"

log "starting the dashboard on http://localhost:8077"
pkill -f 'hri_project_ffa/_serve_dashboard' >/dev/null 2>&1 || true
( cd "${HERE}/hri_project_ffa" && nohup python3 _serve_dashboard.py \
    > /tmp/dashboard.log 2>&1 & )

sleep 3
cat <<EOF

  Ready.

  3D view      the Gazebo window (X11). Missing meshes? run gzclient.sh inside
               the container, it rebuilds GAZEBO_MODEL_PATH.
  Dashboard    http://localhost:8077/dashboard.html
  Logs         docker exec ${CONTAINER} tail -f /tmp/{gazebo,slam,nav2,bridge}.log

  Now start the conversation on the HOST, in another terminal:

      cd ${HERE}/hri_project_ffa
      export GEMINI_API_KEY=...
      python3 waiter.py

EOF

if [[ "${1:-}" == "--shell" ]]; then
  docker exec -ti "${CONTAINER}" bash
fi
