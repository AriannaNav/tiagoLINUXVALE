#!/bin/bash
# overhead_loop.sh — Mac workaround for the overhead (bird's-eye) camera stream.
# Same pattern as frame_grabber_loop.sh: a fresh subscriber each cycle (the
# long-running one stalls under emulation). Run inside the container:
#   bash /root/exchange/exchange/overhead_loop.sh
set -e
source /opt/ros/humble/setup.bash
source ~/tiago_public_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///root/exchange/exchange/waiter_robot/config/cyclonedds_config_mac.xml
GRAB=/root/exchange/exchange/overhead_grabber.py
echo "[overhead-loop] restarting the overhead grabber every ~3s"
while true; do
  timeout 10 python3 "$GRAB" >/dev/null 2>&1 || true
  sleep 0.3
done
