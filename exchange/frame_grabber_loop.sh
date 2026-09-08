
set -e
source /opt/ros/humble/setup.bash
source ~/tiago_public_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///root/exchange/exchange/waiter_robot/config/cyclonedds_config_mac.xml
GRAB=/root/exchange/exchange/frame_grabber_tiago.py
echo "[grab-loop] restarting the frame grabber every ~3s (Mac camera workaround)"
while true; do
  timeout 3 python3 "$GRAB" >/dev/null 2>&1 || true
  sleep 0.3
done
