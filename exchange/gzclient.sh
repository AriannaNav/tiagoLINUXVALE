#!/bin/bash

source /opt/ros/humble/setup.bash
[ -f /root/tiago_public_ws/install/setup.bash ] && \
    source /root/tiago_public_ws/install/setup.bash

PKGS="tiago_description pmb2_description pal_hey5_description \
      pal_gripper_description pal_robotiq_description omni_base_description \
      pal_urdf_utils pal_gazebo_worlds"
MP=""
for p in $PKGS; do
    prefix=$(ros2 pkg prefix "$p" 2>/dev/null) || continue
    [ -n "$prefix" ] && [ -d "$prefix/share" ] && MP="$MP:$prefix/share"
done
WORLDS_SRC=/root/tiago_public_ws/src/pal_gazebo_worlds
export GAZEBO_MODEL_PATH="${MP#:}:$WORLDS_SRC/models:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_RESOURCE_PATH="$WORLDS_SRC:${GAZEBO_RESOURCE_PATH:-}"

export DISPLAY="${DISPLAY:-:0}"
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_SOFTWARE=1

if ! pgrep gzserver >/dev/null; then
    echo "WARNING: no gzserver running — start the sim first, or the GUI will"
    echo "         sit on 'Gazebo is not responding'."
fi
echo "GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH"
exec gzclient "$@"
