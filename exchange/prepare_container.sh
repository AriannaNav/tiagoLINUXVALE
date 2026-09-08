#!/usr/bin/env bash

set -o pipefail

WS=/root/tiago_public_ws
SRC=/root/exchange/exchange/waiter_robot
LINK=${WS}/src/waiter_robot

source /opt/ros/humble/setup.bash
[[ -f ${WS}/install/setup.bash ]] && source ${WS}/install/setup.bash

for dir in ${WS}/src/pal_urdf_utils/urdf/laser \
           /opt/ros/humble/share/pal_urdf_utils/urdf/laser; do
  if [[ -d "${dir}" ]]; then
    sed -i 's#<visualize>true</visualize>#<visualize>false</visualize>#' \
      "${dir}"/*.gazebo.xacro 2>/dev/null || true
    echo "[prepare] laser rendering off in ${dir}"
  fi
done

if [[ -d "${SRC}" ]]; then
  if [[ ! -L "${LINK}" ]] || [[ "$(readlink -f "${LINK}")" != "$(readlink -f "${SRC}")" ]]; then
    ln -sfn "${SRC}" "${LINK}"
    echo "[prepare] waiter_robot linked into the workspace"
  fi
  if [[ ! -f "${WS}/install/waiter_robot/share/waiter_robot/package.xml" ]]; then
    echo "[prepare] building waiter_robot..."
    (cd "${WS}" && colcon build --packages-select waiter_robot --symlink-install \
       --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5)
  fi
  [[ -f ${WS}/install/setup.bash ]] && source ${WS}/install/setup.bash
else
  echo "[prepare] WARNING: ${SRC} not found -- waiter_robot not built"
fi

sysctl -w net.core.rmem_max=10485760 >/dev/null 2>&1 || \
  echo "[prepare] could not raise rmem_max -- CycloneDDS may warn"

CDDS=${WS}/install/waiter_robot/share/waiter_robot/config/cyclonedds_config.xml
if [[ -f "${CDDS}" ]]; then
  export CYCLONEDDS_URI="file://${CDDS}"
  grep -q CYCLONEDDS_URI /root/.bashrc 2>/dev/null || \
    echo "export CYCLONEDDS_URI=\"file://${CDDS}\"" >> /root/.bashrc
  echo "[prepare] CYCLONEDDS_URI set"
fi

echo "[prepare] done"
