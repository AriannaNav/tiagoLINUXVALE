#!/usr/bin/env bash

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS="VisitorKidSit:OpenRobotics"

for entry in ${MODELS}; do
  name="${entry%%:*}"
  owner="${entry##*:}"

  if [[ -f "${HERE}/${name}/model.config" ]]; then
    echo "[models] ${name} already here"
    continue
  fi

  echo "[models] fetching ${name} from Fuel (${owner})"
  tmp="$(mktemp -d)"
  url="https://fuel.gazebosim.org/1.0/${owner}/models/${name}.zip"

  if ! curl -sL --max-time 120 -o "${tmp}/model.zip" "${url}"; then
    echo "[models] WARNING: could not download ${name} -- it will be missing from the scene" >&2
    rm -rf "${tmp}"
    continue
  fi

  if python3 - "$tmp/model.zip" "${HERE}/${name}" <<'PY'
import sys, zipfile, pathlib, shutil
archive, target = sys.argv[1], pathlib.Path(sys.argv[2])
shutil.rmtree(target, ignore_errors=True)
target.mkdir(parents=True)
with zipfile.ZipFile(archive) as zf:
    zf.extractall(target)
if not (target / "model.config").exists():
    raise SystemExit("archive did not contain a model.config")
PY
  then
    echo "[models] ${name} installed"
  else
    echo "[models] WARNING: ${name} downloaded but did not unpack cleanly" >&2
    rm -rf "${HERE:?}/${name}"
  fi
  rm -rf "${tmp}"
done
