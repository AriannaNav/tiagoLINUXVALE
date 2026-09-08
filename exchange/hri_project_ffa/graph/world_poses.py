"""Read model positions directly from waiter_scene.world, so tables, bottles and
counters cannot drift out of sync with the scene the way hand-copied constants did.

Only a model's own pose comes from here. Approach offsets stay as explicit
constants in the caller: they encode a placement decision (room layout, chair
clearance) that the flat list of poses does not itself express."""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
WORLD_FILE = os.path.normpath(os.path.join(_HERE, "..", "..",
                                           "waiter_scene.world"))

_POSE_RE_TMPL = (
    r'(?:<include>\s*<name>{name}</name>.*?<pose>|'
    r'<model\s+name="{name}"[^>]*>\s*(?:<static>[^<]*</static>\s*)?<pose>)'
    r'\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)'
)

_cache = None

def _load(world_file=WORLD_FILE):
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(world_file) as f:
            _cache = f.read()
    except OSError:
        _cache = ""
    return _cache

def world_pose(name, world_file=WORLD_FILE):
    """(x, y, z, yaw) of model `name` in waiter_scene.world, or None if the
    model isn't there (removed from the scene, or the file is unreadable —
    callers should fall back to a sane default rather than crash)."""
    text = _load(world_file)
    if not text:
        return None
    m = re.search(_POSE_RE_TMPL.format(name=re.escape(name)), text, re.S)
    if not m:
        return None
    x, y, z, _roll, _pitch, yaw = (float(g) for g in m.groups())
    return (x, y, z, yaw)
