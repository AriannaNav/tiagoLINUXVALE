#!/usr/bin/env python3
"""The waiter behaviour loop, host side.

The robot goes to a table, greets, listens (Whisper), understands the order (the
LLM), confirms, fetches the drink at the counter and serves it. It drives the robot
by writing shared/command.json and reading shared/status.json, both handled inside
the container by ros_nodes/hri_bridge.py.

Needs the simulation and the bridge running, plus Ollama with qwen2.5:7b.

  python waiter.py
  python waiter.py --no-sim          rehearsal: webcam, mic and LLM, robot stubbed
  python waiter.py --no-sim --text   type the order instead of speaking"""
import argparse
import atexit
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(HERE, "shared")
COMMAND_FILE = os.path.join(SHARED, "command.json")
STATUS_FILE = os.path.join(SHARED, "status.json")
SCENE_GRAPH_FILE = os.path.join(SHARED, "scene_graph.json")
AWARENESS_FILE = os.path.join(SHARED, "awareness.json")
EPISODES_FILE = os.path.join(SHARED, "episodes.json")
EVENTS_FILE = os.path.join(SHARED, "events.json")
MOODS_FILE = os.path.join(SHARED, "customer_moods.json")

_EVENTS_LOCK = threading.Lock()

def log_event(icon, text, triples=None):
    """Append one entry to the dashboard's event timeline (best-effort)."""
    entry = {"ts": time.time(), "icon": icon, "text": str(text)}
    if triples:
        entry["triples"] = [list(t) for t in triples]
    with _EVENTS_LOCK:
        try:
            try:
                with open(EVENTS_FILE) as f:
                    events = json.load(f).get("events", [])
            except (OSError, ValueError):
                events = []
            events.append(entry)
            with open(EVENTS_FILE, "w") as f:
                json.dump({"events": events[-120:]}, f)
        except OSError:
            pass

def log_mood(entity, mood):
    """Record the mood read at a customer's table for the dashboard."""
    if not entity or not mood:
        return
    try:
        try:
            with open(MOODS_FILE) as f:
                moods = json.load(f)
        except (OSError, ValueError):
            moods = {}
        moods[str(entity)] = {"emotion": mood.get("emotion"),
                              "sentiment": mood.get("sentiment"),
                              "ts": time.time()}
        with open(MOODS_FILE, "w") as f:
            json.dump(moods, f)
    except OSError:
        pass

def record_episode(kind, **fields):
    """Store one experience case <situation, action, outcome, valence>."""
    entry = {"stamp": time.time(), "kind": kind, **fields}
    try:
        try:
            with open(EPISODES_FILE) as f:
                episodes = json.load(f)
        except (OSError, ValueError):
            episodes = []
        episodes = (episodes + [entry])[-200:]
        tmp = EPISODES_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(episodes, f, indent=2)
        os.replace(tmp, EPISODES_FILE)
    except OSError:
        pass

def similar_episodes(kind, item=None, near=None, radius=0.6,
                     max_age=7 * 24 * 3600.0):
    """Retrieve similar past cases: same kind, optionally same item and/or
    spatially near a map position. Recency-bounded (default one week)."""
    try:
        with open(EPISODES_FILE) as f:
            episodes = json.load(f)
    except (OSError, ValueError):
        return []
    now, out = time.time(), []
    for e in episodes:
        if e.get("kind") != kind or now - e.get("stamp", 0) > max_age:
            continue
        if item is not None and e.get("item") != item:
            continue
        if near is not None:
            p = e.get("pos")
            if not p or math.hypot(p[0] - near[0], p[1] - near[1]) > radius:
                continue
        out.append(e)
    return out

INTENT_PRIOR = {
    "juice":     ("something refreshing", ["water", "sprite", "coca cola"]),
    "water":     ("something to quench your thirst", ["juice", "sprite"]),
    "sprite":    ("a sweet fizzy drink", ["coca cola", "juice", "water"]),
    "coca cola": ("a sweet fizzy drink", ["sprite", "juice", "water"]),
    "pringles":  ("a snack", ["coca cola", "sprite"]),
}
CUSTOMERS_FILE = os.path.join(SHARED, "customers_present.json")
PLACES_FILE = os.path.join(SHARED, "places_seen.json")

_SESSION = {"t0": 0.0}

def _bridge_file_ok(mtime, max_age):
    """Freshness gate for the files the bridge writes. A file rewritten in the
    last few seconds proves the bridge is ALIVE RIGHT NOW (it refreshes the
    layout every ~5 s), so it describes THIS sim session even if the waiter
    started after it — accept it immediately. Anything older falls back to the
    strict session rule (written after our t0, and not stale)."""
    age = time.time() - mtime
    return age <= 30.0 or (age <= max_age and mtime >= _SESSION["t0"])

def places_seen(max_age=3600.0):
    """{name: {"pos": [x, y], ...}} of the places the robot visited during its
    scan, or {} if the scan is missing/stale/from a previous session."""
    try:
        mtime = os.path.getmtime(PLACES_FILE)
        if not _bridge_file_ok(mtime, max_age):
            return {}
        with open(PLACES_FILE) as f:
            return json.load(f).get("places", {}) or {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}

_PLACES_REGISTERED = set()

def register_places_in_kg():
    """Move the places seen during exploration into the knowledge graph ABox.
    
    Idempotent and called every cycle while the patrol runs, so a place enters the
    graph the moment the robot stands in front of it. Before that,
    kg.ask('bathroom', 'isA', 'Bathroom') is None — unknown, not false."""
    seen = places_seen()
    kg = _GRAPH["kg"]
    sg = _GRAPH.get("sg")
    for name, info in seen.items():
        pos = info.get("pos") or []
        klass = name.capitalize()
        if kg is not None and not kg.ask(name, "isA", klass):
            kg.add(name, "isA", klass, abox=True)
            kg.add(name, "isLocatedIn", "Bar", abox=True)
            if len(pos) >= 2:
                kg.add(name, "hasPosition", "(%.1f, %.1f)" % (pos[0], pos[1]),
                       abox=True)
            kg.infer()
        if name not in _PLACES_REGISTERED and sg is not None and len(pos) >= 2:
            try:
                from graph.object_node import ObjectNode
                sg.add_node(ObjectNode(
                    node_id=f"place_{name}", label=name,
                    centroid=(pos[0], pos[1], 1.0), frame="map",
                    material="structure", shape="room", is_movable=False,
                    source="patrol", depth_source="prior",
                    affordances=("approached", "indicated"),
                    description=f"the {name} of the bar"))
                sg.save(SCENE_GRAPH_FILE)
                _PLACES_REGISTERED.add(name)
                print(f"🕸️  Place learned during the scan: {name} at {tuple(pos)}")
                log_event("🗺️", f"place learned: {name} at ({pos[0]:.1f}, {pos[1]:.1f})",
                          triples=[[name, "isA", name.capitalize()],
                                   [name, "isLocatedIn", "Bar"]])
            except Exception as e:
                print("(could not add place to scene graph:", e, ")")
    return seen

def known_place(name):
    """(x, y) of a place, but ONLY if the robot saw it during exploration —
    when the graph layer is on, the knowledge graph must contain the instance
    (it is registered from the scan); the position comes from the scan file."""
    info = places_seen().get(name)
    if not info or len(info.get("pos") or []) < 2:
        return None
    kg = _GRAPH["kg"]
    if kg is not None and not kg.ask(name, "isA", name.capitalize()):
        register_places_in_kg()
        if not kg.ask(name, "isA", name.capitalize()):
            return None
    return tuple(info["pos"][:2])

def detected_customers(max_age=600.0):
    """List of customers the robot registered during its scan:
    [{table_id, entity, pos}], or [] if the scan is missing/stale/from a
    previous session (see _SESSION: the file is root-owned, not deletable)."""
    try:
        mtime = os.path.getmtime(CUSTOMERS_FILE)
        if not _bridge_file_ok(mtime, max_age):
            return []
        with open(CUSTOMERS_FILE) as f:
            present = json.load(f).get("present", [])
        return [c for c in present if c.get("table_id") in TABLES]
    except (OSError, ValueError, KeyError, TypeError):
        return []

def detected_tables(max_age=600.0):
    """Table ids where the robot saw a customer during its scan, or None if the
    scan produced nothing / is missing (caller then falls back to all tables)."""
    ids = [c["table_id"] for c in detected_customers(max_age)]
    return ids or None

from graph.world_poses import world_pose

# Geometry of model://bar_table (models/bar_table/model.sdf), the model the
# dining tables actually use: a ROUND top, and its centre is NOT the include
# pose — it sits at (+0.40, +0.75) in the model frame, radius 0.65, top z 0.72.
# The docks used to be flat offsets from the include pose, written when the
# tables were the old rectangular corner-origin kitchen_table. Kept after the
# swap, they parked the robot on the table's diagonal facing ~28 deg PAST the
# table and ~23 deg past the customer, and dropped the drink on the far rim.
_TABLE_CENTRE_OFFSET = (0.40, 0.75)
_TABLE_RADIUS = 0.65
_TABLE_TOP_Z = 0.72

# Which way the open aisle lies from each table's centre: tables 1/4 stand on
# the east wall (approach from the west), 2/3 on the west wall.
_AISLE_DIR = {1: (-1.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 0.0), 4: (-1.0, 0.0)}
# Every customer sits at the SOUTH seat of their table (see waiter_scene.world:
# male03, female02, female03, female02bis are all 0.90 m south of the centre).
_CUSTOMER_DIR = (0.0, -1.0)

_BASE_STANDOFF = 0.42   # m from the table rim to the base centre (base r ~0.27)
_SERVE_RADIUS = 0.40    # m from the table centre, on the robot's side of the top

_TABLE_FALLBACK_BASE = {1: (4.0, 2.5), 2: (-2.0, 2.5), 3: (-2.0, 7.0), 4: (4.0, 7.0)}

def _table_positions(tid):
    """(person_base, serve_place) for a table, derived from the round table's
    real centre.

    The dock sits on the bisector of the aisle direction and the customer
    direction, so the robot comes in from the open side yet ends up on the
    customer's corner rather than square to a table edge that no longer exists;
    its yaw points at the CUSTOMER, not along a world axis. The serve point is
    on that same bisector, well inside the rim (0.40 of 0.65) and about 0.67 m
    in front of the dock, i.e. within the arm's offer reach."""
    pose = world_pose("dining_table_%d" % tid)
    bx, by = pose[:2] if pose else _TABLE_FALLBACK_BASE[tid]
    cx = bx + _TABLE_CENTRE_OFFSET[0]
    cy = by + _TABLE_CENTRE_OFFSET[1]

    ax, ay = _AISLE_DIR[tid]
    ux, uy = ax + _CUSTOMER_DIR[0], ay + _CUSTOMER_DIR[1]
    n = math.hypot(ux, uy) or 1.0
    ux, uy = ux / n, uy / n

    d = _TABLE_RADIUS + _BASE_STANDOFF
    px, py = cx + ux * d, cy + uy * d
    custx = cx + _CUSTOMER_DIR[0] * 0.90
    custy = cy + _CUSTOMER_DIR[1] * 0.90
    person = (round(px, 3), round(py, 3),
              round(math.atan2(custy - py, custx - px), 3))
    serve = (round(cx + ux * _SERVE_RADIUS, 3), round(cy + uy * _SERVE_RADIUS, 3),
             round(_TABLE_TOP_Z + 0.02, 3))
    return person, serve

TABLES = {}
for _tid in (1, 2, 3, 4):
    _person, _serve = _table_positions(_tid)
    TABLES[_tid] = {"id": _tid, "name": "table %d" % _tid,
                    "person": _person, "serve": _serve}
del _tid, _person, _serve
TABLE_ORDER = [1, 2, 3, 4]

def scene_from_graph(max_age=30.0):
    """Live scene description from the 3D scene graph, read from
    shared/scene_graph.json. Returns None when the file is missing or stale, so
    callers can fall back to the static menu description."""
    try:
        if time.time() - os.path.getmtime(SCENE_GRAPH_FILE) > max_age:
            return None
        from graph.scene_graph import SceneGraph3D
        sg = SceneGraph3D.load(SCENE_GRAPH_FILE)
        if not sg.nodes:
            return None
        return sg.to_text()
    except Exception:
        return None

NAV_WAIT = float(os.environ.get("WAITER_NAV_WAIT", "500"))
# Quando c'è una caduta, il robot va FISICAMENTE allo spill e aspetta lì la
# pulizia. In rehearse (no sim/bridge) si salta la navigazione.
_REHEARSE = False
SPILL_NAV_WAIT = float(os.environ.get("WAITER_SPILL_NAV_WAIT", "90"))
EXPLORE_WAIT = float(os.environ.get("WAITER_EXPLORE_WAIT", "1100"))
CUSTOMER_VLM_PERIOD = float(os.environ.get("WAITER_CUSTOMER_VLM_PERIOD", "150"))

GRAPH_ENABLED = os.environ.get("WAITER_GRAPH", "1") != "0"
EXPLORE_ENABLED = os.environ.get("WAITER_EXPLORE", "1") != "0"
_GRAPH = {"sg": None, "kg": None, "tm": None, "explore_done": False}

DRINK_APPEARANCE = {"coca cola": ("bottle", "red"), "sprite": ("bottle", "green"),
                    "water": ("bottle", "blue"), "juice": ("bottle", "yellow"),
                    "pringles": ("can", "red")}

def end_exploration():
    """Signal that the exploration patrol is over: the temporal manager
    switches from the exploration phase to tracking on the next frame."""
    _GRAPH["explore_done"] = True

COUNTER_VIEW_FLAG = os.path.join(SHARED, "counter_view.active")

def at_counter(perception=None):
    """True only while the bridge has the robot parked at a counter with its
    head down (it raises this flag for exactly that window). Reading at any
    other moment spends a VLM call to be told there is a wall in view."""
    return os.path.exists(COUNTER_VIEW_FLAG)

def counter_spot():
    """Which counter spot the head is settled on right now, or None.

    The bridge rewrites the flag with a new value every time the head reaches
    another item, so a change here means "a different bottle is in frame" —
    the cue to spend a recognition call. Without it the scan sampled on a
    timer and missed whichever spot the timer skipped."""
    try:
        with open(COUNTER_VIEW_FLAG) as f:
            return f.read().strip() or None
    except OSError:
        return None

def _robot_still(perception, last, tol_xy=0.04, tol_yaw=math.radians(3.0)):
    """True only when the robot has NOT moved since the previous graph cycle
    and a fresh map pose is available. While driving, the camera frame and
    robot_pose.json are written independently (no timestamp alignment), so
    back-projected centroids smear across the map — the graph only ingests
    frames observed from a standstill (the patrol dwells, the tables)."""
    if not perception.pose.refresh():
        last["pose"] = None
        return False
    base = perception.pose.base_xy_yaw
    prev, last["pose"] = last["pose"], base
    if prev is None or base is None:
        return False
    dyaw = abs((base[2] - prev[2] + math.pi) % (2 * math.pi) - math.pi)
    return math.hypot(base[0] - prev[0], base[1] - prev[1]) <= tol_xy \
        and dyaw <= tol_yaw

def _vlm_drink_detections(perception, model, timeout):
    """Ask the vision model which drinks are visible in the latest counter frame, why,
    and whether anything looks wrong. Returns map-frame Detections at the known counter
    positions: identity is perceived, geometry falls back to the layout."""
    try:
        from graph.perception import RGB_FRAME_PATH
        from graph import vlm_perception
    except Exception:
        return []
    if not os.path.exists(RGB_FRAME_PATH):
        return []
    try:
        result = vlm_perception.recognise_drinks(RGB_FRAME_PATH, model=model,
                                                  timeout=timeout)
        _GRAPH["counter_scanned"] = True
        names, reasoning, anomaly = (result["detected"], result["reasoning"],
                                     result["anomaly"])
        kind = result.get("anomaly_kind", "none")
        if not result.get("cached"):
            print(f"👁️  Gemini counter read: {sorted(names) or '(none recognised)'}"
                  + (f" — {reasoning}" if reasoning else ""))
        if names:
            triples = []
            for n in sorted(names):
                triples += [[n, "isA", "Snack" if n == "pringles" else "Drink"],
                            [n, "isLocatedIn", "Counter"]]
            log_event("🍾", "counter scanned: " + ", ".join(sorted(names)),
                      triples=triples)
        if anomaly and not result.get("cached"):
            print(f"⚠️  Gemini spotted something off at the counter "
                  f"({kind}): {anomaly}")
            if kind == "spill":
                _report_vlm_anomaly(anomaly, names)
        return vlm_perception.detections_for(names)
    except Exception as e:
        print("(vlm drink perception failed:", e, ")")
        return []

def _report_vlm_anomaly(anomaly, seen_names):
    """Route a visually-inferred spill through the same drop_drink() hazard
    pipeline as a physical one, so the robot reacts identically either way.

    Only reached when the model itself classified the anomaly as a spill.
    Treating every oddity as a hazard put the robot in a loop: report, clean,
    see the same unrecognised object, report again."""
    try:
        from graph.vlm_perception import DRINK_LAYOUT
        pos = DRINK_LAYOUT[next(iter(seen_names))]["pos"][:2] if seen_names \
            else (2.0, -3.2)
        drop_drink(pos=pos)
    except Exception as e:
        print("(could not register VLM-detected anomaly as a hazard:", e, ")")

GEMINI_CUSTOMER_CHECK_FILE = os.path.join(SHARED, "gemini_customer_check.json")

_TABLE_VLM_STATE = {}

def _write_gemini_customer_check(tables, reasoning="", table_states=None):
    """Persist the overhead-camera occupancy read so the dashboard can show it
    (previously terminal-only) — a visual CONFIRMATION panel next to the
    ground-truth 'Relations & detected customers' one, not a replacement.
    Now also carries the model's own reasoning and per-table state reads."""
    try:
        with open(GEMINI_CUSTOMER_CHECK_FILE, "w") as f:
            json.dump({"tables": sorted(tables), "reasoning": reasoning,
                      "table_states": table_states or {}, "ts": time.time()}, f)
    except OSError:
        pass

def _vlm_customer_check(model, timeout):
    """Confirm table occupancy from the overhead camera, alongside the ground-truth
    customer scan rather than instead of it: vision can tell that a table looks
    occupied but not which simulated customer is sitting there, and the age and
    returning-customer bookkeeping needs that identity. Also infers each table's state
    (active, finished, needs attention) from visual cues, cached in _TABLE_VLM_STATE.
    Returns the set of occupied table_ids."""
    try:
        from graph import vlm_perception
    except Exception:
        return set()
    if not os.path.exists(vlm_perception.OVERHEAD_FRAME_PATH):
        return set()
    try:
        result = vlm_perception.recognise_occupied_tables(model=model, timeout=timeout)
        tables, reasoning, states = (result["occupied"], result["reasoning"],
                                     result["table_states"])
        print(f"👁️  Gemini overhead read: tables occupied = {sorted(tables) or '(none)'}"
              + (f" — {reasoning}" if reasoning else ""))
        if states:
            print(f"👁️  Gemini table-state read: {states}")
        _TABLE_VLM_STATE.update(states)
        _write_gemini_customer_check(tables, reasoning, states)
        return tables
    except Exception as e:
        print("(vlm customer perception failed:", e, ")")
        return set()

def start_scene_graph(max_explore_seconds=240.0, period=1.5):
    """Start the in-process perception -> temporal manager loop (daemon).
    The graph stays in the EXPLORATION phase until end_exploration() is called
    (i.e. until the patrol finishes), with a hard time cap as a safety net."""
    if not GRAPH_ENABLED:
        return
    try:
        from graph.perception import PerceptionModule
        from graph.scene_graph import SceneGraph3D
        from graph.temporal_manager import TemporalManager
        from graph.knowledge_graph import build_waiter_ontology
    except Exception as e:
        print("(scene graph unavailable, continuing without it:", e, ")")
        return
    perception = PerceptionModule()
    sg = SceneGraph3D(frame_id="head_camera")
    tm = TemporalManager(sg, verbose=False)
    kg = build_waiter_ontology()
    _GRAPH["sg"], _GRAPH["kg"], _GRAPH["tm"] = sg, kg, tm
    _GRAPH["perception"] = perception
    try:
        sg.save(SCENE_GRAPH_FILE)
    except Exception:
        pass

    try:
        from config import (PERCEPTION_VLM_PERIOD, PERCEPTION_VLM_MODEL,
                            PERCEPTION_VLM_TIMEOUT)
    except Exception:
        PERCEPTION_VLM_PERIOD = 10.0
        PERCEPTION_VLM_MODEL, PERCEPTION_VLM_TIMEOUT = "gemini-3.1-flash-lite", 40

    def _loop():
        explore_until = time.time() + max_explore_seconds
        last_base = {"pose": None}
        last_vlm = [0.0, None]   # [ultima chiamata, ultimo punto guardato]
        last_customer_vlm = [0.0]
        while True:
            register_places_in_kg()
            add_customers_to_graph()
            add_counter_drinks_to_graph()
            try:
                if sg.nodes:
                    sg.refresh_relations()
                    sg.save(SCENE_GRAPH_FILE)
            except Exception as e:
                print("(relations refresh error:", e, ")")
            if tm.phase == "exploration":
                if _GRAPH["explore_done"] or time.time() > explore_until:
                    tm.start_tracking()
                    print("🕸️  Scene graph: exploration done, tracking phase.")
            frame = perception.read_bridge_frame()
            if frame is not None:
                try:
                    if _robot_still(perception, last_base) and at_counter(perception):
                        now = time.time()
                        # Look once per SPOT (the head just settled on a new
                        # item) and otherwise fall back to the timer, which
                        # still covers standing at a counter outside a scan.
                        spot = counter_spot()
                        new_spot = spot is not None and spot != last_vlm[1]
                        if new_spot or now - last_vlm[0] >= PERCEPTION_VLM_PERIOD:
                            last_vlm[0], last_vlm[1] = now, spot
                            dets = _vlm_drink_detections(
                                perception, PERCEPTION_VLM_MODEL,
                                PERCEPTION_VLM_TIMEOUT)
                            if dets:
                                tm.update(dets, fov_check=perception.in_view)
                                sg.save(SCENE_GRAPH_FILE)
                                kg.sync_scene_graph(sg)
                except Exception as e:
                    print("(scene graph update error:", e, ")")
            try:
                now = time.time()
                if now - last_customer_vlm[0] >= CUSTOMER_VLM_PERIOD:
                    last_customer_vlm[0] = now
                    _vlm_customer_check(PERCEPTION_VLM_MODEL, PERCEPTION_VLM_TIMEOUT)
            except Exception as e:
                print("(vlm customer check error:", e, ")")
            time.sleep(period)

    threading.Thread(target=_loop, daemon=True).start()
    print("🕸️  Scene-graph thread started (watching the robot camera frames).")

def seed_demo_graph():
    """Rehearsal-only: put a few counter bottles into the scene graph so the
    graph-driven behaviours (grounding, substitution, fake failures) can be
    exercised interactively WITHOUT the sim: two sprites and a water are
    visible, coca cola / juice / pringles are NOT — ordering one of those
    triggers the substitution offer. Disable with WAITER_SEED_GRAPH=0."""
    sg, kg = _GRAPH.get("sg"), _GRAPH.get("kg")
    if sg is None or any(n.label.endswith("bottle") for n in sg.nodes.values()):
        return
    try:
        from graph.object_node import ObjectNode
        from graph.perception import characterize
    except Exception:
        return
    rgb = {"green": (60, 180, 75), "blue": (0, 120, 255)}
    for nid, color, x in (("green_bottle_demo_1", "green", 1.2),
                          ("green_bottle_demo_2", "green", 1.7),
                          ("blue_bottle_demo_1", "blue", 2.2)):
        det = characterize(f"{color}_bottle", None, None, 0.9,
                           (x, -3.2, 0.85), 0.9, "demo", "prior")
        det.color_name, det.color_rgb = color, rgb[color]
        sg.add_node(ObjectNode.from_detection(nid, det))
    if not any(n.label == "person" for n in sg.nodes.values()):
        px, py = TABLES[1]["serve"][:2]
        if "table_1" not in sg.nodes:
            sg.add_node(ObjectNode(node_id="table_1", label="table",
                        centroid=(px, py, 0.4), frame="map", material="wood",
                        shape="surface", is_movable=False, source="ground_truth",
                        depth_source="prior",
                        affordances=("approached", "usedAsSupport"),
                        description="dining table 1"))
        sg.add_node(ObjectNode(node_id="customer_demo", label="person",
                    centroid=(px, py, 0.9), frame="map", material="organic",
                    shape="human", is_movable=True, source="ground_truth",
                    depth_source="prior",
                    affordances=("greeted", "askedForOrder", "servedTo"),
                    description="a demo customer at table 1"))
        sg.edges.append(("customer_demo", "isAtTable", "table_1"))
    try:
        sg.save(SCENE_GRAPH_FILE)
        if kg is not None:
            kg.sync_scene_graph(sg)
    except Exception:
        pass
    print("🕸️  (rehearsal) demo scene seeded: sprite x2 + water on the counter "
          "— coca cola, juice and pringles are NOT visible, so ordering one "
          "of those triggers the substitution offer.")

def counter_stock_known():
    """True if at least one drink in the graph was actually SEEN on the counter
    (source='vlm'). Ground-truth furniture and rehearsal seeds do not count."""
    sg = _GRAPH.get("sg")
    if sg is None:
        return False
    return any(getattr(n, "source", "") == "vlm" for n in sg.nodes.values())

def scan_counter_now():
    """Force one VLM read of the current head-camera frame into the graph and
    return how many drinks it found. Used to retry after a patrol that did not
    manage to stop in front of the counter."""
    perception, tm, sg, kg = (_GRAPH.get(k) for k in
                              ("perception", "tm", "sg", "kg"))
    if perception is None or tm is None or sg is None:
        return 0
    try:
        from config import PERCEPTION_VLM_MODEL, PERCEPTION_VLM_TIMEOUT
    except Exception:
        PERCEPTION_VLM_MODEL, PERCEPTION_VLM_TIMEOUT = "gemini-3.1-flash-lite", 40
    dets = _vlm_drink_detections(perception, PERCEPTION_VLM_MODEL,
                                 PERCEPTION_VLM_TIMEOUT)
    if dets:
        tm.update(dets, fov_check=perception.in_view)
        try:
            sg.save(SCENE_GRAPH_FILE)
            if kg is not None:
                kg.sync_scene_graph(sg)
        except Exception:
            pass
    return len(dets)

def ensure_counter_scanned(attempts=2):
    """The stock has to come from LOOKING at the counter. When the patrol did
    not manage it — a navigation goal that never completed, a missed VLM window
    — retry the sweep instead of taking orders blind. Returns True once the
    counter has been read; a False result is reported to the model as an
    explicit unknown (see scene_grounding_text), never hidden.

    Uses the bridge's scan_counter action, which parks in front of the counter
    and dwells: re-running the whole patrol would redo the tables and the room
    for nothing."""
    if counter_stock_known():
        return True
    for i in range(1, attempts + 1):
        print(f"🍾 counter not scanned yet — going to look at it "
              f"({i}/{attempts})")
        log_event("🍾", f"counter not scanned — going to look at it ({i}/{attempts})")
        eid = send_command("scan_counter")
        wait_status({"scanned", "failed"}, timeout=EXPLORE_WAIT, want_id=eid)
        if counter_stock_known():
            log_event("🍾", "counter scanned on retry")
            return True
    print("⚠️  the counter could not be scanned — the robot will say so instead "
          "of guessing what is in stock")
    log_event("⚠️", "counter could not be scanned — stock stays unknown")
    return False

def ground_order(target, exclude=()):
    """Resolve the ordered item to a node of the live scene graph, or None.
    `exclude` skips node_ids a previous fetch attempt already failed on."""
    sg, kg = _GRAPH["sg"], _GRAPH["kg"]
    if not sg or not sg.nodes:
        return None
    from graph.grounding import find_target_in_graph
    node = find_target_in_graph(sg, kg, {"target": target, "constraints": []},
                                exclude=exclude)
    if node is not None:
        return node
    label, color = DRINK_APPEARANCE.get(target, (target, ""))
    return find_target_in_graph(sg, kg,
                                {"target": label,
                                 "constraints": [color] if color else []},
                                exclude=exclude)

def measured_awareness(item):
    """Self-awareness (HRAI 9) measured from the graphs instead of guessed by the LLM.
    None when the graph layer is off or empty.
    
      P  = fraction of the serve task's required elements present in the scene graph
      C  = fraction of capability requirements met in the knowledge graph
      Pr = kg.project_task(): fraction of the symbolic chain whose preconditions hold"""
    sg, kg = _GRAPH["sg"], _GRAPH["kg"]
    if not sg or not sg.nodes or kg is None:
        return None
    node = ground_order(item) if item else None
    target_id = node.node_id if node is not None else (item or "")
    required = {
        "item visible": node is not None,
        "customer present": any(n.label == "person" for n in sg.nodes.values()),
        "table known": any(n.label == "table" for n in sg.nodes.values()),
    }
    P = sum(required.values()) / len(required)
    servable, _ = kg.can_perform("ServeItem", item or "")
    checks = {
        "servable per ontology": servable,
        "robot performs ServeItem": kg.ask("tiago", "performs", "ServeItem") is True,
        "target graspable": kg.ask(target_id, "canBe", "grasped") is True,
        "target carriable": kg.ask(target_id, "canBe", "carried") is True,
    }
    C = sum(checks.values()) / len(checks)
    Pr, trace = kg.project_task("ServeItem", target_id)
    return {"P": P, "C": C, "Pr": Pr,
            "required": required, "checks": checks, "trace": trace}

def refine_decision_with_graphs(decision):
    """Replace the LLM's guessed awareness scores with the graph-measured ones and
    recompute Ft. Purely informational: it reports the robot's own awareness for the
    dashboard and decides nothing — whether to serve stays the model's call, never a
    threshold on this score. Returns the measurement dict or None."""
    m = measured_awareness(decision.get("item", ""))
    if m is None:
        decision["scores_source"] = "llm"
        return None
    try:
        import config as _cfg
        a, b, g = getattr(_cfg, "AWARENESS_WEIGHTS", (1/3, 1/3, 1/3))
    except Exception:
        a, b, g = 1/3, 1/3, 1/3
    decision["perception_score"] = round(m["P"], 3)
    decision["comprehension_score"] = round(m["C"], 3)
    decision["projection_score"] = round(m["Pr"], 3)
    decision["feasibility_score"] = round(a*m["P"] + b*m["C"] + g*m["Pr"], 3)
    decision["scores_source"] = "graph"
    return m

def log_awareness(request, decision, measured=None):
    """Append the decision to shared/awareness.json for the dashboard panel."""
    entry = {"stamp": time.time(), "request": (request or "").strip(),
             "action": decision.get("action", ""), "item": decision.get("item", ""),
             "P": decision.get("perception_score"),
             "C": decision.get("comprehension_score"),
             "Pr": decision.get("projection_score"),
             "Ft": decision.get("feasibility_score"),
             "source": decision.get("scores_source", "llm"),
             "reasoning": decision.get("reasoning", "")}
    if measured:
        entry["details"] = {"required": measured["required"],
                            "checks": measured["checks"],
                            "trace": measured["trace"]}
    try:
        try:
            with open(AWARENESS_FILE) as f:
                history = json.load(f).get("history", [])
        except (OSError, ValueError):
            history = []
        history = ([entry] + history)[:6]
        tmp = AWARENESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"latest": entry, "history": history}, f, indent=2)
        os.replace(tmp, AWARENESS_FILE)
    except OSError as e:
        print("(could not write awareness log:", e, ")")

def order_is_servable(target):
    """Feasibility from the KNOWLEDGE GRAPH task ontology: ServeItem requires
    the item to be a Drink or Snack (see graph/knowledge_graph.py). Falls back
    to True when the graph layer is off (the brain already filters the menu)."""
    kg = _GRAPH["kg"]
    if kg is None:
        return True, []
    return kg.can_perform("ServeItem", target)

VOICE = os.environ.get("WAITER_VOICE", "Samantha")

DRINK_WORDS = {"coca", "cola", "coke", "cocacola", "sprite", "can", "drink",
               "soda", "beverage", "bottle", "water", "wine", "vino"}

DISPLAY = {"coca": "Coca-Cola", "cola": "Coca-Cola", "coke": "Coca-Cola",
           "cocacola": "Coca-Cola", "sprite": "Sprite", "water": "water",
           "pringles": "Pringles", "chips": "Pringles",
           "juice": "orange juice", "orange": "orange juice",
           "wine": "wine", "vino": "wine"}

def is_yes(text):
    t = (text or "").lower()
    return any(w in t for w in ("yes", "sure", "yeah", "yep", "ok", "okay",
                                "please", "why not", "sounds good", "go ahead",
                                "of course", "absolutely"))

BATHROOM_WORDS = ("bathroom", "restroom", "toilet", "washroom", "the loo",
                  " wc", "bagno")

def handle_bathroom_request(mood, rehearse):
    log_event("🚻", "customer asked for the bathroom — indicating the way")
    """In-between task: point the way to the bathroom, then hand control back
    to the order dialogue. Blocks until the robot finished the gesture.

    The robot only knows where the bathroom is if it PASSED it during the
    exploration patrol (the place is then in the knowledge graph); otherwise
    it honestly says it does not know."""
    pos = known_place("bathroom")
    if pos is None:
        say("I'm sorry, I haven't come across the bathroom on my rounds yet, "
            "so I can't point you to it. Let me ask a colleague later!")
        return
    say("Of course — it's this way, let me show you.")
    if not rehearse:
        iid = send_command("indicate", target="bathroom", mood=mood,
                           place_pos=pos)
        st = wait_status({"indicated", "rejected", "failed"}, timeout=120,
                         want_id=iid)
        if st != "indicated":
            print("   (bridge could not run the pointing gesture: %s)" % st)
    say("The bathroom is right where I'm pointing — the door in the far corner "
        "at the back of the room. Just head that way.")

MENU_ITEMS = ["Coca-Cola", "Sprite", "water", "orange juice", "wine", "Pringles"]

MINOR_ENTITIES = {"female02"}

WEBCAM_AGE = False

def _point_to_reception(mood, rehearse):
    """Point at the reception desk (if the robot saw it on patrol), so the
    customer can go there to have their age/ID checked."""
    pos = known_place("reception")
    if pos is not None and not rehearse:
        iid = send_command("indicate", target="reception", mood=mood,
                           place_pos=pos)
        st = wait_status({"indicated", "rejected", "failed"}, timeout=120,
                         want_id=iid)
        if st != "indicated":
            print("   (bridge could not run the pointing gesture: %s)" % st)

PRIORITY_REORDER_ENABLED = os.environ.get("WAITER_PRIORITY_REORDER", "1") != "0"
# Second experimental condition. When off, the robot still reads the mood and
# still adapts its speech and its driving speed to it; only the mood-driven
# adjustment of the APPROACH GEOMETRY is suppressed, so that the robot stops at
# the same nominal dock for every customer. Keeping the other mood-driven
# behaviours active isolates proxemics as the single manipulated variable.
PROXEMICS_ENABLED = os.environ.get("WAITER_PROXEMICS", "1") != "0"

USE_LLM = os.environ.get("WAITER_LLM", "1") != "0"
LLM_TEMP = float(os.environ.get("WAITER_LLM_TEMP", "0.7"))

DEFAULTS = {
    "greet": "Hi, welcome to our bar! What would you like to order?",
    "greet_back": "Welcome back! Great to see you again. Your usual {last}, or something different today?",
    "greet_known": "Welcome back! Great to see you again. What would you like today?",
    "repeat": "Sorry, I did not understand. Can you repeat, please?",
    "notdrink": "Sorry, I can only serve drinks, like coca cola or sprite. What would you like?",
    "menu": "We have Coca-Cola, Sprite, water, orange juice, wine, and Pringles. What would you like?",
    "confirm": "Okay, I'll bring you a {drink} right away.",
    "more": "Anything else for you?",
    "next_customer": "Is there another customer I should take an order from?",
    "serve": "Here is your {drink}. Enjoy!",
    "decline": "Sorry, I can only bring drinks like Coca-Cola, Sprite, water, orange juice, or Pringles. What would you like?",
    "clarify": "Sorry, I didn't quite catch that — which drink can I get you?",
    "no_order": "Since you don't seem to need anything right now, I'll let you be — have a great day!",
}

_LLM_PROMPTS = {
    "greet": "You are a friendly waiter robot in a bar that serves soft drinks like coca cola and sprite. Greet the customer warmly and ask what they would like to drink. ONE short spoken sentence, English, no quotes, do NOT invent menu items.",
    "greet_back": "You are a friendly waiter robot who RECOGNISES a returning customer. Their last order here was {last}. Warmly welcome them back (show you remember them) and offer 'the usual' ({last}) or ask what they'd like today. ONE or TWO short sentences, English, no quotes, do NOT invent menu items.",
    "greet_known": "You are a friendly waiter robot who RECOGNISES a returning customer, but you do NOT know their past orders yet. Warmly welcome them back and ask what they would like today. Do NOT mention 'the usual'. ONE short sentence, English, no quotes, do NOT invent menu items.",
    "repeat": "You are a friendly waiter robot. You did not understand. Politely ask the customer to repeat their drink order. ONE short sentence, max 12 words, English, no quotes.",
    "notdrink": "You are a friendly waiter robot that can only serve drinks like coca cola or sprite. The customer asked for something you cannot bring. Politely say you only serve drinks and ask what they would like. ONE short sentence, English, no quotes.",
    "menu": "You are a friendly waiter robot. The customer asked what is available. Tell them what the bar has: Coca-Cola, Sprite, water, orange juice, wine (for adults), and Pringles (crisps). List them naturally in ONE short spoken sentence and ask what they'd like. English, no quotes, do NOT invent any other items.",
    "confirm": "You are a friendly waiter robot. The customer just ordered a {drink}. Reply confirming you will bring it right away. ONE short cheerful spoken sentence, English, no quotes.",
    "more": "You are a friendly waiter robot in a bar that ONLY serves drinks (Coca-Cola, Sprite, water, orange juice, wine) and Pringles. The customer just ordered. Politely ask if they would like ANY OTHER DRINK. Do NOT invent or mention desserts, food, specials, or a menu. ONE short sentence that ends with a question, English, no quotes.",
    "next_customer": "You are a friendly waiter robot taking drink orders around the bar. Ask whether there is ANOTHER customer whose order you should take now. Do NOT mention desserts, food, specials, or a menu. ONE short sentence ending with a question, English, no quotes.",
    "serve": "You are a friendly waiter robot. You are now handing the {drink} to the customer. Say one short friendly sentence. English, no quotes.",
    "decline": "You are a waiter robot in a bar that serves ONLY drinks (Coca-Cola, Sprite, water, orange juice, wine) plus Pringles — NO food, no hot dishes, no burgers. The customer just asked for something you do NOT have (food or an off-menu item). You MUST: (1) clearly and politely tell them you don't have THAT specific item and that you only serve drinks; (2) offer 1-2 real drinks instead; (3) ask what they'd like. Do NOT greet them, do NOT change the subject, do NOT pretend you can bring it, do NOT invent items. ONE or TWO short spoken sentences, English, no quotes.",
    "clarify": "You are a friendly waiter robot. You did NOT clearly understand what the customer wants. Ask a brief, natural clarification question to find out which drink they'd like — refer to what they said if it helps. Do NOT guess or invent an order. ONE short sentence ending with a question, English, no quotes.",
    "no_order": "You are a friendly waiter robot. This customer has not ordered anything after being asked and given a few chances to answer. Politely let them know you'll leave them be for now, warmly, without being rude or pushy — do NOT ask again what they'd like. ONE short spoken sentence, English, no quotes.",
}

def _bar_facts():
    return (
        "Facts you must respect (never contradict them or add to them):\n"
        f"- The bar serves ONLY these items: {', '.join(MENU_ITEMS)}. Nothing "
        "else is available.\n"
        "- Wine is for adults only; the other drinks are fine for anyone.\n"
        "- You are TIAGo, a waiter robot. You can greet people, take a drink "
        "order, fetch ONE drink at a time and carry it to the table, and point "
        "the way to the reception or the bathroom.\n"
        "- You CANNOT make cocktails or coffee, cook or serve food (only "
        "Pringles), take payment, or promise prices, discounts, or specials.")

_OFF_MENU_RE = re.compile(
    r"\b(coffee|espresso|cappuccino|latte|tea|beer|cocktail|whisk(?:e)?y|vodka|"
    r"gin|rum|tequila|martini|mojito|smoothie|milkshake|dessert|cake|"
    r"ice[- ]?cream|pizza|burger|sandwich|fries|pasta|salad|soup|breakfast|"
    r"today'?s special|specials|discount|promotion|loyalty card|reservation|"
    r"wi-?fi|password|\beuro\b|\bdollar|\$\d)\b", re.I)

_recent_lines = []

_REFUSAL_RE = re.compile(
    r"\b(do(?:n'?t| not)|can'?t|cannot|only (?:serve|bring|have|offer)|not (?:have|"
    r"serve|on the menu|able)|afraid|unfortunately|isn'?t|aren'?t|we don'?t|"
    r"out of|unable|no (?:food|burgers?|hamburgers?|hot))\b", re.I)

def _refuses(text):
    """True if the line clearly declines the request (not just a pleasantry)."""
    return bool(_REFUSAL_RE.search(text))

def _grounded(text, user_text=""):
    """True if the reply stays within the real menu and the robot's capabilities.
    
    The customer's own words are removed first: echoing an off-menu item while refusing
    it is fine, inventing one is not."""
    checked = text
    for w in re.findall(r"[a-z]+", (user_text or "").lower()):
        if len(w) > 2:
            checked = re.sub(r"\b" + re.escape(w) + r"\b", " ", checked, flags=re.I)
    return not _OFF_MENU_RE.search(checked)

_META_MARKERS = re.compile(
    r'\s*(?:how about|or perhaps|or (?:you could|maybe|simply|just)|you could say|'
    r'alternatively|another option|or:|or ")', re.I)

def _clean_line(raw):
    text = raw.replace("\n", " ").strip().strip('"').strip()
    m = _META_MARKERS.search(text)
    if m and m.start() > 15:
        text = text[:m.start()]
    text = text.replace('"', " ")
    text = re.sub(r"\s+", " ", text).strip().strip("'").strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) > 2:
        text = " ".join(sents[:2]).strip()
    return text

def llm_reply(situation, drink="", mood=None, customer=None, user_text=""):
    """Generate the robot's spoken line with the LLM, grounded in the real menu and
    validated so it neither hallucinates nor sounds canned. Falls back to a fixed
    phrase only when the LLM is off, errors, or cannot produce a grounded reply.
    
    `mood` adapts the tone, `customer` supplies a returning customer's remembered
    order, `user_text` keeps the reply coherent with what was just said."""
    last = (customer or {}).get("last_order")
    last = nice_name(last) if last else ""
    fallback = DEFAULTS[situation].format(drink=drink, last=last)
    if not USE_LLM:
        return fallback
    try:
        from config import OLLAMA_MODEL, OLLAMA_URL
        parts = [_bar_facts(), "",
                 _LLM_PROMPTS[situation].format(drink=drink, last=last)]
        if user_text:
            parts.append(
                f'The customer just said: "{user_text.strip()}". Respond to that '
                "specifically and naturally; if it is unclear or off-menu, say so "
                "kindly and ask a short question instead of guessing or inventing.")
        if mood:
            parts.append(
                f"The customer seems {mood.get('emotion', 'neutral')} "
                f"({mood.get('sentiment', 'neutral')} mood). "
                f"{mood.get('empathy_note', '')} Match their mood in your tone, "
                "but stay natural and never say that you analysed them.")
        if _recent_lines:
            parts.append(
                "Vary your wording — do NOT reuse these phrasings you just used: "
                + " / ".join(f'"{l}"' for l in _recent_lines[-3:]) + ".")
        parts.append(
            "Reply with ONLY what you say out loud: warm, natural and "
            "conversational, 1-2 short sentences, plain text, no quotes, no "
            "narration, no emoji.")
        prompt = "\n".join(parts)

        temps = (0.35, 0.15) if situation == "decline" else (LLM_TEMP, 0.2)
        for temp in temps:
            r = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "keep_alive": "10m",
                      "options": {"num_predict": 110, "temperature": temp}},
                timeout=40)
            text = _clean_line(r.json()["response"])
            if not (text and len(text) <= 240 and _grounded(text, user_text)):
                continue
            if situation == "decline" and not _refuses(text):
                continue
            _recent_lines.append(text)
            del _recent_lines[:-5]
            return text
    except Exception:
        pass
    return fallback

def prewarm_llm():
    """Load the LLM into memory in the background so the first real call isn't a
    cold model load. Fired while the robot is still driving to the table."""
    if not USE_LLM:
        return
    def _go():
        try:
            from config import OLLAMA_MODEL, OLLAMA_URL
            requests.post(OLLAMA_URL,
                          json={"model": OLLAMA_MODEL, "prompt": "ok",
                                "stream": False, "keep_alive": "10m",
                                "options": {"num_predict": 1}},
                          timeout=60)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF\U00002190-\U000021FF️]")

# Ritmo del parlato (parole/min per il TTS): con un cliente di fretta il robot
# parla PIU' SVELTO e resta così finché non si passa al cliente successivo
# (set_speech_pace lo rialza/riabbassa dal mood). None = ritmo di default.
_SPEECH_WPM = None
_SPEECH_WPM_BY_URGENCY = {"high": 220, "medium": None, "low": 165}

def set_speech_pace(urgency):
    """Lega la velocità di parlato all'urgenza: fretta -> più veloce."""
    global _SPEECH_WPM
    _SPEECH_WPM = _SPEECH_WPM_BY_URGENCY.get(str(urgency).lower(), None)

def say(text):
    """The robot speaks (English TTS). Cross-platform: macOS `say`, else Linux
    `spd-say`/`espeak`. Il ritmo segue _SPEECH_WPM (fretta = più veloce). Falls
    back to text-only if no TTS engine is installed."""
    text = _EMOJI_RE.sub("", text).strip()
    print("TIAGo:", text)
    log_event("💬", text)
    wpm = _SPEECH_WPM
    mac = ["say", "-v", VOICE] + (["-r", str(wpm)] if wpm else []) + [text]
    esp = ["espeak"] + (["-s", str(wpm)] if wpm else []) + [text]
    for cmd in (mac, ["spd-say", "-w", text], esp):
        try:
            subprocess.run(cmd, check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

def nice_name(target):
    for k, v in DISPLAY.items():
        if k in target:
            return v
    return target

def send_command(action, target="", mood=None, grounding=None, table=None,
                 place_pos=None, delta=None, angle=None):
    cmd = {"command_id": datetime.now().isoformat(timespec="seconds"),
           "action": action, "target": target, "status": "ready"}
    if delta is not None:
        cmd["delta"] = float(delta)
    if angle is not None:
        cmd["angle"] = float(angle)
    if place_pos:
        cmd["place_pos"] = list(place_pos)
    if table:
        cmd["table"] = {"name": table.get("name"),
                        "person": list(table["person"]),
                        "serve": list(table["serve"])}
    if mood:
        cmd["mood"] = {k: mood.get(k) for k in
                       ("emotion", "sentiment", "sentiment_score", "urgency")}
    if grounding:
        cmd["grounding"] = grounding
    os.makedirs(SHARED, exist_ok=True)
    with open(COMMAND_FILE, "w") as f:
        json.dump(cmd, f)
    return cmd["command_id"]

def serve_urgency(mood):
    """Map a mood to how urgently to serve. An unhappy customer gets the same priority
    as an explicitly urgent one. Mirrors the speed logic in hri_bridge.py, so the
    decision is the same with or without the simulator."""
    if not mood:
        return "medium"
    urgency = str(mood.get("urgency", "medium")).lower()
    if str(mood.get("sentiment", "")).lower() == "negative" and urgency != "high":
        urgency = "high"
    return urgency if urgency in ("high", "medium", "low") else "medium"

# Il LLM a volte scambia un saluto/congedo ("thank you, bye") per fretta. Alziamo
# l'urgenza dal giudizio del LLM SOLO se il testo contiene un vero segnale di
# fretta; la prosodia (loud+fast) resta un canale indipendente. Evita che con un
# cliente tranquillo il robot acceleri voce/servizio senza motivo.
_HURRY_RE = re.compile(
    r"\b(hurr\w+|in a rush|rush\w*|quick\w*|faster|hurry up|as ?ap|no time|"
    r"do ?n'?t have time|running late|i'?m late|make it quick|"
    r"fretta|in fretta|veloc\w*|di corsa|sbrigat\w*|poco tempo|di fretta)\b",
    re.I)

def _text_suggests_hurry(text):
    return bool(_HURRY_RE.search(text or ""))

def speech_urgency(mood):
    """Ritmo di PARLATO (diverso da serve_urgency, che regge priorità + corsa).
    La voce accelera SOLO per una fretta genuina: cliente urgente ma
      • NON arrabbiato/negativo -> con l'arrabbiato la voce resta calma/pacata
        (accelerare suonerebbe sbrigativo e peggiorerebbe le cose);
      • NON felice/rilassato -> un cliente sereno non va "sbrigato" (doppia rete
        contro eventuali falsi 'in a hurry').
    Priorità e velocità di MOVIMENTO restano su serve_urgency in entrambi i casi."""
    if not mood:
        return "medium"
    emotion = str(mood.get("emotion", "")).lower()
    sentiment = str(mood.get("sentiment", "")).lower()
    if emotion in ("angry", "happy") or sentiment in ("negative", "positive"):
        return "medium"
    return "high" if str(mood.get("urgency", "medium")).lower() == "high" else "medium"

def _proxemics_delta(mood):
    """Metres to move closer (+) or back (-) by mood, deliberately only a few cm. Angry
    gives more space, happy comes a touch closer, anything else leaves the standard
    approach distance alone."""
    if not mood:
        return 0.0
    emotion = str(mood.get("emotion", "")).lower()
    if emotion == "angry" or serve_urgency(mood) == "high":
        return -0.15
    if emotion == "happy":
        return 0.10
    return 0.0

def _proxemics_angle(mood):
    """Radians to angle the approach away from dead-on by mood (Hall's proxemic zones):
    a distressed person tends to read a square-on approach as confrontational, while a
    relaxed one is comfortable with it. About 20 degrees — a real cue without turning
    the robot away from the table."""
    if not mood:
        return 0.0
    emotion = str(mood.get("emotion", "")).lower()
    if emotion in ("angry", "tired") or serve_urgency(mood) == "high":
        return math.radians(20)
    return 0.0

def wait_status(states, timeout=400, want_id=None):
    """Wait until status.json reaches one of the given states.
    
    With `want_id`, only a status written for that command_id counts, so stale status
    from an earlier run cannot make the robot look like it already arrived."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(STATUS_FILE) as f:
                d = json.load(f)
            if d.get("state") in states and (
                    want_id is None or d.get("command_id") == want_id):
                return d.get("state")
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        time.sleep(1.0)
    return None

_WHISPER_NOISE = {"you", "thank you", "thanks", "thanks for watching",
                  "thank you for watching", "bye", "so", "okay", "ok", "uh",
                  "um", "mm", "hmm", "please subscribe", "subtitles", "."}

def _is_whisper_noise(text):
    t = text.strip().lower().strip(" .!?,")
    return len(t) < 2 or t in _WHISPER_NOISE

def _apply_voice_tone(mood, audio, seconds, text):
    """Tone of voice from the same audio Whisper just transcribed: loudness (RMS) and
    speaking rate. Not a trained speech-emotion model, just two features computed on
    data already captured for STT — the facial and text signals only look at what was
    said, never at how.
    
    Thresholds are uncalibrated. Loud and fast reads as rushed; quiet and slow covers
    both tired and hesitant, which call for the same gentle response anyway."""
    energy = float(np.sqrt(np.mean(np.square(audio))))
    rate = len(text.split()) / max(seconds, 1.0)
    loud, quiet = energy > 0.04, energy < 0.01
    fast, slow = rate > 2.5, rate < 1.0
    voice_label = ("agitated/rushed" if loud and fast else
                  "tired/hesitant" if quiet and slow else "normal")
    print(f"🎙️  voice: energy={energy:.4f} rate={rate:.2f}w/s "
          f"-> {voice_label} (current mood emotion: "
          f"{mood.get('emotion', 'none')})")
    if loud and fast and mood.get("urgency") != "high":
        mood["urgency"] = "high"
        mood.setdefault("emotion", "angry")
        print("🎙️  Voice sounds loud and fast — reading as agitated/rushed.")
    elif quiet and slow and not mood.get("emotion"):
        mood["emotion"] = "tired"
        print("🎙️  Voice sounds quiet and slow — reading as tired/hesitant, softening tone.")

def listen_en(seconds=5, text_input=False, mood=None):
    """Get the customer's utterance — typed (rehearsal) or from the mic+Whisper.
    `mood`, if given, is updated in place with a rough tone-of-voice read from
    the SAME audio (see _apply_voice_tone) — voice mode only, no extra recording."""
    if text_input:
        try:
            text = input("⌨️  Type the customer's order (empty = say nothing): ").strip()
        except EOFError:
            text = ""
        print("🗣️  Customer:", text)
        return text
    from core.language_module import (whisper_model, record_audio,
                                      MicUnavailable)
    import scipy.io.wavfile as wav
    try:
        audio, fs = record_audio(seconds=seconds)
    except MicUnavailable:
        print("🎤 Microphone unavailable on this machine right now "
              "— type this turn instead (empty = say nothing).")
        try:
            text = input("⌨ Customer: ").strip()
        except EOFError:
            text = ""
        print("🗣  Customer:", text)
        return text
    if len(audio) < fs:
        print("🗣️  Customer: (silence)")
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        wav.write(tmp.name, fs, audio)
        res = whisper_model.transcribe(
            tmp.name, language="en", fp16=False, temperature=0.0,
            condition_on_previous_text=False, no_speech_threshold=0.6,
            initial_prompt="Order at a bar: Coca-Cola, Sprite, water, orange "
                           "juice, wine, Pringles, the usual, bathroom, "
                           "yes, no, thanks.")
    text = res.get("text", "").strip()
    if _is_whisper_noise(text):
        text = ""
    print("🗣️  Customer:", text or "(silence)")
    if mood is not None and text:
        _apply_voice_tone(mood, audio, seconds, text)
    return text

_QUICK_DRINK = {"cocacola": "coca cola", "coke": "coca cola", "cola": "coca cola",
                "coca": "coca cola", "sprite": "sprite", "water": "water",
                "juice": "juice", "orange": "juice", "pringles": "pringles",
                "chips": "pringles", "crisps": "pringles",
                "wine": "wine", "vino": "wine"}

def understand(text):
    """Understand the requested drink with the LLM."""
    from core.language_module import parse_user_command_with_llm
    scene = "Drinks available at the bar: coca cola, sprite."
    try:
        parsed = parse_user_command_with_llm(text, scene)
    except Exception as e:
        print("understand() error:", e)
        return ""
    return str(parsed.get("target", "")).lower().strip()

def is_drink(target):
    return target in DRINK_WORDS or any(w in target for w in DRINK_WORDS)

def _remember(customer, target):
    """Persist an order to the customer's history (their 'usual' next time)."""
    if not customer:
        return
    try:
        from core.face_id import remember_order
        remember_order(customer["id"], target)
        customer["last_order"] = target
    except Exception:
        pass

def read_face_and_mood(text_input):
    """Turn on the webcam: read the face + emotion (vision model). Returns
    (facial, mood). Identity is NOT taken from the face here — with one laptop
    webcam it can't tell apart people at different tables; the caller resolves
    identity from WHICH table the robot is at (the scan)."""
    facial = None
    try:
        from core.emotion_module import read_facial_expression, EMOTION_ENABLED
        if EMOTION_ENABLED:
            facial = read_facial_expression()
    except Exception as e:
        print("Emotion sensing unavailable (continuing on speech only):", e)

    mood = None
    if facial and facial.get("best_frame") is not None:
        try:
            from core.emotion_module import VISION_MODEL, analyze_sentiment
            if VISION_MODEL:
                print(f"👁️  Reading your expression with {VISION_MODEL}...")
                mood = analyze_sentiment(facial, "")
                if mood:
                    print(f"👁️🧠 mood: {mood.get('emotion')} "
                          f"({mood.get('sentiment')})")
        except Exception as e:
            print("(vision mood unavailable:", e, ")")
    if mood is None and facial and facial.get("face_found"):
        emo = facial.get("emotion") or (
            "happy" if facial.get("expression") == "smiling" else "neutral")
        positive = emo in ("happy", "surprised")
        negative = emo in ("sad", "angry", "disgust", "fear")
        mood = {"emotion": emo,
                "sentiment": "positive" if positive else
                             "negative" if negative else "neutral",
                "empathy_note": ""}

    return facial, mood

def identify_customer(entity, facial):
    """Who is the robot serving at this table? Prefer the table's occupant from
    the scan (a stable per-table identity); fall back to webcam face recognition
    only when there is no table context (single-customer use)."""
    customer = None
    if entity:
        try:
            from core.face_id import customer_by_key
            customer = customer_by_key(entity)
        except Exception as e:
            print("(customer profile unavailable:", e, ")")
    elif facial and facial.get("best_frame") is not None:
        try:
            from core.face_id import recognize_customer
            customer = recognize_customer(facial["best_frame"])
        except Exception as e:
            print("(face recognition unavailable:", e, ")")
    if customer:
        customer["is_minor"] = entity in MINOR_ENTITIES
        customer["age_uncertain"] = False
    if WEBCAM_AGE and customer and facial and facial.get("best_frame") is not None:
        try:
            from core.face_id import estimate_age
            bucket, conf, minor = estimate_age(facial["best_frame"])
            if bucket is not None:
                customer["age_estimate"] = bucket
                customer["age_confidence"] = conf
                customer["is_minor"] = minor
                customer["age_uncertain"] = (bucket == "(15-20)")
                print(f"🎂 Estimated age bracket: {bucket} (confidence {conf:.0%})")
        except Exception as e:
            print("(age estimation unavailable:", e, ")")
    if customer and not customer.get("is_new"):
        print(f"👤 Returning customer {customer['id']} — visit "
              f"#{customer['visits']}, last order: {customer.get('last_order')}"
              f"{'  [MINOR]' if customer.get('is_minor') else ''}")
    elif customer:
        print(f"👤 New customer: {customer['id']}"
              f"{'  [MINOR]' if customer.get('is_minor') else ''}")
    return customer

def scene_grounding_text():
    """Menu + which drinks are actually available to serve right now, as a CLEAN
    summary. The conversational agent must NEVER receive raw scene-graph node
    names/IDs (e.g. "red bottle 32", "table 1") — the model parrots them into
    speech. So we expose only nice item names, never the internal graph dump."""
    scene = ("Bar menu TIAGo can fetch and serve, each on a counter: Coca-Cola, "
             "Sprite, water, orange juice, wine (adults only), and Pringles chips. "
             "TIAGo grasps one item at a time and carries it to the customer. "
             "No cooking, no hot food.")
    try:
        from graph.vlm_perception import DRINK_LAYOUT
        sg = _GRAPH.get("sg")
        names = sorted({nice_name(n.label) for n in sg.nodes.values()
                        if n.label in DRINK_LAYOUT}) if sg else []
        if names:
            print("🕸️  Using live scene graph for grounding.")
            scene += "\nCurrently available on the counter right now: " + \
                     ", ".join(names) + "."
        elif _GRAPH.get("counter_scanned"):
            scene += ("\nYou HAVE looked at the counter and it is empty right "
                      "now: there is nothing you can actually fetch, so do not "
                      "promise any item.")
        else:
            scene += ("\nYou have NOT managed to look at the counter yet, so you "
                      "do not know what is in stock. Do not promise any specific "
                      "item as if it were available: say honestly that you still "
                      "have to check the bar.")
    except Exception:
        pass
    return scene

_TONE_LABEL = {
    "angry": "calm/reassuring tone", "sad": "warm/encouraging tone",
    "happy": "upbeat/playful tone", "surprised": "calm/clear tone",
    "tired": "gentle/efficient tone",
}

_TONE_BY_EMOTION = {
    "angry": "This customer seems ANGRY or frustrated — speak in a calm, steady, "
             "reassuring tone, briefly acknowledge it without dwelling on it, and "
             "get straight to helping them (no extra chit-chat).",
    "sad": "This customer seems SAD or down — be warm and gentle, and say "
           "something genuinely kind or encouraging to lift their mood before "
           "moving on to the order (one short caring remark, not overdone).",
    "happy": "This customer seems HAPPY and cheerful — match their energy: be "
              "upbeat, warm and a little playful in your tone.",
    "surprised": "This customer seems surprised — be clear, calm and reassuring.",
    "tired": "This customer seems tired — be gentle and efficient, keep the "
             "small talk light so you don't tire them further.",
}

def _agent_conversation(customer, mood, rehearse, text_input):
    """LLM-driven dialogue for ONE customer. The agent reasons
    over the running conversation and decides each turn's action; here we just
    speak its reply and carry out the action. Returns the list of drinks ordered."""
    from core.dialogue_agent import WaiterAgent
    agent = WaiterAgent()

    minor = bool(customer and customer.get("is_minor"))
    age_uncertain = bool(customer and customer.get("age_uncertain"))
    ctx = []
    if age_uncertain:
        ctx.append("This customer's age could not be told for certain (borderline "
                   "around 18) — if they ask for wine, politely ask them to confirm "
                   "they are over 18 before serving it.")
    elif minor:
        ctx.append("This customer is a CONFIRMED CHILD: never serve alcohol, no "
                   "need to ask their age — if they ask, refuse warmly and offer a "
                   "soft drink instead.")
    else:
        ctx.append("This customer is a CONFIRMED ADULT (their age is already known, "
                   "not something to verify) — if they ask for wine, serve it "
                   "directly, do NOT ask them to confirm their age.")
    if mood:
        tone = _TONE_BY_EMOTION.get(mood.get("emotion", "neutral"))
        if tone:
            ctx.append(tone)
        else:
            ctx.append(f"They seem {mood.get('emotion', 'neutral')} "
                       f"({mood.get('sentiment', 'neutral')} mood) — match their "
                       "tone naturally.")
    if not customer or customer.get("is_new"):
        ctx.append("This is this customer's FIRST EVER visit — greet them as "
                   "someone you're meeting for the first time, do not claim "
                   "to remember them or reference a past visit.")
    elif customer.get("last_order"):
        ctx.append(f"This is a RETURNING customer you remember — if they ask "
                   f"for 'the usual', that is {nice_name(customer['last_order'])}.")
    else:
        ctx.append("This is a RETURNING customer you remember, though they "
                   "didn't order anything on their last visit (no 'usual' to "
                   "offer) — greet them as someone you recognise.")
    ctx.append(scene_grounding_text())
    context = " ".join(ctx)

    opener, _, _, _, _ = agent.turn("[A customer just sat down at your table. Greet "
                                    "them and ask what they'd like.]", context)
    say(opener)

    targets, empties = [], 0
    while len(targets) < 6:
        text = listen_en(text_input=text_input, mood=mood)
        if not text:
            empties += 1
            if targets:
                break
            if empties >= 3:
                say(llm_reply("no_order", mood=mood, customer=customer))
                break
            continue
        if text_input and text.strip().lower() == "d":
            drop_drink()
            continue
        reply, action, items, customer_mood, customer_urgency = agent.turn(text, context)
        if mood is not None:
            if (customer_urgency == "high" and mood.get("urgency") != "high"
                    and _text_suggests_hurry(text)):
                mood["urgency"] = "high"
                print(f"⏱️  Customer in a hurry (LLM + segnale nel testo): {text!r}")
            if customer_mood == "angry" and mood.get("emotion") != "angry":
                mood["urgency"] = "high"
                mood["emotion"] = "angry"
                print(f"😠 LLM read the customer's tone as angry/dismissive: {text!r}")
            set_speech_pace(speech_urgency(mood))
        if action == "bathroom":
            handle_bathroom_request(mood, rehearse)
            continue
        say(reply)
        if action == "serve" and items:
            for item in items:
                targets.append(item)
                _remember(customer, item)
        elif action in ("reception", "help"):
            _point_to_reception(mood, rehearse)
        elif action == "end":
            break
    return targets

def take_order_at_table(table, rehearse, text_input, entity=None):
    """Drive to one table, read the customer, and collect ONE OR MORE drinks.
    `entity` is the table's occupant from the scan (its stable identity).
    Returns (customer, [targets], mood) or None if nobody ordered."""
    say(f"I'm coming to {table['name']}.")
    if not rehearse:
        approach_table = table
        if entity in MINOR_ENTITIES and table.get("person"):
            px, py, pyaw = table["person"]
            extra = 0.15
            approach_table = dict(table, person=(
                px - extra * math.cos(pyaw), py - extra * math.sin(pyaw), pyaw))
        aid = send_command("approach", table=approach_table)
        print(f"   (driving to {table['name']}...)")
        if wait_status({"at_customer", "failed"}, timeout=NAV_WAIT,
                       want_id=aid) != "at_customer":
            say("Sorry, I could not reach that table.")
            return None

    facial, mood = read_face_and_mood(text_input)
    if mood is None:
        mood = {}
    customer = identify_customer(entity, facial)

    vlm_state = _TABLE_VLM_STATE.get(table.get("id"))
    if vlm_state == "needs-attention" and mood.get("urgency") != "high":
        mood["urgency"] = "high"
        print(f"👁️  Gemini's visual read of {table['name']} suggests the "
              f"customer needs attention — raising priority.")
        log_event("👁️", f"{table['name']}: visually inferred needs-attention "
                  "— priority raised")

    # Ritmo di parlato per QUESTO cliente: si resetta a ogni cliente e sale se ha
    # fretta / va assistito (viene poi aggiornato ad ogni turno nella conversazione).
    set_speech_pace(speech_urgency(mood) if mood else "medium")

    delta = _proxemics_delta(mood) if (mood and PROXEMICS_ENABLED) else 0.0
    angle = _proxemics_angle(mood) if (mood and PROXEMICS_ENABLED) else 0.0
    if not rehearse and (delta or angle):
        pid = send_command("proxemics", delta=delta, angle=angle)
        wait_status({"adjusted", "failed"}, timeout=30, want_id=pid)

    if mood and mood.get("emotion") and PROXEMICS_ENABLED:
        emotion = str(mood.get("emotion")).lower()
        tone_note = _TONE_LABEL.get(emotion, "neutral tone")
        if delta > 0:
            move_note = f"moved {delta*100:.0f}cm closer"
        elif delta < 0:
            move_note = f"backed off {abs(delta)*100:.0f}cm"
        else:
            move_note = "no distance change"
        angle_note = f", angled {math.degrees(angle):.0f}° off-frontal" if angle else ""
        line = (f"{table['name']}: read mood '{emotion}' "
               f"({mood.get('sentiment', 'neutral')}) → {tone_note}, {move_note}"
               f"{angle_note}")
        print(f"🎭 {line}")
        log_event("🎭", line)

    targets = _agent_conversation(customer, mood, rehearse, text_input)
    if not targets:
        return None
    say("Great, I've got your order.")
    return customer, targets, mood

def suggest_substitute(target):
    """Commonsense substitution (HRAI 7): the ordered item is not in the scene
    graph, so propose the closest SERVABLE menu item the robot can actually
    see. Same ontology class first (a drink for a drink), then name
    similarity. Returns (alt_target, alt_node) or (None, None)."""
    kg = _GRAPH["kg"]
    if kg is None:
        return None, None
    try:
        from graph.similarity import word_similarity
    except Exception:
        return None, None

    def klass(item):
        for k in ("Drink", "Snack"):
            if kg.ask(item, "isA", k):
                return k
        return None

    best, best_node, best_score = None, None, -1.0
    for alt in DRINK_APPEARANCE:
        if alt == target:
            continue
        ok, _ = kg.can_perform("ServeItem", alt)
        if not ok:
            continue
        node = ground_order(alt)
        if node is None:
            continue
        score = word_similarity(alt, target)
        if klass(alt) == klass(target):
            score += 1.0
        prefs = INTENT_PRIOR.get(target, ("", []))[1]
        if alt in prefs:
            score += 0.8 - 0.2 * prefs.index(alt)
        for e in similar_episodes("substitution", item=target):
            if e.get("substitute") == alt:
                score += 0.3 if e.get("accepted") else -0.3
        if score > best_score:
            best, best_node, best_score = alt, node, score
    return best, best_node

def report_possible_spill(node):
    """Causal-chain reasoning (HRAI 8 wet-floor example): a grasp that failed
    right at the bottle may have knocked it over. Assert the event in the KG,
    chase the 'causes' chain (spilledDrink -> floorWet -> floorSlippery ->
    hazardForHumans), mark a hazard node in the scene graph (dashboard shows
    ⚠), and warn the customers. Returns the derived effects."""
    kg, sg = _GRAPH.get("kg"), _GRAPH.get("sg")
    if kg is None or sg is None or node is None or not node.centroid:
        return []
    effects = kg.propagate_causes("spilledDrink")
    if "hazardForHumans" not in effects:
        return effects
    kg.add("spilledDrink", "occurredNear", node.node_id, abox=True)
    hid = f"hazard_{node.node_id}"
    if hid not in sg.nodes:
        try:
            from graph.object_node import ObjectNode
            x, y = node.centroid[0], node.centroid[1]
            sg.add_node(ObjectNode(
                node_id=hid, label="hazard", centroid=(x, y, 0.0),
                frame=getattr(node, "frame", "map"), material="liquid",
                shape="spill", is_movable=False, source="causal_rule",
                depth_source="prior", affordances=("avoided", "cleanedUp"),
                description=f"possible spill where the grasp of "
                            f"{node.node_id} failed"))
            sg.save(SCENE_GRAPH_FILE)
        except Exception as e:
            print("(could not add hazard node:", e, ")")
        print(f"🕸️  Causal chain: spilledDrink -> {' -> '.join(effects)}")
        chain = ["spilledDrink"] + list(effects)
        log_event("🕸️", "causal chain fired: " + " → ".join(chain),
                  triples=[[chain[i], "causes", chain[i+1]]
                           for i in range(len(chain) - 1)])
        drop_drink(pos=(node.centroid[0], node.centroid[1]))
    return effects

def diagnose_serve_failure(node):
    """Explain WHY a fetch failed from the graph state (HRAI 7 'recognizing
    failures': lost sight / object moved / positioning problem) and demote the
    failed node so a retry grounds to a different physical object."""
    sg = _GRAPH.get("sg")
    if node is None or sg is None:
        return ("I was working from the fixed bar layout and the fetch "
                "did not succeed.")
    current = sg.nodes.get(node.node_id)
    if current is None:
        return (f"I lost track of the {node.label} — it disappeared from my "
                "scene graph while I was on my way.")
    if current.status == "uncertain":
        return (f"I lost sight of the {node.label} — it was not where I "
                "remembered it.")
    current.status = "uncertain"
    try:
        sg.save(SCENE_GRAPH_FILE)
    except Exception:
        pass
    return (f"I could not complete the grasp — probably a positioning "
            f"problem in front of the {node.label}.")

def _grounding_from(node):
    """command.json grounding payload for a scene-graph node."""
    pretty = node.label if node.color_name in node.label \
        else f"{node.color_name} {node.label}".strip()
    print(f"🕸️  Order grounded in the scene graph: {node.node_id} ({pretty})")
    return {"node_id": node.node_id, "label": node.label,
            "color": node.color_name, "centroid": node.centroid,
            "frame": getattr(node, "frame", "camera"),
            "depth_source": node.depth_source,
            "status": node.status, "times_seen": node.times_seen}

def serve_one(target, mood, table, rehearse, text_input=False):
    """Fetch one drink and deliver it to the given table. True on success.
    
    Two behaviours on top of the plain fetch: commonsense substitution offers a similar
    servable item when the ordered one is not in the scene graph, and a failed fetch is
    diagnosed from the graph state, explained aloud, and retried once on a different
    node with the same appearance."""
    drink = nice_name(target)
    servable, unmet = order_is_servable(target)
    if not servable:
        print(f"🕸️  KG veto: {unmet}")
        log_event("⛔", f"KG veto on {drink}: " + "; ".join(unmet),
                  triples=[["ServeItem", "hasPrecondition", "targetIsDrinkOrSnack"],
                           [target, "fails", "precondition"]])
        say(f"I'm sorry, I cannot serve {drink}: " + "; ".join(unmet))
        return False

    node = ground_order(target)

    _awareness_decision = {"action": "SERVE", "item": target}
    _measured = refine_decision_with_graphs(_awareness_decision)
    log_awareness(f"serve {drink}", _awareness_decision, _measured)

    if node is not None and node.centroid:
        bad = [e for e in similar_episodes("serve", item=target,
                                           near=node.centroid[:2])
               if e.get("valence", 0) < 0]
        if len(bad) >= 2:
            other = ground_order(target, exclude={node.node_id})
            if other is not None:
                print(f"🗂️  Case memory: {len(bad)} past failures near "
                      f"{node.node_id} — grounding to {other.node_id} instead.")
                node = other

    from graph.vlm_perception import DRINK_LAYOUT
    on_counter = target in DRINK_LAYOUT
    if node is None and not on_counter and _GRAPH["sg"] and _GRAPH["sg"].nodes:
        alt, alt_node = suggest_substitute(target)
        if alt:
            why = INTENT_PRIOR.get(target, ("", []))[0]
            offer = (f"Since you wanted {why}, can I bring you "
                     f"{nice_name(alt)} instead?" if why else
                     f"Can I bring you {nice_name(alt)} instead?")
            say(f"I'm sorry, I don't see any {drink} at the bar right now. "
                + offer)
            reply = listen_en(seconds=4, text_input=text_input, mood=mood)
            accepted = is_yes(reply)
            record_episode("substitution", item=target, substitute=alt,
                           accepted=accepted, valence=1 if accepted else -1)
            if accepted:
                target, node = alt, alt_node
                drink = nice_name(target)
                say(f"Great, {drink} it is!")
            else:
                say(f"Alright, I'll still try to find some {drink} for you.")

    urgency = serve_urgency(mood)
    tier = {"high": "quickly", "medium": "at a normal pace", "low": "calmly"}[urgency]
    print(f"⚙️  serving {drink} to {table['name']} ({tier}, urgency={urgency})")

    if rehearse:
        if os.environ.get("WAITER_FAKE_FAIL") == "1" and node is not None:
            reason = diagnose_serve_failure(node)
            print(f"🕸️  (rehearsal) fetch failed: {reason}")
            record_episode("serve", item=target,
                           pos=list(node.centroid[:2]) if node.centroid else None,
                           node_id=node.node_id, outcome="failed",
                           diagnosis=reason, valence=-1)
            if "positioning" in reason:
                report_possible_spill(node)
            retry = ground_order(target, exclude={node.node_id})
            if retry is not None:
                say(f"Sorry — {reason} Let me try another {retry.label} "
                    "I can see.")
                print(f"(rehearsal: retrying on {retry.node_id})")
            else:
                say(f"Sorry — {reason}")
        print(f"(rehearsal: pretending {drink} was served to {table['name']})")
        return True

    tried = set()
    spill_reported = False
    for attempt in (1, 2):
        grounding = None
        if node is not None:
            tried.add(node.node_id)
            grounding = _grounding_from(node)
        bid = send_command("bring", target, mood=mood, grounding=grounding,
                           table=table)
        if attempt == 1 and mood and \
                str(mood.get("sentiment", "")).lower() == "negative":
            say("I'm sorry for the wait — I'll be quick.")
        print("   (robot is fetching and serving — watch the Gazebo window...)")
        st = wait_status({"done", "failed"}, timeout=NAV_WAIT * 2, want_id=bid)
        if st == "done":
            if _SERVING.get("redo"):
                print("   (drink dropped mid-delivery — fetching a fresh one)")
                return False
            record_episode("serve", item=target,
                           pos=(list(node.centroid[:2])
                                if node is not None and node.centroid else None),
                           node_id=node.node_id if node is not None else None,
                           outcome="done", valence=1)
            say(llm_reply("serve", drink, mood=mood))
            return True
        reason = diagnose_serve_failure(node)
        print(f"🕸️  Fetch failed: {reason}")
        record_episode("serve", item=target,
                       pos=(list(node.centroid[:2])
                            if node is not None and node.centroid else None),
                       node_id=node.node_id if node is not None else None,
                       outcome="failed", diagnosis=reason, valence=-1)
        if _SERVING.get("aborted_by_drop"):
            _SERVING["aborted_by_drop"] = False
            return False
        if node is not None and "positioning" in reason:
            report_possible_spill(node)
            spill_reported = spill_reported or _FLOOR_HAZARD.get("active", False)
            # Il robot si FERMA e aspetta che la caduta sia pulita PRIMA di
            # ritentare o proseguire: senza questo ripartiva col drink ancora a
            # terra (no-op se non è stato registrato nessun pericolo).
            wait_for_floor_clear()
        if attempt == 2:
            break
        node = ground_order(target, exclude=tried)
        if node is None:
            break
        say(f"Sorry — {reason} Let me try another {node.label} I can see.")
    if not spill_reported:
        say("Sorry, there was a problem serving your drink.")
    return False

def add_customers_to_graph(customers=None):
    """Add `table` nodes for every table toured and `person` nodes for the scanned
    customers, linked by `isAtTable`, so the graph knows who is at which table.
    
    Idempotent and called every cycle during tracking, which also covers the bridge
    writing customers_present.json only after the waiter's wait on the explore action
    has already timed out."""
    sg = _GRAPH.get("sg")
    if sg is None:
        return
    try:
        from graph.object_node import ObjectNode
    except Exception:
        return
    added = []
    for tid, t in TABLES.items():
        tnid = f"table_{tid}"
        if tnid not in sg.nodes:
            x, y = t["serve"][:2]
            sg.add_node(ObjectNode(node_id=tnid, label="table",
                        centroid=(x, y, 0.4), frame="map", material="wood",
                        shape="surface", is_movable=False, source="ground_truth",
                        depth_source="prior",
                        affordances=("approached", "usedAsSupport"),
                        description=f"dining table {tid}"))
            added.append(tnid)
    for cnid, (x, y, z) in COUNTERS.items():
        if cnid not in sg.nodes:
            sg.add_node(ObjectNode(node_id=cnid,
                        label=cnid.replace("_", " "),
                        centroid=(x, y, z), frame="map", material="wood",
                        shape="surface", is_movable=False,
                        source="ground_truth", depth_source="prior",
                        affordances=("approached", "usedAsSupport"),
                        description=f"the {cnid.replace('_', ' ')} of the bar"))
            added.append(cnid)
    if customers is None:
        customers = detected_customers()
    for c in customers:
        tid = c.get("table_id")
        x, y = (c.get("pos") or [0.0, 0.0])[:2]
        entity = c.get("entity", tid)
        minor = entity in MINOR_ENTITIES
        pid, tnid = f"customer_{entity}", f"table_{tid}"
        if pid not in sg.nodes:
            sg.add_node(ObjectNode(node_id=pid, label="person",
                        centroid=(x, y, 0.9), frame="map", material="organic",
                        shape="human", is_movable=True, source="ground_truth",
                        depth_source="prior", is_minor=minor,
                        affordances=("greeted", "askedForOrder", "servedTo"),
                        description=(f"a child at table {tid}" if minor
                                     else f"a customer at table {tid}")))
            added.append(pid)
        if tnid in sg.nodes and (pid, "isAtTable", tnid) not in sg.edges:
            sg.edges.append((pid, "isAtTable", tnid))
    if not added:
        return
    try:
        sg.save(SCENE_GRAPH_FILE)
        kg = _GRAPH.get("kg")
        if kg is not None:
            kg.sync_scene_graph(sg)
    except Exception as e:
        print("(could not save scene graph with customers:", e, ")")
    print(f"🕸️  Scene graph: injected {', '.join(added)}.")
    people = [a.replace("customer_", "") for a in added if a.startswith("customer_")]
    if people:
        log_event("🧑", "customers at the tables: " + ", ".join(people))

_COUNTER_MODEL = {"counter_east": "bar_counter_top", "counter_west": "side_counter_top"}
_COUNTER_OFFSET = {"counter_east": (0.25, 0.23, -0.285),
                   "counter_west": (0.0, 0.23, -0.285)}
_COUNTER_FALLBACK_BASE = {"counter_east": (2.0, -3.43), "counter_west": (-1.8, -3.43)}

COUNTERS = {}
for _cid, _model in _COUNTER_MODEL.items():
    _pose = world_pose(_model)
    _bx, _by = _pose[:2] if _pose else _COUNTER_FALLBACK_BASE[_cid]
    _ox, _oy, _oz = _COUNTER_OFFSET[_cid]
    COUNTERS[_cid] = (_bx + _ox, _by + _oy,
                      (_pose[2] if _pose else 0.685) + _oz)
del _cid, _model, _pose, _bx, _by, _ox, _oy, _oz

_DRINK_MODEL = {"coca cola": ("coke_bottle", "red"), "sprite": ("sprite_bottle", "green"),
                "water": ("water_bottle", "blue"), "wine": ("wine_bottle", "purple"),
                "juice": ("juice_bottle", "orange"), "pringles": ("pringles_can", "red")}
_DRINK_FALLBACK_XY = {"coca cola": (2.5, -3.2), "sprite": (2.0, -3.2), "water": (1.5, -3.2),
                      "wine": (3.0, -3.2), "juice": (-2.1, -3.2), "pringles": (-1.5, -3.2)}

COUNTER_DRINKS = []
for _name, (_model, _color) in _DRINK_MODEL.items():
    _pose = world_pose(_model)
    _bx, _by = _pose[:2] if _pose else _DRINK_FALLBACK_XY[_name]
    COUNTER_DRINKS.append((_name, (_bx, _by, 0.85), _color))
del _name, _model, _color, _pose, _bx, _by

def add_counter_drinks_to_graph():
    """Rehearsal-only: seed the known bar layout into the graph when there's no
    live perception to populate it (offline demo, no bridge). In a live run the
    counter drinks come from the VLM perception pass instead (see
    start_scene_graph()'s loop) — this is a no-op there. Idempotent: a drink
    already in the graph is left as-is (persistence)."""
    sg = _GRAPH.get("sg")
    if sg is None or not _GRAPH.get("rehearse"):
        return
    try:
        from graph.object_node import ObjectNode
    except Exception:
        return
    items = [(n, p, c) for n, p, c in COUNTER_DRINKS]
    src = "patrol"
    added, new_names = 0, []
    for name, pos, color in items:
        x, y, z = (list(pos) + [0.85])[:3]
        nid = f"drink_{name.replace(' ', '_')}"
        if nid in sg.nodes:
            continue
        shape = "can" if name == "pringles" else "bottle"
        sg.add_node(ObjectNode(node_id=nid, label=name,
                    centroid=(x, y, z), frame="map", color_name=color,
                    material="glass", shape=shape, is_movable=True,
                    source=src, depth_source="prior",
                    description=f"{name} ({color} {shape}) seeded for rehearsal"))
        added += 1
        new_names.append(name)
    if added:
        print(f"🕸️  Rehearsal: +{added} drink(s) seeded in the graph.")
        log_event("🍾", "rehearsal: seeded on the counter: " + ", ".join(new_names))

def run_round(rehearse, text_input):
    """Take orders from several customers (each at their own table), remember what
    each ordered, then bring the drinks one by one to the right table."""
    orders = []
    add_customers_to_graph(detected_customers())
    add_counter_drinks_to_graph()
    entity_by_table = {c["table_id"]: c["entity"] for c in detected_customers()}
    tour = detected_tables() or list(TABLE_ORDER)
    tour = [t for t in TABLE_ORDER if t in tour]
    print(f"🕸️  Customers known from the scan — visiting tables: {tour}")
    for tid in tour:
        table = TABLES[tid]
        entity = entity_by_table.get(tid, f"table_{tid}")
        res = take_order_at_table(table, rehearse, text_input, entity=entity)
        if res is not None:
            customer, drinks, mood = res
            orders.append({"customer": customer, "table": table,
                           "drinks": drinks, "mood": mood})
            print(f"📝 {table['name']}: {[nice_name(d) for d in drinks]}")
            log_event("📝", f"order at {table['name']}: "
                      + ", ".join(nice_name(d) for d in drinks))
            log_mood(entity, mood)

    if not orders:
        say("No orders to bring right now. See you!")
        return

    if PRIORITY_REORDER_ENABLED:
        _URGENCY_RANK = {"high": 0, "medium": 1, "low": 2}
        orders.sort(key=lambda od: _URGENCY_RANK.get(serve_urgency(od["mood"]), 1))
    order_str = " -> ".join(f"{od['table']['name']} ({serve_urgency(od['mood'])})"
                            for od in orders)
    print(f"🧾 Serving order: {order_str}")
    log_event("🧾", f"serving order: {order_str} "
              f"(priority_reorder={'on' if PRIORITY_REORDER_ENABLED else 'off'})")
    if orders and serve_urgency(orders[0]["mood"]) == "high":
        print(f"⏱️  Serving {orders[0]['table']['name']} first (priority: urgent/hurried).")

    total = sum(len(o["drinks"]) for o in orders)
    say(f"Alright! Let me bring the {total} "
        f"{'drink' if total == 1 else 'drinks'} now.")
    for od in orders:
        who = od["table"]["name"]
        unfulfilled = []
        for target in od["drinks"]:
            while True:
                wait_for_floor_clear()
                say(f"Now serving {nice_name(target)} to {who}.")
                _SERVING.update(active=True, target=target, table=od["table"],
                                redo=False)
                ok = serve_one(target, od["mood"], od["table"], rehearse, text_input)
                _SERVING["active"] = False
                log_event("✅" if ok else "⚠️",
                          f"{nice_name(target)} → {who}: "
                          + ("served" if ok else "not served"))
                if not _SERVING["redo"]:
                    break
                _SERVING["redo"] = False
                wait_for_floor_clear()
                say(f"Oh no — the {nice_name(target)} slipped and fell! "
                    f"Let me fetch a fresh one for {who}.")
            if not ok:
                unfulfilled.append(target)
        if unfulfilled:
            items_str = " and ".join(nice_name(t) for t in unfulfilled)
            say(f"Before I move on — {who}, I wasn't able to bring your "
                f"{items_str} after all. Sorry about that.")
            log_event("🧠", f"belief correction: told {who} that {items_str} "
                      "could not be served, after promising it earlier")
    say("All orders served. Enjoy, everyone!")

    if not rehearse:
        print("   (returning to the starting point...)")
        hid = send_command("home")
        wait_status({"home", "failed"}, timeout=NAV_WAIT, want_id=hid)

_DROPPED = {"n": 0}
DROP_REQUEST_FILE = os.path.join(SHARED, "drop_drink.request")

_FLOOR_HAZARD = {"active": False, "drink_id": None, "floor_id": None, "where": None,
                 "announced": False}
_SERVING = {"active": False, "target": None, "table": None, "redo": False,
            "aborted_by_drop": False}
CLEAN_DONE_FILE = os.path.join(SHARED, "clean_done.request")
CLEANER_DELAY = float(os.environ.get("WAITER_CLEANER_DELAY", "12"))
GZ_DROP_REQUEST = os.path.join(SHARED, "drop_object.request")
GZ_CLEAN_REQUEST = os.path.join(SHARED, "clean_object.request")
ABORT_HOLD_FILE = os.path.join(SHARED, "abort_delivery.request")

def _robot_xy_yaw():
    """(x, y, yaw) del robot da shared/robot_pose.json (lo scrive il bridge ogni
    ~1.5s), o None se non disponibile."""
    try:
        with open(os.path.join(SHARED, "robot_pose.json")) as f:
            d = json.load(f)
        return float(d["x"]), float(d["y"]), float(d.get("yaw", 0.0))
    except Exception:
        return None

def drop_drink(table=None, pos=None):
    """Register a fallen drink: put it on the floor in the scene graph, mark the spot as
    a hazard in the knowledge graph, and announce it. Returns the fallen drink's id.
    
    `pos` overrides the location for a grasp that failed at the counter, where the
    spill is at the bottle's own position rather than at a customer's table."""
    if _FLOOR_HAZARD.get("active"):
        print("   (a spill is already being handled — ignoring the extra drop.)")
        return _FLOOR_HAZARD.get("drink_id")
    _FLOOR_HAZARD["active"] = True

    sg, kg = _GRAPH.get("sg"), _GRAPH.get("kg")
    _DROPPED["n"] += 1
    idx = _DROPPED["n"]

    if table is None and _SERVING["active"]:
        table = _SERVING["table"]
        _SERVING["redo"] = True
    if pos is not None:
        fx, fy, where = float(pos[0]), float(pos[1]), "the counter"
    elif table is None:
        tid = TABLE_ORDER[0] if TABLE_ORDER else None
        table = TABLES.get(tid) if tid else None
    if pos is None:
        if table and table.get("person"):
            fx, fy = float(table["person"][0]), float(table["person"][1])
            where = table.get("name", "a table")
        else:
            fx, fy, where = 0.0, 0.0, "the bar floor"

    # La bottiglia cade DAL robot: il punto reale della caduta è dove sta il robot
    # ADESSO (poco davanti a lui), non il bancone/tavolo. Così si ferma lì sul
    # posto invece di andarci. Se la posa non è disponibile, resta il fallback sopra.
    _rp = _robot_xy_yaw()
    if _rp is not None:
        _rx, _ry, _ryaw = _rp
        fx = _rx + 0.35 * math.cos(_ryaw)
        fy = _ry + 0.35 * math.sin(_ryaw)
        where = "right in front of me"

    drink_id = f"fallen_drink_{idx}"
    floor_id = f"floor_spot_{idx}"

    if sg is not None:
        try:
            from graph.object_node import ObjectNode
            node = ObjectNode(
                node_id=drink_id, label="drink", confidence=1.0,
                centroid=(round(fx, 2), round(fy, 2), 0.05), frame="map",
                material="glass", shape="bottle", is_movable=True,
                description="a drink that fell and spilled on the floor",
                source="event", depth_source="prior",
                affordances=("cleaned", "picked_up"), status="confirmed")
            sg.add_node(node)
            sg.refresh_relations()
            sg.save(SCENE_GRAPH_FILE)
        except Exception as e:
            print("(drop: scene-graph add failed:", e, ")")

    if kg is not None:
        try:
            kg.add(drink_id, "isA", "Drink", abox=True)
            kg.add(drink_id, "hasStatus", "fallen", abox=True)
            kg.add(drink_id, "isOnTopOf", floor_id, abox=True)
            kg.add(floor_id, "isA", "Floor", abox=True)
            kg.add(floor_id, "isLocatedIn", "Bar", abox=True)
            kg.add(floor_id, "hasPosition", "(%.1f, %.1f)" % (fx, fy), abox=True)
            kg.add(floor_id, "hasHazard", "spilled_drink", abox=True)
            kg.add(floor_id, "hasSpilledDrink", drink_id, abox=True)
            kg.infer()
            with open(os.path.join(SHARED, "knowledge_graph.ttl"), "w") as f:
                f.write(kg.to_turtle())
        except Exception as e:
            print("(drop: knowledge-graph add failed:", e, ")")

    say(f"Attention everyone! A drink has fallen on the floor near {where}. "
        f"Please be careful and mind your step.")
    print(f"🥤 Drink dropped: {drink_id} on the floor at ({fx:.1f}, {fy:.1f}); "
          f"floor spot '{floor_id}' marked as a hazard in the memory graph.")
    log_event("🥤", f"a drink fell near {where} — wet floor hazard, serving paused",
              triples=[["spilledDrink", "causes", "floorWet"],
                       ["floorWet", "causes", "floorSlippery"],
                       ["floorSlippery", "causes", "hazardForHumans"]])

    _FLOOR_HAZARD.update({"active": True, "drink_id": drink_id,
                          "floor_id": floor_id, "where": where, "announced": True})
    if _SERVING.get("active"):
        try:
            with open(ABORT_HOLD_FILE, "w") as f:
                f.write("1")
        except OSError:
            pass
        _SERVING["aborted_by_drop"] = True
    # Spawna SEMPRE la bottiglia caduta in Gazebo, al punto REALE della caduta
    # (prima solo fuori da un serve), così il robot ha davvero qualcosa davanti.
    try:
        with open(GZ_DROP_REQUEST, "w") as f:
            json.dump({"x": fx, "y": fy}, f)
    except OSError:
        pass
    # La bottiglia è caduta QUI, ai piedi del robot: NON naviga da nessuna parte,
    # resta fermo sul posto (dopo un grasp fallito è già idle) e aspetta la pulizia.
    say("The drink slipped and fell right here — I'll stop and wait by it until "
        "the floor is clean.")
    print(f"🧑‍🔧 Cleaner dispatched (auto in ~{CLEANER_DELAY:.0f}s, or now with "
          f"`touch {CLEAN_DONE_FILE}`).")
    threading.Thread(target=_cleaner_thread,
                     args=(drink_id, floor_id, where), daemon=True).start()
    return drink_id

def _cleaner_thread(drink_id, floor_id, where):
    """The 'signore delle pulizie': waits for the cleaner to arrive (a short
    delay, or immediately if CLEAN_DONE_FILE is touched), then wipes the spill."""
    deadline = time.time() + CLEANER_DELAY
    while time.time() < deadline:
        if os.path.exists(CLEAN_DONE_FILE):
            try:
                os.remove(CLEAN_DONE_FILE)
            except OSError:
                pass
            break
        time.sleep(0.3)
    clean_spill(drink_id, floor_id, where)

def clean_spill(drink_id=None, floor_id=None, where=None):
    """The cleaner has mopped the spill: the fallen drink DISAPPEARS from both the
    3D scene graph and the knowledge graph, and the floor-hazard gate is lowered
    so the robot can resume serving."""
    log_event("🧹", f"spill near {where or 'the bar'} cleaned — hazard gone, serving resumes")
    if drink_id is None:
        drink_id = _FLOOR_HAZARD.get("drink_id")
        floor_id = _FLOOR_HAZARD.get("floor_id")
        where = _FLOOR_HAZARD.get("where")
    sg, kg = _GRAPH.get("sg"), _GRAPH.get("kg")

    if sg is not None and drink_id:
        try:
            sg.remove_node(drink_id)
            sg.refresh_relations()
            sg.save(SCENE_GRAPH_FILE)
        except Exception as e:
            print("(clean: scene-graph remove failed:", e, ")")

    if kg is not None and drink_id:
        try:
            for hid in (drink_id, floor_id):
                if not hid:
                    continue
                for tr in kg.query(h=hid) + kg.query(t=hid):
                    kg.remove(*tr)
            kg.infer()
            with open(os.path.join(SHARED, "knowledge_graph.ttl"), "w") as f:
                f.write(kg.to_turtle())
        except Exception as e:
            print("(clean: knowledge-graph remove failed:", e, ")")

    try:
        with open(GZ_CLEAN_REQUEST, "w") as f:
            json.dump({"clean": True}, f)
    except OSError:
        pass

    if _FLOOR_HAZARD.get("drink_id") == drink_id:
        _FLOOR_HAZARD.update({"active": False, "drink_id": None,
                              "floor_id": None, "where": None, "announced": False})
    say(f"The floor near {where or 'the bar'} has been cleaned. It's safe now.")
    print(f"🧹 Spill cleaned: {drink_id} removed from the floor.")

def wait_for_floor_clear():
    """Serving gate. If a drink is on the floor the robot must NOT carry the next
    drink to a table: it stops, waits for the cleaner to mop the spill (the fallen
    drink disappears from the graphs), then resumes."""
    if not _FLOOR_HAZARD.get("active"):
        return
    if not _FLOOR_HAZARD.get("announced"):
        where = _FLOOR_HAZARD.get("where") or "the bar floor"
        say(f"I can't serve yet — there's a drink on the floor near {where}. "
            f"I'll wait here until it has been cleaned up.")
        _FLOOR_HAZARD["announced"] = True
    print("⏸️  Serving paused: waiting for the spill to be cleaned...")
    while _FLOOR_HAZARD.get("active"):
        time.sleep(0.3)
    say("Thanks for your patience — let me continue serving.")
    print("▶️  Floor clear — resuming service.")

def _start_drop_hotkey(text_input):
    """Enable the 'drop a drink' trigger. Two ways to fire it:
      - press the 'd' key (single keypress; only when stdin is a free TTY, i.e.
        voice mode — in --text mode stdin is busy reading the typed order);
      - `touch shared/drop_drink.request` from any terminal / any mode.
    Both run in daemon threads so they never block the interaction."""
    if not text_input and sys.stdin and sys.stdin.isatty():
        try:
            import termios, tty, select

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old))

            def _key_loop():
                try:
                    tty.setcbreak(fd)
                    while True:
                        if select.select([sys.stdin], [], [], 0.2)[0]:
                            ch = sys.stdin.read(1)
                            if ch in ("d", "D"):
                                drop_drink()
                except Exception:
                    pass

            threading.Thread(target=_key_loop, daemon=True).start()
            print("⌨️  Hotkey ready: press 'd' to drop a drink on the floor.")
        except Exception as e:
            print("(drop hotkey unavailable:", e, ")")

    def _file_loop():
        try:
            if os.path.exists(DROP_REQUEST_FILE):
                os.remove(DROP_REQUEST_FILE)
        except OSError:
            pass
        while True:
            try:
                if os.path.exists(DROP_REQUEST_FILE):
                    os.remove(DROP_REQUEST_FILE)
                    drop_drink()
            except OSError:
                pass
            time.sleep(0.4)

    threading.Thread(target=_file_loop, daemon=True).start()
    print(f"📩 Or drop via file:  touch {DROP_REQUEST_FILE}")

def main(rehearse=False, text_input=False):
    global _REHEARSE
    _REHEARSE = rehearse
    banner = " [REHEARSAL — no sim/bridge]" if rehearse else ""
    print(f"=== TIAGo Waiter (interaction){banner} ===\n")

    prewarm_llm()

    if not rehearse:
        _SESSION["t0"] = time.time()
    _GRAPH["rehearse"] = rehearse

    try:
        with open(AWARENESS_FILE, "w") as f:
            json.dump({"latest": None, "history": []}, f)
    except OSError:
        pass
    try:
        with open(EVENTS_FILE, "w") as f:
            json.dump({"events": []}, f)
        with open(MOODS_FILE, "w") as f:
            json.dump({}, f)
    except OSError:
        pass
    log_event("🤖", "waiter session started")
    log_event("⚙️", "conditions: priority_reorder=%s, proxemics=%s"
              % ("on" if PRIORITY_REORDER_ENABLED else "off",
                 "on" if PROXEMICS_ENABLED else "off"))

    start_scene_graph()

    # Objective motion metrics for the experimental evaluation (path length,
    # closest approach to each customer, personal-space intrusions). Sampled
    # from the ground-truth pose the bridge publishes; disabled in rehearsal,
    # where the robot does not move.
    if not rehearse:
        try:
            import metrics_logger
            metrics_logger.start(SHARED)
        except Exception as e:
            print("(motion metrics unavailable:", e, ")")

    _start_drop_hotkey(text_input)

    if rehearse:
        print("(rehearsal: skipping the exploration patrol — the graph fills "
              "from whatever frames are on disk)")
        if GRAPH_ENABLED and os.environ.get("WAITER_SEED_GRAPH", "1") != "0":
            seed_demo_graph()
    elif GRAPH_ENABLED and EXPLORE_ENABLED:
        say("Give me a moment to have a look at the bar.")
        explore_id = send_command("explore")
        print("   (robot is patrolling: bathroom first, then the counters...)")
        st = wait_status({"explored", "failed"}, timeout=EXPLORE_WAIT, want_id=explore_id)
        if st != "explored":
            print("⚠️  exploration did not finish (bridge running?) — "
                  "continuing on the ontology priors only")
    register_places_in_kg()
    end_exploration()
    if not rehearse and GRAPH_ENABLED:
        ensure_counter_scanned()
    add_customers_to_graph()
    if _GRAPH["sg"] and _GRAPH["sg"].nodes:
        print(f"🕸️  Bar memorized: {len(_GRAPH['sg'].nodes)} objects "
              "in the scene graph.")

    run_round(rehearse, text_input)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TIAGo waiter interaction")
    ap.add_argument("--no-sim", "--rehearse", dest="rehearse", action="store_true",
                    help="Rehearsal mode: run greet/face/mood/serve without the "
                         "Gazebo sim or robot bridge")
    ap.add_argument("--text", dest="text_input", action="store_true",
                    help="Type the customer's order instead of speaking (no mic)")
    ap.add_argument("--age", dest="webcam_age", action="store_true",
                    help="Estimate is_minor from the WEBCAM face (real perception) "
                         "instead of the scenario's per-table MINOR_ENTITIES tag")
    args = ap.parse_args()
    WEBCAM_AGE = args.webcam_age
    main(rehearse=args.rehearse, text_input=args.text_input)
