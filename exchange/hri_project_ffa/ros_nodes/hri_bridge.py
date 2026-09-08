#!/usr/bin/env python3

import json
import math
import os
import re
import sys
import time

import rclpy
from std_msgs.msg import String
from gazebo_msgs.srv import GetEntityState, SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose

from waiter_robot.arm_controller import ArmController
from waiter_robot.grasp_test import GraspDemo, run_grasp_and_serve, PERSON_BASE
import waiter_robot.grasp_test as gt
import waiter_robot.arm_controller as ac_mod

_HRI_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _HRI_ROOT not in sys.path:
    sys.path.insert(0, _HRI_ROOT)
from graph.world_poses import world_pose

# menu.yaml is the single source of truth for WHAT the bar serves (see the
# header of that file); this maps each menu "serve" word to the Gazebo entity
# that physically stands for it on the counter. Only these six exist in
# waiter_scene.world — a menu item added without an entity here is unfetchable
# and must be REJECTED, not silently swapped for another drink.
SERVE_TO_ENTITY = {"coca cola": "coke_bottle",
                   "sprite": "sprite_bottle",
                   "water": "water_bottle",
                   "juice": "juice_bottle",
                   "wine": "wine_bottle",
                   "pringles": "pringles_can"}

# Words the LLM may emit that menu.yaml doesn't list (mostly Italian: the
# customer speaks Italian and qwen echoes the word back untranslated).
_EXTRA_WORDS = {"cocacola": "coke_bottle", "coca": "coke_bottle",
                "acqua": "water_bottle",
                "aranciata": "juice_bottle", "succo": "juice_bottle",
                "patatine": "pringles_can",
                "vino": "wine_bottle"}

def _load_menu_words():
    """{word: entity} from menu.yaml's serve words + synonyms, falling back to
    SERVE_TO_ENTITY alone when the menu can't be read (the bridge runs in the
    container; a missing/broken menu must not stop deliveries)."""
    words = dict(SERVE_TO_ENTITY)
    words.update(_EXTRA_WORDS)
    try:
        import yaml
        with open(os.path.join(_HRI_ROOT, "menu.yaml")) as f:
            items = (yaml.safe_load(f) or {}).get("items") or []
        for it in items:
            entity = SERVE_TO_ENTITY.get(str(it.get("serve", "")).lower().strip())
            if not entity:
                continue
            for w in [it.get("serve"), it.get("name")] + list(it.get("synonyms") or []):
                if w:
                    words[str(w).lower().strip()] = entity
    except Exception as e:
        print("[hri_bridge] menu.yaml non letto (%s); uso la mappa interna." % e)
    return words

BOTTLE_FOR = _load_menu_words()
# Longest first: "coca cola" must win over "cola", "orange juice" over "orange".
# Dict order used to decide this, so a menu edit could silently change which
# bottle a word resolved to.
_BOTTLE_WORDS = sorted(BOTTLE_FOR, key=len, reverse=True)

def bottle_for(target):
    """Gazebo entity for the requested item, or None when nothing on the
    counter matches.

    Returning None matters: the old default was sprite_bottle, so any word the
    map didn't know ("acqua", "una bibita", "snack", "drink") sent the robot to
    the sprite's spot at x=2.0 and it fetched the wrong drink from the wrong
    place instead of saying it couldn't."""
    t = (target or "").lower()
    for word in _BOTTLE_WORDS:
        if re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(word), t):
            return BOTTLE_FOR[word]
    return None

_BOTTLE_FALLBACK = {"sprite_bottle": (2.0, -3.2, 0.80),
                    "coke_bottle":   (2.5, -3.2, 0.80),
                    "water_bottle":  (1.5, -3.2, 0.80),
                    "juice_bottle":  (-2.1, -3.2, 0.80),
                    "pringles_can":  (-1.5, -3.2, 0.80),
                    "wine_bottle":   (3.0, -3.2, 0.71)}
BOTTLE_HOME = {}
for _entity, _fallback in _BOTTLE_FALLBACK.items():
    _pose = world_pose(_entity)
    BOTTLE_HOME[_entity] = _pose[:3] if _pose else _fallback
del _entity, _fallback, _pose
_GRASP_Y, _GRASP_YAW = -2.70, math.radians(-90)

RESTOCK = os.environ.get("HRI_RESTOCK", "1") != "0"

_ARM_TUCKED = True

def grasp_base_for(entity):
    x = BOTTLE_HOME.get(entity, (2.0, -3.2, 0.80))[0]
    return (x, _GRASP_Y, _GRASP_YAW)

USE_GRAPH_POSE = os.environ.get("HRI_GRAPH_POSE", "1") != "0"
_MAX_LATERAL_OFFSET = 0.5
_GRASP_STANDOFF = 0.5
_MAP_X_RANGE = (-3.0, 3.5)
_MAP_Y_RANGE = (-2.75, -2.40)

def grasp_base_from_grounding(entity, grounding):
    base_x, base_y, _ = BOTTLE_HOME.get(entity, (2.0, -3.2, 0.80))
    g = grounding or {}
    centroid = g.get("centroid")
    if USE_GRAPH_POSE and not RESTOCK and centroid:
        if g.get("frame") == "map":
            gx = max(_MAP_X_RANGE[0], min(_MAP_X_RANGE[1], float(centroid[0])))
            gy = float(centroid[1]) + _GRASP_STANDOFF
            gy = max(_MAP_Y_RANGE[0], min(_MAP_Y_RANGE[1], gy))
            print("[hri_bridge] grasp base dal graph (frame mappa): "
                  "(%.2f, %.2f) — bottiglia vista a (%.2f, %.2f), casa (%.2f)"
                  % (gx, gy, float(centroid[0]), float(centroid[1]), base_x))
            return (gx, gy, _GRASP_YAW)
        lateral = max(-_MAX_LATERAL_OFFSET,
                      min(_MAX_LATERAL_OFFSET, float(centroid[0])))
        x = base_x - lateral
        print("[hri_bridge] grasp base dal graph (camera frame): x=%.2f "
              "(casa %.2f, offset %.2f m, depth %s)"
              % (x, base_x, lateral, g.get("depth_source")))
        return (x, _GRASP_Y, _GRASP_YAW)
    return (base_x, _GRASP_Y, _GRASP_YAW)

_BASE_VMAX = gt.V_MAX
_BASE_WMAX = gt.W_MAX
_NAV_BASE_VEL = 0.25
SPEED_BY_URGENCY = {"high": 1.30, "medium": 1.0, "low": 0.90}

def _set_nav_speed(vel):
    """Set Nav2's FollowPath.max_vel_x. Best-effort: a failure here costs the
    pace cue, not the delivery."""
    from rcl_interfaces.srv import SetParameters
    from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue
    from rcl_interfaces.msg import ParameterType

    node = rclpy.create_node("hri_nav_speed")
    try:
        cli = node.create_client(SetParameters, "/controller_server/set_parameters")
        if not cli.wait_for_service(timeout_sec=3.0):
            return False
        p = ParamMsg()
        p.name = "FollowPath.max_vel_x"
        p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                 double_value=float(vel))
        req = SetParameters.Request()
        req.parameters = [p]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        return fut.result() is not None
    except Exception as e:
        print("[hri_bridge] velocita' Nav2 non impostata:", e)
        return False
    finally:
        node.destroy_node()

COMMAND_FILE = os.environ.get(
    "HRI_COMMAND_FILE",
    "/root/exchange/exchange/hri_project_ffa/shared/command.json")
STATUS_FILE = os.environ.get(
    "HRI_STATUS_FILE",
    "/root/exchange/exchange/hri_project_ffa/shared/status.json")
CUSTOMERS_FILE = os.environ.get(
    "HRI_CUSTOMERS_FILE",
    "/root/exchange/exchange/hri_project_ffa/shared/customers_present.json")

CUSTOMER_ENTITIES = [(1, "male03"), (2, "female02"),
                     (3, "female03"), (4, "female02bis")]

DROP_REQUEST_FILE = os.environ.get(
    "HRI_DROP_FILE",
    "/root/exchange/exchange/hri_project_ffa/shared/drop_object.request")
CLEAN_REQUEST_FILE = os.environ.get(
    "HRI_CLEAN_FILE",
    "/root/exchange/exchange/hri_project_ffa/shared/clean_object.request")
FALLEN_ENTITY = "fallen_drink_gz"
_FALLEN_SDF = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><cylinder><radius>0.035</radius><length>0.20</length></cylinder></geometry>
        <material>
          <ambient>0.85 0.1 0.1 1</ambient>
          <diffuse>0.85 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

def _delete_fallen():
    """Remove the fallen-drink model from Gazebo (idempotent)."""
    node = rclpy.create_node("hri_drop_cleaner")
    try:
        cli = node.create_client(DeleteEntity, "/delete_entity")
        if not cli.wait_for_service(timeout_sec=3.0):
            print("[hri_bridge] /delete_entity non disponibile — salto delete")
            return False
        req = DeleteEntity.Request()
        req.name = FALLEN_ENTITY
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        print("[hri_bridge] 🧹 bottiglia caduta rimossa dalla scena (%s)" % FALLEN_ENTITY)
        return True
    except Exception as e:
        print("[hri_bridge] delete fallen fallito:", e)
        return False
    finally:
        node.destroy_node()

def _spawn_fallen(x, y):
    """Spawn the fallen-drink model lying on the floor at (x, y)."""
    _delete_fallen()
    node = rclpy.create_node("hri_drop_spawner")
    try:
        cli = node.create_client(SpawnEntity, "/spawn_entity")
        if not cli.wait_for_service(timeout_sec=3.0):
            print("[hri_bridge] /spawn_entity non disponibile — salto spawn")
            return False
        req = SpawnEntity.Request()
        req.name = FALLEN_ENTITY
        req.xml = _FALLEN_SDF.format(name=FALLEN_ENTITY)
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = 0.05
        pose.orientation.x = 0.70711
        pose.orientation.w = 0.70711
        req.initial_pose = pose
        req.reference_frame = "world"
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        print("[hri_bridge] 🥤 bottiglia caduta creata a (%.2f, %.2f)" % (x, y))
        return True
    except Exception as e:
        print("[hri_bridge] spawn fallen fallito:", e)
        return False
    finally:
        node.destroy_node()

def poll_drop_clean():
    """Handle the physical drop/clean request files written by the host. Runs
    between commands in the main loop; consumes (deletes) the request file."""
    if os.path.exists(DROP_REQUEST_FILE):
        x, y = 0.0, 0.0
        try:
            with open(DROP_REQUEST_FILE) as f:
                d = json.load(f)
            x, y = float(d.get("x", 0.0)), float(d.get("y", 0.0))
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
        try:
            os.remove(DROP_REQUEST_FILE)
        except OSError:
            pass
        _spawn_fallen(x, y)
    if os.path.exists(CLEAN_REQUEST_FILE):
        try:
            os.remove(CLEAN_REQUEST_FILE)
        except OSError:
            pass
        _delete_fallen()

PLACES_FILE = os.environ.get(
    "HRI_PLACES_FILE",
    "/root/exchange/exchange/hri_project_ffa/shared/places_seen.json")
PLACE_ENTITIES = [("bathroom", "bathroom_door", (-5.0, 8.6)),
                  ("reception", "reception_desk", (6.9, -5.3))]

DRINK_WORDS = {"cup", "bottle", "can", "coke", "cola", "coca", "cocacola",
               "sprite", "drink", "soda", "beverage", "water",
               "juice", "orange", "pringles", "chips", "crisps", "snack",
               "lattina", "bibita", "bevanda", "aranciata", "acqua", "cocacola",
               "wine", "vino"}

APPROACH_ACTIONS = {"approach", "greet", "take_order", "go_to_customer", "go", "saluta"}

PROXEMICS_ACTIONS = {"proxemics", "adjust_distance"}

HOME_ACTIONS = {"home", "go_home", "return_home", "return", "casa"}

INDICATE_ACTIONS = {"indicate", "point", "show_route", "indica"}
PLACES = {name: xy for name, _entity, xy in PLACE_ENTITIES}

EXPLORE_ACTIONS = {"explore", "patrol", "esplora"}
SCAN_COUNTER_ACTIONS = {"scan_counter", "scan"}
SPILL_ACTIONS = {"goto_spill"}
COUNTER_DWELL = 9.0
COUNTER_TILT = -0.7
COUNTER_PANS = (0.9, 0.6, 0.0, -0.5, -0.9, -1.2)
COUNTER_VIEW_FLAG = os.environ.get(
    "HRI_COUNTER_VIEW_FLAG",
    "/root/exchange/exchange/hri_project_ffa/shared/counter_view.active")

_counter_spot = [0]

def _counter_view(on, tag=None):
    """Tell the host side whether the robot is parked at a counter with its head
    down — and WHICH spot it is looking at.

    The tag matters: the host used to fire its recognition on a free-running
    8 s timer (PERCEPTION_VLM_PERIOD) while this flag was up, with the head
    holding each spot for 9 s. Whether a given bottle got looked at was down to
    where the timer happened to fall, so the scan came back with a different
    subset every run — pringles and water were the usual casualties, being the
    only item at their spot. Writing a new tag each time the head settles lets
    the host key one call to each spot instead of to the clock."""
    try:
        if on:
            _counter_spot[0] += 1
            with open(COUNTER_VIEW_FLAG, "w") as f:
                f.write(tag or ("spot%d" % _counter_spot[0]))
            os.chmod(COUNTER_VIEW_FLAG, 0o666)
        elif os.path.exists(COUNTER_VIEW_FLAG):
            os.remove(COUNTER_VIEW_FLAG)
    except OSError:
        pass

PAN_LIMIT = 1.2
PAN_MIN_GAP = 0.18

# The frame the host reads, written by frame_grabber_tiago.py.
ROBOT_FRAME = os.environ.get(
    "HRI_ROBOT_FRAME", "/root/exchange/exchange/robot_frame.jpg")
FRESH_FRAME_TIMEOUT = 75.0

def _frame_stamp():
    try:
        return os.path.getmtime(ROBOT_FRAME)
    except OSError:
        return 0.0

def _wait_for_fresh_frame(ac, before, timeout=FRESH_FRAME_TIMEOUT):
    """Block until the camera frame file has been rewritten since `before`.

    The grabber asks for a frame every 0.4 s but Gazebo under software
    rendering delivers one roughly every 27 s (measured 2026-08-29). Holding a
    spot for a fixed COUNTER_DWELL therefore told the host nothing about WHICH
    spot the frame on disk showed: it was routinely one to three spots stale,
    which is why a scan reported sprite only after the head had moved past it,
    and why the set of items found changed from run to run. look_at_customer
    blocks until the head has arrived, so the first frame written after it
    returns is the first one that actually shows this spot.

    Returns the seconds waited. On timeout it gives up and lets the scan carry
    on: a stalled camera should slow the robot down, not strand it."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _frame_stamp() > before:
            return time.time() - t0
        ac._spin(0.5)
    print("[hri_bridge]   nessun frame nuovo in %.0fs — la camera e' ferma?"
          % timeout, flush=True)
    return time.time() - t0

def counter_pans_from_here(ac):
    """Head angles that actually point at the known counter spots, computed
    from where the robot ended up.

    Fixed sweep angles were the reason water and juice were never seen: chosen
    by eye, they straddled the items and half of them framed the floor. Aiming
    the head is not deciding what is there — the model is still the one that
    has to look and say what it sees, and it reports nothing when a spot is
    empty."""
    try:
        x, y, th = ac._base
    except Exception:
        return list(COUNTER_PANS)
    pans = []
    for bx, by, _ in BOTTLE_HOME.values():
        pan = gt.norm_angle(math.atan2(by - y, bx - x) - th)
        if abs(pan) > PAN_LIMIT:
            continue
        if all(abs(pan - q) > PAN_MIN_GAP for q in pans):
            pans.append(pan)
    pans.sort(reverse=True)
    return pans or list(COUNTER_PANS)

def observe_counter(ac, demo, yaw):
    """Face the counter, look down at its surface and hold on each spot where
    a counter item is known to sit, long enough for the host's VLM window to
    land inside it."""
    demo._rotate_to(yaw)
    pans = counter_pans_from_here(ac)
    print("[hri_bridge]   %d punti da guardare: %s"
          % (len(pans), ", ".join("%.2f" % p for p in pans)), flush=True)
    try:
        for pan in pans:
            # Flag DOWN while the head travels, UP only once it has settled on
            # the spot. It used to be raised once for the whole sweep, so the
            # host kept scanning frames grabbed mid-movement — and the camera
            # passes over the decorative wall shelf (bar_bottles_*, dozens of
            # liquor bottles that are not on the menu) on its way down and
            # across. Those frames cost a VLM call and showed the model a
            # shelf-full of drinks that are not the counter stock.
            _counter_view(False)
            stamp = _frame_stamp()
            ac.look_at_customer(pan=pan, tilt=COUNTER_TILT)
            waited = _wait_for_fresh_frame(ac, stamp)
            _counter_view(True, "%.2f@%.2f,%.2f" % (pan, ac._base[0], ac._base[1]))
            print("[hri_bridge]   sguardo pan=%.2f: frame nuovo dopo %.0fs, "
                  "sosta %.0fs" % (pan, waited, COUNTER_DWELL), flush=True)
            ac._spin(COUNTER_DWELL)
    finally:
        _counter_view(False)
        ac.look_at_customer(pan=0.0, tilt=0.05)
EXPLORE_VIEWPOINTS = [
    ((2.0, _GRASP_Y, _GRASP_YAW), 10.0),
    ((-1.8, _GRASP_Y, _GRASP_YAW), 10.0),
]

def effective_urgency(mood):
    """Urgency tier from the mood; an unhappy customer is treated as urgent."""
    mood = mood or {}
    urgency = str(mood.get("urgency", "medium")).lower()
    if str(mood.get("sentiment", "")).lower() == "negative" and urgency != "high":
        urgency = "high"
    return urgency if urgency in SPEED_BY_URGENCY else "medium"

def apply_mood_speed(urgency):
    """Set the walking pace for this delivery. Returns the factor applied."""
    factor = SPEED_BY_URGENCY.get(urgency, 1.0)
    _set_nav_speed(min(0.45, _NAV_BASE_VEL * factor))
    gt.V_MAX = round(min(0.95, _BASE_VMAX * factor), 3)
    gt.W_MAX = round(min(1.80, _BASE_WMAX * (1.15 if factor > 1 else 1.0)), 3)
    return factor

def restore_mood_speed():
    """Back to the baseline pace, so the next customer starts from neutral."""
    _set_nav_speed(_NAV_BASE_VEL)
    gt.V_MAX = _BASE_VMAX
    gt.W_MAX = _BASE_WMAX

def write_status(state, command=None):
    """Scrive lo stato corrente del robot in status.json (letto dal Mac)."""
    data = {"state": state}
    if command is not None:
        data["command_id"] = command.get("command_id")
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f)
    except OSError:
        pass

def detect_customers(ac):
    """'Percezione' dei clienti durante lo scan: interroga il ground-truth di
    Gazebo per sapere quali clienti (entita' citizen) sono presenti e a quale
    tavolo. Ritorna una lista [{table_id, entity, pos:[x,y]}]. Cosi' il robot
    SA dove sono i clienti dal grafo, invece di chiedere "c'e' un altro cliente?"."""
    present = []
    if not ac._get_state.wait_for_service(timeout_sec=2.0):
        print("[hri_bridge] /get_entity_state non disponibile — salto il rilevamento clienti")
        return present
    for tid, name in CUSTOMER_ENTITIES:
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = "world"
        res = ac._wait_future(ac._get_state.call_async(req), 2.0)
        if res is not None and res.success:
            p = res.state.pose.position
            present.append({"table_id": tid, "entity": name,
                            "pos": [round(p.x, 2), round(p.y, 2)]})
    return present

def write_customers(present):
    try:
        with open(CUSTOMERS_FILE, "w") as f:
            json.dump({"present": present}, f)
    except OSError:
        pass

def detect_places(ac):
    """'Percezione' dei LUOGHI durante lo scan: verifica via ground-truth quali
    luoghi noti (bagno, reception) esistono e ne prende il punto da indicare.
    Ritorna {name: {"pos": [x, y]}}. Cosi' il robot SA dove sono dopo lo scan
    (e puo' indicarli), invece di dire 'non l'ho ancora incontrato'."""
    places = {}
    if not ac._get_state.wait_for_service(timeout_sec=2.0):
        return places
    for name, entity, point in PLACE_ENTITIES:
        req = GetEntityState.Request()
        req.name = entity
        req.reference_frame = "world"
        res = ac._wait_future(ac._get_state.call_async(req), 2.0)
        if res is not None and res.success:
            places[name] = {"pos": list(point)}
    return places

def write_places(places):
    try:
        with open(PLACES_FILE, "w") as f:
            json.dump({"places": places}, f)
    except OSError:
        pass

ROBOT_POSE_FILE = os.environ.get(
    "HRI_ROBOT_POSE_FILE",
    "/root/exchange/exchange/hri_project_ffa/shared/robot_pose.json")
ROBOT_ENTITY = os.environ.get("HRI_ROBOT_ENTITY", "tiago")

def _pose_thread():
    """Scrive la posa ground-truth del robot in robot_pose.json ogni ~1.5 s,
    da un thread con un proprio nodo E un proprio SingleThreadedExecutor:
    l'executor GLOBALE di rclpy non e' thread-safe, e spinnarlo da due thread
    (questo + il main che muove il robot) corrompe il wait set ("wait set
    index out of bounds"). Best-effort: se il nodo/servizio muore, si ricrea."""
    from rclpy.executors import SingleThreadedExecutor
    node, cli, execr = None, None, None
    while rclpy.ok():
        try:
            if node is None:
                node = rclpy.create_node("hri_pose_snapshot")
                execr = SingleThreadedExecutor()
                execr.add_node(node)
                cli = node.create_client(GetEntityState, "/get_entity_state")
                if not cli.wait_for_service(timeout_sec=3.0):
                    raise RuntimeError("servizio non disponibile")
            req = GetEntityState.Request()
            req.name = ROBOT_ENTITY
            req.reference_frame = "world"
            fut = cli.call_async(req)
            deadline = time.time() + 2.0
            while not fut.done() and time.time() < deadline:
                execr.spin_once(timeout_sec=0.1)
            res = fut.result() if fut.done() else None
            if res is not None and res.success:
                p = res.state.pose.position
                q = res.state.pose.orientation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                with open(ROBOT_POSE_FILE, "w") as f:
                    json.dump({"x": round(p.x, 3), "y": round(p.y, 3),
                               "yaw": round(yaw, 3), "ts": time.time()}, f)
        except Exception:
            try:
                if execr is not None:
                    execr.shutdown(timeout_sec=0.5)
                if node is not None:
                    node.destroy_node()
            except Exception:
                pass
            node, cli, execr = None, None, None
            time.sleep(3.0)
        time.sleep(1.5)

def _gt_states(names):
    """Pose ground-truth {entity: (x, y, z)} per una lista di entita' Gazebo,
    via un nodo transitorio (stesso pattern di _spawn_fallen: niente nodo
    persistente, vedi il commento DESIGN in testa al file)."""
    out = {}
    node = rclpy.create_node("hri_gt_snapshot")
    try:
        cli = node.create_client(GetEntityState, "/get_entity_state")
        if not cli.wait_for_service(timeout_sec=2.0):
            return out
        for name in names:
            req = GetEntityState.Request()
            req.name = name
            req.reference_frame = "world"
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=2.0)
            res = fut.result()
            if res is not None and res.success:
                p = res.state.pose.position
                out[name] = (p.x, p.y, p.z)
    except Exception as e:
        print("[hri_bridge] snapshot ground-truth fallito:", e)
    finally:
        node.destroy_node()
    return out

def publish_static_layout():
    """Write the places (bathroom, reception) and the customers at their tables from
    Gazebo ground truth, at startup and refreshed by the main loop, so the dashboard
    shows them as soon as the robot appears instead of after the patrol. Drinks are not
    written here — they enter only through the counter scan. The refresh also outlives
    the waiter's session gate, which ignores files older than its own start."""
    states = _gt_states([e for _, e, _ in PLACE_ENTITIES]
                        + [n for _, n in CUSTOMER_ENTITIES])
    if not states:
        return False
    places = {name: {"pos": list(point)}
              for name, entity, point in PLACE_ENTITIES if entity in states}
    if places:
        write_places(places)
    present = [{"table_id": tid, "entity": n,
                "pos": [round(states[n][0], 2), round(states[n][1], 2)]}
               for tid, n in CUSTOMER_ENTITIES if n in states]
    if present:
        write_customers(present)
    return bool(places or present)

def table_poses(command):
    """(person_base, serve_place) dalla destinazione 'table' del comando, o (None, None).
    Il Mac (waiter.py) manda person=[x,y,yaw] e serve=[x,y,z] del tavolo del cliente,
    cosi' il robot consegna al tavolo giusto invece che a un unico posto fisso."""
    t = (command or {}).get("table") or {}
    person = tuple(t["person"]) if t.get("person") else None
    serve = tuple(t["serve"]) if t.get("serve") else None
    return person, serve

MAX_PENDING_AGE = 300.0   # s

def startup_key():
    """Chiave da cui ripartire all'avvio.

    NON e' semplicemente quella in command.json. Marcando come "gia' visto" tutto
    cio' che c'e' nel file all'avvio, un comando scritto mentre il bridge era giu'
    veniva inghiottito per sempre: dopo un rilancio della sim il supervisore
    riavvia il bridge ~60 s dopo Gazebo, e waiter.py nel frattempo manda il suo
    'explore' di partenza. Visto dal vivo il 2026-08-29: command.json con
    command_id 17:25:16, bridge ripartito alle 17:25:26, status.json fermo al
    comando precedente e il robot immobile a (0, 0) mentre waiter.py aspettava.

    Il riferimento giusto e' cio' che e' stato DAVVERO consumato, cioe' il
    command_id in status.json. Se il comando nel file e' diverso ed e' recente,
    va eseguito. La soglia di eta' evita di rieseguire un comando vecchio di una
    sessione precedente quando il bridge muore prima di scrivere lo stato."""
    key = file_key()
    if key is None:
        return None
    try:
        age = time.time() - os.path.getmtime(COMMAND_FILE)
    except OSError:
        return key
    if age > MAX_PENDING_AGE:
        return key      # troppo vecchio: consideralo consumato
    try:
        with open(STATUS_FILE) as f:
            done = json.load(f).get("command_id")
    except (json.JSONDecodeError, OSError, ValueError):
        return key
    if done == key:
        return key      # gia' eseguito prima del riavvio
    print("[hri_bridge] comando %r trovato non eseguito (%.0f s fa, status "
          "fermo a %r): lo prendo in carico." % (key, age, done))
    return None         # in sospeso: fallo eseguire al primo giro

def file_key():
    """Chiave del comando attualmente nel file (command_id o mtime), o None."""
    if not os.path.exists(COMMAND_FILE):
        return None
    try:
        with open(COMMAND_FILE) as f:
            c = json.load(f)
        return c.get("command_id") or os.path.getmtime(COMMAND_FILE)
    except (json.JSONDecodeError, OSError, ValueError):
        return None

def poll_command(last_key):
    """Ritorna (comando, nuova_chiave) se c'e' un nuovo comando 'ready', altrimenti (None, last_key)."""
    if not os.path.exists(COMMAND_FILE):
        return None, last_key
    try:
        with open(COMMAND_FILE) as f:
            command = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None, last_key
    key = command.get("command_id") or os.path.getmtime(COMMAND_FILE)
    if key == last_key:
        return None, last_key
    if str(command.get("status", "")).lower() != "ready":
        return None, key
    return command, key

def execute(command):
    """Crea l'ArmController, pubblica lo stato su /hri/* ed esegue grasp-and-serve."""
    global _ARM_TUCKED
    target = str(command.get("target", "")).lower().strip()
    action = str(command.get("action", "")).lower().strip()
    mood = command.get("mood") or {}
    print("[hri_bridge] Nuovo comando: action=%r target=%r mood=%r"
          % (action, target, mood))
    grounding = command.get("grounding")
    if grounding:
        print("[hri_bridge] ordine ancorato al scene graph: %s (%s %s, centroid=%s)"
              % (grounding.get("node_id"), grounding.get("color"),
                 grounding.get("label"), grounding.get("centroid")))

    # Resolve the actual bottle up front: DRINK_WORDS only says "this sounds
    # like something drinkable", which is far wider than what stands on the
    # counter. Both must agree, or the robot fetches the wrong item.
    entity = bottle_for(target)
    sounds_drinkable = (target in DRINK_WORDS
                        or any(w in target for w in DRINK_WORDS))

    ac = ArmController()
    sel_pub = ac.create_publisher(String, "/hri/selected_target", 10)
    cmd_pub = ac.create_publisher(String, "/hri/action_command", 10)
    stat_pub = ac.create_publisher(String, "/hri/action_status", 10)
    mood_pub = ac.create_publisher(String, "/hri/mood", 10)

    def publish(status):
        sel_pub.publish(String(data=target))
        cmd_pub.publish(String(data=json.dumps(command)))
        stat_pub.publish(String(data=status))
        ac._spin(0.2)

    try:
        publish("received")
        t0 = time.time()
        while ac._base is None and time.time() - t0 < 10:
            rclpy.spin_once(ac, timeout_sec=0.1)
        if ac._base is None:
            print("[hri_bridge] nessun /ground_truth_odom - la sim e' attiva?")
            publish("failed"); write_status("failed", command)
            return

        demo = GraspDemo(ac)

        if action in APPROACH_ACTIONS:
            publish("approaching"); write_status("approaching", command)
            person, _ = table_poses(command)
            dest = person or gt.PERSON_BASE
            print("[hri_bridge] Vado dal cliente (%s)..."
                  % ((command.get("table") or {}).get("name") or "tavolo di default"))
            if not _ARM_TUCKED:
                _ARM_TUCKED = ac.tuck_arm()
            # navigate_to returns None when Nav2 never executed the approach
            # (server absent, goal refused, plan aborted). Ignoring it made the
            # bridge announce "at_customer" from wherever it happened to be
            # standing, so a failed approach looked exactly like a good one and
            # the robot went on to greet an empty chair.
            err = demo.navigate_to(*dest)
            if err is None:
                print("[hri_bridge] Nav2 non ha portato a termine l'avvicinamento "
                      "a %s; non sono dal cliente."
                      % ((command.get("table") or {}).get("name") or "il tavolo"))
                publish("failed"); write_status("failed", command)
                return
            ac.look_at_customer()
            print("[hri_bridge] Arrivato dal cliente (scarto %.1f cm)." % (err * 100))
            publish("at_customer"); write_status("at_customer", command)
            return

        if action in SPILL_ACTIONS:
            # Il robot va FISICAMENTE allo spill (a distanza di sicurezza, rivolto
            # verso di esso) e ci resta finché non arriva il comando successivo
            # (il "cleaner" host lo tiene fermo con la pulizia in corso).
            publish("going_to_spill"); write_status("going_to_spill", command)
            pp = command.get("place_pos") or [0.0, 0.0]
            sx, sy = float(pp[0]), float(pp[1])
            rx, ry, _ = demo._pose()
            dx, dy = sx - rx, sy - ry
            d = math.hypot(dx, dy) or 1.0
            standoff = 0.65                       # fermati PRIMA dello spill
            gx, gy = sx - dx / d * standoff, sy - dy / d * standoff
            yaw = math.atan2(dy, dx)              # rivolto verso lo spill
            if not _ARM_TUCKED:
                _ARM_TUCKED = ac.tuck_arm()
            print("[hri_bridge] Vado allo spill (%.1f, %.1f), mi fermo a %.2fm e "
                  "aspetto la pulizia." % (sx, sy, standoff))
            if demo.navigate_to(gx, gy, yaw) is None:
                publish("failed"); write_status("failed", command); return
            try:
                ac.look_at_customer(pan=0.0, tilt=-0.6)   # sguardo in basso allo spill
            except Exception:
                pass
            publish("at_spill"); write_status("at_spill", command)
            return

        if action in PROXEMICS_ACTIONS:
            delta = float(command.get("delta", 0.0))
            angle = float(command.get("angle", 0.0))
            if delta:
                print("[hri_bridge] Prossemica: %s di %.2fm (mood-based)."
                      % ("avvicinamento" if delta > 0 else "allontanamento", abs(delta)))
                demo.nudge(delta)
            if angle:
                print("[hri_bridge] Prossemica: angolo %.0f° (mood-based)."
                      % math.degrees(angle))
                demo.angle_by(angle)
            publish("adjusted"); write_status("adjusted", command)
            return

        if action in SCAN_COUNTER_ACTIONS:
            publish("scanning"); write_status("scanning", command)
            reached = 0
            for (x, y, yaw), _ in EXPLORE_VIEWPOINTS:
                print("[hri_bridge] Scansione bancone: vado a (%.1f, %.1f) e "
                      "sosto %.0fs." % (x, y, COUNTER_DWELL))
                if demo.navigate_to(x, y, yaw) is None:
                    print("[hri_bridge]   bancone (%.1f, %.1f) non raggiunto" % (x, y))
                    continue
                observe_counter(ac, demo, yaw)
                reached += 1
            if reached == 0:
                print("[hri_bridge] nessun bancone raggiunto")
                publish("failed"); write_status("failed", command)
                return
            print("[hri_bridge] Scansione bancone completata.")
            publish("scanned"); write_status("scanned", command)
            return

        if action in EXPLORE_ACTIONS:
            publish("exploring"); write_status("exploring", command)
            print("[hri_bridge] Esplorazione del bar: passo davanti ai banconi...")
            for (x, y, yaw), dwell in EXPLORE_VIEWPOINTS:
                if demo.navigate_to(x, y, yaw) is None:
                    print("[hri_bridge]   viewpoint (%.1f, %.1f) non raggiunto"
                          % (x, y), flush=True)
                    continue
                print("[hri_bridge]   viewpoint (%.1f, %.1f): osservo il bancone"
                      % (x, y), flush=True)
                observe_counter(ac, demo, yaw)
            present = detect_customers(ac)
            write_customers(present)
            print("[hri_bridge] Clienti rilevati ai tavoli: %s"
                  % ([c["table_id"] for c in present] or "nessuno"))
            places = detect_places(ac)
            write_places(places)
            print("[hri_bridge] Luoghi rilevati: %s" % (list(places) or "nessuno"))
            print("[hri_bridge] Esplorazione completata.")
            publish("explored"); write_status("explored", command)
            return

        if action in INDICATE_ACTIONS:
            publish("indicating"); write_status("indicating", command)
            place = PLACES.get(target)
            if place is None:
                print("[hri_bridge] luogo %r sconosciuto; ignoro." % target)
                publish("rejected"); write_status("rejected", command)
                return
            x, y, yaw0 = ac._base
            yaw_place = math.atan2(place[1] - y, place[0] - x)
            print("[hri_bridge] Indico %r: mi giro verso (%.1f, %.1f), gesto, "
                  "e torno dal cliente..." % (target, place[0], place[1]))
            demo._rotate_to(yaw_place)
            ac.move_torso(gt.TORSO_SERVE)
            ac.offer()
            ac._spin(3.0)
            ac.tuck_arm()
            ac.move_torso(gt.TORSO_NAV)
            demo._rotate_to(yaw0)
            publish("indicated"); write_status("indicated", command)
            return

        if action in HOME_ACTIONS:
            publish("going_home"); write_status("going_home", command)
            print("[hri_bridge] Torno al punto di partenza %s..." % (gt.HOME_BASE,))
            demo.navigate_to(*gt.HOME_BASE)
            print("[hri_bridge] Arrivato al punto di partenza.")
            publish("home"); write_status("home", command)
            return

        if entity is None:
            why = ("non e' sul bancone" if sounds_drinkable
                   else "non e' una bibita che so prendere")
            print("[hri_bridge] target %r %s; ignoro (nessuna bottiglia "
                  "corrispondente in scena)." % (target, why))
            publish("rejected"); write_status("rejected", command)
            return
        publish("executing"); write_status("serving", command)

        urgency = effective_urgency(mood)
        factor = apply_mood_speed(urgency)
        mood_pub.publish(String(data=json.dumps(mood)))
        print("[hri_bridge] mood emotion=%r sentiment=%r -> urgency=%s, "
              "nav speed x%.2f (V_MAX=%.2f)"
              % (mood.get("emotion"), mood.get("sentiment"), urgency,
                 factor, gt.V_MAX))

        ac_mod.BOTTLE_NAME = entity
        base = grasp_base_from_grounding(entity, grounding)
        prev_base = gt.GRASP_BASE
        gt.GRASP_BASE = base
        person, serve = table_poses(command)
        prev_person, prev_serve = gt.PERSON_BASE, gt.SERVE_PLACE
        if person:
            gt.PERSON_BASE = person
        if serve:
            gt.SERVE_PLACE = serve
        print("[hri_bridge] fetching %r (%s) — grasp base %s, serving to %s at %s"
              % (target, entity, base,
                 (command.get("table") or {}).get("name") or "default table",
                 gt.SERVE_PLACE))
        if RESTOCK:
            ac._spin(0.3)
            ac._teleport_bottle(*BOTTLE_HOME[entity])
            ac._spin(0.3)
        try:
            ok = run_grasp_and_serve(ac, demo)
        finally:
            restore_mood_speed()
            gt.GRASP_BASE = prev_base
            gt.PERSON_BASE = prev_person
            gt.SERVE_PLACE = prev_serve
        _ARM_TUCKED = ok
        publish("done" if ok else "failed")
        write_status("done" if ok else "failed", command)
        print("[hri_bridge] Esecuzione %s." % ("OK" if ok else "FALLITA"))
    finally:
        ac.detach_bottle()
        ac.destroy_node()

def main():
    import threading
    rclpy.init()
    print("[hri_bridge] In ascolto. File comandi: " + COMMAND_FILE)
    threading.Thread(target=_pose_thread, daemon=True).start()
    last_key = startup_key()
    last_layout, layout_announced = 0.0, False
    try:
        while rclpy.ok():
            if time.time() - last_layout > 5.0:
                if publish_static_layout() and not layout_announced:
                    layout_announced = True
                    print("[hri_bridge] layout iniziale pubblicato: luoghi + "
                          "clienti visibili alla dashboard (bevande dopo lo scan).")
                last_layout = time.time()
            cmd, last_key = poll_command(last_key)
            if cmd is not None:
                try:
                    execute(cmd)
                except Exception as e:
                    print("[hri_bridge] ERRORE inatteso durante l'esecuzione: "
                          "%s: %s" % (type(e).__name__, e))
                    write_status("failed", cmd)
            poll_drop_clean()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
