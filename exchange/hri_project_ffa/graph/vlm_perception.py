
import base64
import json
import os
import re
import time

import requests

from .object_node import Detection

try:
    from config import PERCEPTION_VLM_MODEL, PERCEPTION_VLM_TIMEOUT
except Exception:
    PERCEPTION_VLM_MODEL = "gemini-3.1-flash-lite"
    PERCEPTION_VLM_TIMEOUT = 40

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")

# Which Gazebo entity stands for each menu item on the counter, plus how the
# robot may hear it named. One entry per menu.yaml item — nothing else is on
# the working counters, so anything else the model reports is not stock.
_DRINK_ENTITY = {
    "coca cola": ("coke_bottle", "red",
                  ("coca-cola", "coca cola", "coke", "cola", "coca")),
    "sprite":    ("sprite_bottle", "green", ("sprite",)),
    "water":     ("water_bottle", "blue",
                  ("water", "aqua", "acqua", "bottle of water")),
    "juice":     ("juice_bottle", "orange",
                  ("juice", "orange juice", "orange")),
    "pringles":  ("pringles_can", "red",
                  ("pringles", "chips", "crisps", "potato")),
    "wine":      ("wine_bottle", "purple", ("wine", "vino")),
}

# Fallback positions, used only if waiter_scene.world cannot be read.
_LAYOUT_FALLBACK = {"coca cola": (2.5, -3.2), "sprite": (2.0, -3.2),
                    "water": (1.5, -3.2), "juice": (-2.1, -3.2),
                    "pringles": (-1.5, -3.2), "wine": (3.0, -3.2)}
_LAYOUT_Z = 0.85   # where a Detection for a counter item is placed in the map

def _build_layout():
    """Counter layout with the positions READ FROM THE WORLD FILE.

    They used to be copied into this file by hand, which is the same trap the
    table docks fell into: move a bottle in waiter_scene.world and the robot
    keeps reporting the drink at the old spot, with nothing to flag the drift.
    Only the position is scene data; colour and aliases stay here because they
    are recognition vocabulary, not geometry."""
    from .world_poses import world_pose
    layout = {}
    for name, (entity, color, aliases) in _DRINK_ENTITY.items():
        pose = world_pose(entity)
        x, y = pose[:2] if pose else _LAYOUT_FALLBACK[name]
        layout[name] = dict(pos=(x, y, _LAYOUT_Z), color=color,
                            aliases=aliases, entity=entity)
    return layout

DRINK_LAYOUT = _build_layout()

_CANONICAL = list(DRINK_LAYOUT)

_MENU_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "menu.yaml")

_FALLBACK_APPEARANCE = {
    "coca cola": "a red can with the white Coca-Cola script",
    "sprite":    "a white and green bottle with the Sprite logo",
    "water":     "a plain pale blue or clear bottle",
    "juice":     "an orange bottle",
    "pringles":  "a tall orange cylindrical tube, the shape of a Pringles can",
    "wine":      "a tall dark maroon bottle with a narrow neck and no readable "
                 "label",
}

def _load_appearance(path=_MENU_FILE):
    """How each menu item LOOKS, read from menu.yaml so the appearance lives
    with the rest of the menu data instead of in code. These are scene facts,
    not decisions: what is actually on the counter is still the model's call."""
    try:
        import yaml
        with open(path) as f:
            items = (yaml.safe_load(f) or {}).get("items") or []
        out = {}
        for it in items:
            look = it.get("appearance")
            serve = str(it.get("serve", it.get("name", ""))).lower()
            if look and serve in DRINK_LAYOUT:
                out[serve] = str(look)
        return out or dict(_FALLBACK_APPEARANCE)
    except Exception:
        return dict(_FALLBACK_APPEARANCE)

APPEARANCE = _load_appearance()
_APPEARANCE_TEXT = "\n".join(
    "    %s: %s" % (name, APPEARANCE[name]) for name in _CANONICAL
    if name in APPEARANCE)

_PROMPT = (
    "You are a bar-service robot looking at a drinks counter. The counter may "
    "hold some of these items: " + ", ".join(_CANONICAL) + ".\n\n"
    "These are simple 3D models, so several carry no readable label at all — "
    "identify them by shape, colour and size, not only by text. In this bar "
    "they roughly look like this:\n" + _APPEARANCE_TEXT + "\n\n"
    "Treat those as HINTS, not as requirements: they are approximate. If "
    "branding or a logo clearly identifies an item, report it even when its "
    "shape does not match the hint — a Sprite can is still sprite. Only leave "
    "an item out when you genuinely cannot tell what it is.\n\n"
    "Look carefully at the image and reason about what you actually see, then "
    "respond with ONLY a single JSON object (no other text, no markdown "
    "fences) with exactly these keys:\n"
    '  "detected": a list using ONLY these exact words for items you can '
    "actually see standing upright on the counter right now (empty list if "
    "none) — " + ", ".join(_CANONICAL) + ".\n"
    '  "reasoning": one short sentence on the visual evidence you used '
    "(shape, colour, label text, container type) to reach that conclusion.\n"
    '  "anomaly": "none", or a short description of anything that looks '
    "visually wrong, and roughly where.\n"
    '  "anomaly_kind": one of "spill", "oddity", "none". Use "spill" ONLY for '
    "a real hazard someone could slip on or be cut by: pooled or spilled "
    "liquid, a bottle or can lying knocked over, broken glass. Use "
    '"oddity" for anything that merely looks strange — an object you cannot '
    "identify, something at an odd height or angle, an item you did not "
    "expect. An unrecognised object is an oddity, NEVER a spill.")

def _encode_counter(frame_path):
    """Encode the frame, upscaled so the model can make out labels.

    No crop: the robot tilts its head down onto the counter before scanning, so
    the decorative shelf bottles are already out of shot. The old lower-band
    crop assumed a level head and, with the head down, cut off the items that
    had moved to the top of the image."""
    try:
        import cv2
        img = cv2.imread(frame_path)
        if img is not None:
            big = cv2.resize(img, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_CUBIC)
            ok, buf = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                return base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_warned_no_key = False

_last_seen = {"digest": None, "result": None}

def _digest(frame_path):
    import hashlib
    try:
        with open(frame_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)

def _parse_json_response(text):
    """Gemini is ASKED for pure JSON but doesn't always comply exactly (stray
    markdown fences, a leading sentence) — pull out the first {...} block and
    parse that. Returns {} on any failure, never raises: a malformed reply is
    a perception gap, not a crash."""
    m = _JSON_BLOCK_RE.search(text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _call_gemini(prompt, image_b64, model, timeout):
    """Shared Gemini vision call. Returns the raw text reply, or None on any
    failure (missing key, network, bad frame...) — every caller degrades to
    its own graceful empty result on None."""
    if not GEMINI_API_KEY:
        return None
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
        ]}],
        "generationConfig": {"temperature": 0.1},
    }
    r = requests.post(GEMINI_URL.format(model=model),
                      params={"key": GEMINI_API_KEY}, json=payload,
                      timeout=timeout)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def recognise_drinks(frame_path, model=None, timeout=None):
    """Ask Gemini which known drinks are visible, WHY it thinks so, and
    whether anything on the counter looks visually wrong. Returns
    {"detected": set of canonical names, "reasoning": str, "anomaly": str or
    None} — always this shape, empty/blank on any failure (missing key, no
    network, bad frame, unparseable reply, ...), same as any perception gap."""
    global _warned_no_key
    empty = {"detected": set(), "reasoning": "", "anomaly": None,
             "anomaly_kind": "none"}
    model = model or PERCEPTION_VLM_MODEL
    timeout = timeout or PERCEPTION_VLM_TIMEOUT
    if not frame_path or not os.path.exists(frame_path):
        return empty
    if not GEMINI_API_KEY:
        if not _warned_no_key:
            print("[vlm_perception] GEMINI_API_KEY not set — counter drink "
                  "recognition is disabled until it is (see EXPERIMENT.md).")
            _warned_no_key = True
        return empty
    digest = _digest(frame_path)
    if digest is not None and digest == _last_seen["digest"]:
        cached = dict(_last_seen["result"] or empty)
        cached["cached"] = True
        return cached
    try:
        text = _call_gemini(_PROMPT, _encode_counter(frame_path), model, timeout)
    except Exception as e:
        msg = str(e)
        if "429" in msg:
            print("(gemini: quota esaurita, riconoscimento sospeso)")
        else:
            print("(gemini perception call failed:", msg, ")")
        return empty
    data = _parse_json_response(text)
    raw_detected = " ".join(str(x) for x in data.get("detected", []) or []).lower()
    raw_detected = raw_detected.replace("-", " ")
    detected = {name for name, spec in DRINK_LAYOUT.items()
                if any(alias in raw_detected for alias in spec["aliases"])}
    anomaly = data.get("anomaly")
    anomaly = None if not anomaly or str(anomaly).strip().lower() == "none" else str(anomaly).strip()
    kind = str(data.get("anomaly_kind", "")).strip().lower()
    if anomaly is None or kind not in ("spill", "oddity"):
        kind = "none"
    out = {"detected": detected,
           "reasoning": str(data.get("reasoning", "")).strip(),
           "anomaly": anomaly, "anomaly_kind": kind}
    _last_seen["digest"], _last_seen["result"] = digest, dict(out)
    return out

def detections_for(seen_names):
    """Build map-frame Detection objects (at the known counter positions) for
    the drinks Gemini recognised, ready for TemporalManager.update(). Labelled
    by the real canonical NAME (not just appearance) — the model already
    identified which specific drink this is, so grounding can match on name
    directly instead of falling back to an appearance guess."""
    dets = []
    for name in seen_names:
        spec = DRINK_LAYOUT.get(name)
        if not spec:
            continue
        is_can = (name == "pringles")
        dets.append(Detection(
            label=name, confidence=1.0,
            centroid=spec["pos"], frame="map",
            color_name=spec["color"],
            material="plastic" if is_can else "glass",
            shape="can" if is_can else "bottle",
            is_movable=True, source="vlm", depth_source="prior",
            description=f"{name} ({spec['color']} {'can' if is_can else 'bottle'}) "
                        f"recognised on the counter by the vision model"))
    return dets

def perceive_drinks(frame_path, model=None, timeout=None):
    """Convenience: recognise + build detections in one call."""
    return detections_for(recognise_drinks(frame_path, model, timeout)["detected"])

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXCHANGE = os.path.abspath(os.path.join(_HERE, "..", ".."))
OVERHEAD_FRAME_PATH = os.environ.get(
    "SG_OVERHEAD", os.path.join(_EXCHANGE, "overhead_frame.jpg"))

QUADRANT_TO_TABLE = {
    "top-left": 3, "top-right": 4, "bottom-left": 2, "bottom-right": 1,
}

_TABLE_STATES = ("active", "finished", "needs-attention")

_CUSTOMER_PROMPT = (
    "You are looking at a bird's-eye (top-down) view of a bar with 4 tables "
    "arranged in a 2x2 grid: top-left, top-right, bottom-left, bottom-right.\n\n"
    "Look carefully and reason about the scene, then respond with ONLY a "
    "single JSON object (no other text, no markdown fences) with exactly "
    "these keys:\n"
    '  "occupied": a list of the table positions (top-left/top-right/'
    "bottom-left/bottom-right) that currently have a person seated at them "
    "(empty list if none).\n"
    '  "reasoning": one short sentence on the visual evidence you used '
    "(posture, position relative to the table, ...).\n"
    '  "table_states": a JSON object mapping EACH occupied table\'s position '
    'to your best read of its state, one of: "active" (customer present, '
    "nothing unusual), \"finished\" (customer appears to be leaving/has left "
    "their seat, or the table looks cleared), \"needs-attention\" (the "
    "customer appears to be gesturing, waving, or otherwise trying to get "
    "attention). Only include occupied tables. Base this ONLY on what is "
    "visually apparent in this single image — do not guess at anything you "
    "cannot actually see.")

def recognise_occupied_tables(frame_path=None, model=None, timeout=None):
    """Ask Gemini which tables look occupied in the overhead frame, why, and
    what STATE each occupied table appears to be in. Returns {"occupied": set
    of real table_ids (1-4), "reasoning": str, "table_states": {table_id:
    state}} — always this shape, empty/blank on any failure."""
    global _warned_no_key
    empty = {"occupied": set(), "reasoning": "", "table_states": {}}
    frame_path = frame_path or OVERHEAD_FRAME_PATH
    model = model or PERCEPTION_VLM_MODEL
    timeout = timeout or PERCEPTION_VLM_TIMEOUT
    if not frame_path or not os.path.exists(frame_path):
        return empty
    if not GEMINI_API_KEY:
        if not _warned_no_key:
            print("[vlm_perception] GEMINI_API_KEY not set — counter drink "
                  "recognition is disabled until it is (see EXPERIMENT.md).")
            _warned_no_key = True
        return empty
    try:
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        text = _call_gemini(_CUSTOMER_PROMPT, b64, model, timeout)
    except Exception as e:
        print("(gemini customer perception call failed:", e, ")")
        return empty
    data = _parse_json_response(text)
    raw_occupied = [str(x).strip().lower() for x in data.get("occupied", []) or []]
    occupied = {tid for quad, tid in QUADRANT_TO_TABLE.items() if quad in raw_occupied}
    states = {}
    for quad, state in (data.get("table_states") or {}).items():
        tid = QUADRANT_TO_TABLE.get(str(quad).strip().lower())
        state = str(state).strip().lower()
        if tid is not None and tid in occupied and state in _TABLE_STATES:
            states[tid] = state
    return {"occupied": occupied,
            "reasoning": str(data.get("reasoning", "")).strip(),
            "table_states": states}
