#!/usr/bin/env python3
"""Returning-customer recognition and memory.

YuNet detects the face, SFace embeds it in 128 dimensions, and shared/customers.json
holds one embedding plus profile (name, visits, past orders) per customer — enough
for a personalised greeting, "the usual?", and history-based upselling.

API: recognize_customer(bgr_frame), remember_order(id, drink), set_name(id, name)."""
import json
import os
import time

import numpy as np

try:
    import cv2
    _HAS_CV2 = hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
except Exception:
    _HAS_CV2 = False

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")
DB_PATH = os.path.join(HERE, "..", "shared", "customers.json")
YUNET = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODELS, "face_recognition_sface_2021dec.onnx")
AGE_MODEL = os.path.join(MODELS, "age_googlenet.onnx")

MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.363"))

_detector = None
_recognizer = None

def _available():
    return _HAS_CV2 and os.path.exists(YUNET) and os.path.exists(SFACE)

def _lazy_load():
    global _detector, _recognizer
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(YUNET, "", (320, 320),
                                              score_threshold=0.7)
        _recognizer = cv2.FaceRecognizerSF.create(SFACE, "")
    return _detector, _recognizer

def _embedding(bgr):
    """Embedding SFace del volto più grande nel frame, o None."""
    det, rec = _lazy_load()
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    aligned = rec.alignCrop(bgr, face)
    return rec.feature(aligned)

_AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
                "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
_MINOR_AGE_BUCKETS = {"(0-2)", "(4-6)", "(8-12)", "(15-20)"}
_age_net = None
_age_tried = False

def _age_load():
    global _age_net, _age_tried
    if _age_tried:
        return _age_net
    _age_tried = True
    if _HAS_CV2 and os.path.exists(AGE_MODEL):
        try:
            _age_net = cv2.dnn.readNetFromONNX(AGE_MODEL)
            print("🎂 Age estimation model loaded (GoogLeNet/Adience, 8 brackets).")
        except Exception as exc:
            print(f"🎂 (age model failed to load: {exc})")
            _age_net = None
    return _age_net

def estimate_age(bgr):
    """Detect the largest face in `bgr` and classify its age bracket.

    Returns (bucket_label, confidence 0..1, is_minor) or (None, 0.0, None) if
    no face is found or the model is unavailable — callers should fall back
    to another signal (e.g. a scenario-authored flag) in that case."""
    net = _age_load()
    if net is None or bgr is None or bgr.size == 0:
        return None, 0.0, None
    det, _ = _lazy_load()
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None, 0.0, None
    x, y, fw, fh = [int(round(v)) for v in
                    max(faces, key=lambda f: float(f[2]) * float(f[3]))[:4]]
    x, y = max(x, 0), max(y, 0)
    face = bgr[y:y + fh, x:x + fw]
    if face.size == 0:
        return None, 0.0, None
    blob = cv2.dnn.blobFromImage(face, 1.0, (224, 224), (104, 117, 123),
                                 swapRB=False)
    net.setInput(blob)
    probs = net.forward().flatten()
    idx = int(probs.argmax())
    bucket = _AGE_BUCKETS[idx]
    return bucket, float(probs[idx]), bucket in _MINOR_AGE_BUCKETS

def _load_db():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return {}

def _save_db(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    tmp = DB_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp, DB_PATH)

def _profile_view(cid, prof, is_new):
    orders = prof.get("orders", [])
    return {"id": cid, "is_new": is_new,
            "name": prof.get("name"),
            "visits": prof.get("visits", 1),
            "orders": orders,
            "last_order": orders[-1] if orders else None}

def recognize_customer(bgr):
    """Riconosce o registra il cliente dal frame. Ritorna il profilo (dict) o None
    se non si vede un volto / il riconoscimento non è disponibile."""
    if not _available() or bgr is None:
        return None
    feat = _embedding(bgr)
    if feat is None:
        return None
    _, rec = _lazy_load()
    db = _load_db()
    best_id, best_score = None, -1.0
    for cid, prof in db.items():
        emb = prof.get("embedding")
        if not emb:
            continue
        emb = np.array(emb, dtype=np.float32).reshape(1, -1)
        score = rec.match(feat, emb, cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_id, best_score = cid, score
    if best_id is not None and best_score >= MATCH_THRESHOLD:
        db[best_id]["visits"] = db[best_id].get("visits", 0) + 1
        db[best_id]["last_seen"] = time.time()
        _save_db(db)
        return _profile_view(best_id, db[best_id], is_new=False)
    new_id = "cust_%d" % (int(time.time() * 10) % 1000000)
    prof = {"name": None, "visits": 1, "orders": [],
            "embedding": feat.flatten().tolist(), "last_seen": time.time()}
    db[new_id] = prof
    _save_db(db)
    return _profile_view(new_id, prof, is_new=True)

def customer_by_key(key, name=None):
    """Profile keyed by a stable id (the table's occupant), not by the webcam face.
    
    One webcam serving several tables would merge every customer into one, so identity
    comes from which table the robot is at. Same return shape as recognize_customer()."""
    if not key:
        return None
    db = _load_db()
    is_new = key not in db
    if is_new:
        db[key] = {"name": name, "visits": 0, "orders": [], "embedding": None}
    db[key]["visits"] = db[key].get("visits", 0) + 1
    db[key]["last_seen"] = time.time()
    if name and not db[key].get("name"):
        db[key]["name"] = name
    _save_db(db)
    return _profile_view(key, db[key], is_new=is_new)

def remember_order(customer_id, drink):
    """Aggiunge una bevanda allo storico del cliente."""
    if not customer_id:
        return
    db = _load_db()
    if customer_id in db:
        db[customer_id].setdefault("orders", []).append(drink)
        _save_db(db)

def set_name(customer_id, name):
    if not customer_id or not name:
        return
    db = _load_db()
    if customer_id in db:
        db[customer_id]["name"] = name
        _save_db(db)
