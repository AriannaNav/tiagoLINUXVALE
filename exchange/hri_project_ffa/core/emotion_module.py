
import base64
import json
import re
import time

import requests

try:
    from config import OLLAMA_MODEL, OLLAMA_URL
except Exception:
    OLLAMA_MODEL = "qwen2.5:3b"
    OLLAMA_URL = "http://localhost:11434/api/generate"

try:
    import config as _cfg
except Exception:
    _cfg = None

def _cfg_get(name, default):
    return getattr(_cfg, name, default) if _cfg is not None else default

EMOTION_ENABLED = _cfg_get("EMOTION_ENABLED", True)
CAMERA_INDEX = _cfg_get("EMOTION_CAMERA_INDEX", 0)
CAPTURE_SECONDS = _cfg_get("EMOTION_CAPTURE_SECONDS", 5.0)
SHOW_WINDOW = _cfg_get("EMOTION_SHOW_WINDOW", True)
VISION_MODEL = _cfg_get("EMOTION_VISION_MODEL", "")
VISION_TIMEOUT = _cfg_get("EMOTION_VISION_TIMEOUT", 15)

try:
    import cv2
    import numpy as np
    _CV_OK = True
    _FACE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _SMILE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml")
    _EYE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml")
except Exception:
    _CV_OK = False

import os as _os

_EMO_MODEL_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "models",
    "emotion-ferplus-8.onnx")
_EMO_LABELS = ["neutral", "happy", "surprised", "sad", "angry",
               "disgust", "fear", "contempt"]
_emo_net = None
_emo_tried = False

def _emo_load():
    """Lazy-load the FER+ net once. Returns the net or None if unavailable."""
    global _emo_net, _emo_tried
    if _emo_tried:
        return _emo_net
    _emo_tried = True
    if _CV_OK and _os.path.exists(_EMO_MODEL_PATH):
        try:
            _emo_net = cv2.dnn.readNetFromONNX(_EMO_MODEL_PATH)
            print("🧠 FER+ emotion model loaded (8-class deep classifier).")
        except Exception as exc:
            print(f"🧠 (FER+ model failed to load: {exc} — using smile cue)")
            _emo_net = None
    else:
        print("🧠 (FER+ model not found — using Haar smile cue only)")
    return _emo_net

def classify_emotion(bgr_face):
    """Classify a face crop (BGR) into one FER+ emotion.

    Returns (label, confidence 0..1), or (None, 0.0) if the model is
    unavailable or the crop is empty.
    """
    net = _emo_load()
    if net is None or bgr_face is None or bgr_face.size == 0:
        return None, 0.0
    try:
        gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (64, 64)).astype("float32")
        blob = face.reshape(1, 1, 64, 64)
        net.setInput(blob)
        out = net.forward().flatten()
        ex = np.exp(out - out.max())
        probs = ex / ex.sum()
        idx = int(probs.argmax())
        return _EMO_LABELS[idx], float(probs[idx])
    except Exception:
        return None, 0.0

def read_facial_expression(seconds=None, camera_index=None, show=None):
    """Read the customer's facial expression from the webcam.
    
    Returns {face_found, expression, smile_ratio, eyes_ratio, frames, best_frame}, or a
    "no_face" result if OpenCV or the webcam is unavailable, so the caller can always
    continue on speech alone."""
    seconds = CAPTURE_SECONDS if seconds is None else seconds
    camera_index = CAMERA_INDEX if camera_index is None else camera_index
    show = SHOW_WINDOW if show is None else show

    blank = {"face_found": False, "expression": "no_face",
             "emotion": "no_face", "emotion_conf": 0.0, "smile_ratio": 0.0,
             "eyes_ratio": 0.0, "frames": 0, "best_frame": None}

    if not _CV_OK:
        print("👁️  (OpenCV not installed — skipping facial expression, "
              "using speech only)")
        return blank

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"👁️  (Webcam {camera_index} not accessible — using speech only)")
        return blank

    print(f"📷 Webcam on — reading your expression for {seconds:.0f}s "
          "(look at the camera)...")

    win = "TIAGo - reading your expression"
    if show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 640, 480)
        except cv2.error:
            print("👁️  (no display — reading face without a preview window)")
            show = False

    face_frames = 0
    smile_frames = 0
    eye_frames = 0
    best_frame = None
    best_area = 0
    emo_scores = {}
    emo_counts = {}
    t0 = time.monotonic()

    while time.monotonic() - t0 < seconds:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = _FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(80, 80))
        remaining = seconds - (time.monotonic() - t0)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            area = w * h
            face_frames += 1
            if area > best_area:
                best_area = area
                best_frame = frame.copy()

            roi_gray = gray[y:y + h, x:x + w]
            lower = roi_gray[h // 2:, :]
            smiles = _SMILE.detectMultiScale(
                lower, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
            eyes = _EYE.detectMultiScale(
                roi_gray[: h // 2, :], scaleFactor=1.1, minNeighbors=6)
            if len(smiles) > 0:
                smile_frames += 1
            if len(eyes) > 0:
                eye_frames += 1

            emo_label, emo_conf = classify_emotion(frame[y:y + h, x:x + w])
            if emo_label is not None:
                emo_scores[emo_label] = emo_scores.get(emo_label, 0.0) + emo_conf
                emo_counts[emo_label] = emo_counts.get(emo_label, 0) + 1

            if emo_label is not None:
                colour = (0, 220, 0) if emo_label == "happy" else (0, 170, 255)
                label = f"{emo_label} {emo_conf * 100:.0f}%"
            else:
                colour = (0, 220, 0) if len(smiles) > 0 else (0, 170, 255)
                label = "smiling" if len(smiles) > 0 else "neutral"
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, label, (x, max(y - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        else:
            cv2.putText(frame, "looking for your face...", (14, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        cv2.putText(frame, f"reading expression - {remaining:0.1f}s",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if show:
            try:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            except cv2.error:
                show = False

    cap.release()
    if show:
        cv2.destroyWindow(win)
        cv2.waitKey(1)

    if face_frames == 0:
        print("👁️  No face detected — using speech only.")
        return blank

    smile_ratio = smile_frames / face_frames
    eyes_ratio = eye_frames / face_frames

    if emo_scores:
        emotion = max(emo_scores, key=emo_scores.get)
        emotion_conf = round(emo_scores[emotion] / emo_counts[emotion], 2)
        expression = "smiling" if emotion == "happy" else "neutral"
        votes = ", ".join(f"{lbl}={emo_counts[lbl]}"
                          for lbl in sorted(emo_counts, key=emo_counts.get,
                                            reverse=True))
        print(f"👁️  Face read: {emotion} ({emotion_conf * 100:.0f}% avg conf "
              f"over {emo_counts[emotion]}/{face_frames} frames — votes: {votes})")
    else:
        emotion = "happy" if smile_ratio >= 0.30 else "neutral"
        emotion_conf = round(smile_ratio, 2)
        expression = "smiling" if smile_ratio >= 0.30 else "neutral"
        print(f"👁️  Face read: {expression} "
              f"(smiles in {smile_ratio * 100:.0f}% of frames, "
              f"{face_frames} frames)")
    return {"face_found": True, "expression": expression,
            "emotion": emotion, "emotion_conf": emotion_conf,
            "smile_ratio": round(smile_ratio, 2),
            "eyes_ratio": round(eyes_ratio, 2),
            "frames": face_frames, "best_frame": best_frame}

_SENTIMENT_PROMPT = """\
You are the empathic brain of TIAGo, a friendly waiter robot in a bar.
Judge the customer's mood from two signals and answer with JSON only.

Visual signal (from the camera): a face-expression model read the customer's \
face as "{emotion}" (confidence {emo_pct}%; face_found={face_found}).
What the customer said: "{speech}"

Return ONLY this JSON, no other text:
{{
  "emotion": "<happy|neutral|sad|angry|tired|surprised>",
  "sentiment": "<positive|neutral|negative>",
  "sentiment_score": <float 0.0-1.0, 0=very negative, 1=very positive>,
  "urgency": "<low|medium|high>",
  "empathy_note": "<one short internal note on how TIAGo should treat this customer>"
}}"""

def _fallback_mood(facial, speech):
    """Rule-based mood when the LLM is unavailable/unparseable."""
    text = (speech or "").lower()
    neg = any(w in text for w in
              ("angry", "annoyed", "slow", "late", "terrible", "bad", "hate",
               "waiting", "hurry", "awful"))
    emo = facial.get("emotion", facial.get("expression", "neutral"))
    smiling = emo == "happy" or facial.get("expression") == "smiling"
    if smiling and not neg:
        return {"emotion": "happy", "sentiment": "positive",
                "sentiment_score": 0.8, "urgency": "low",
                "empathy_note": "Customer seems cheerful — keep it warm and light.",
                "source": "fallback"}
    if neg or emo in ("angry", "disgust"):
        return {"emotion": "angry", "sentiment": "negative",
                "sentiment_score": 0.2, "urgency": "high",
                "empathy_note": "Customer seems unhappy — apologise and be quick.",
                "source": "fallback"}
    if emo in ("sad", "fear"):
        return {"emotion": "sad", "sentiment": "negative",
                "sentiment_score": 0.35, "urgency": "medium",
                "empathy_note": "Customer looks down — be gentle and reassuring.",
                "source": "fallback"}
    if emo == "surprised":
        return {"emotion": "surprised", "sentiment": "positive",
                "sentiment_score": 0.6, "urgency": "low",
                "empathy_note": "Customer looks surprised — be friendly and clear.",
                "source": "fallback"}
    return {"emotion": "neutral", "sentiment": "neutral",
            "sentiment_score": 0.5, "urgency": "medium",
            "empathy_note": "Neutral customer — be polite and efficient.",
            "source": "fallback"}

_VISION_PROMPT = (
    "You are an expert at reading facial expressions. Look carefully at the "
    "person's EYEBROWS, EYES and MOUTH. Frowning brows or a tight mouth mean "
    "angry; downturned mouth means sad; wide eyes and open mouth mean surprised; "
    "raised cheeks and a smile mean happy; droopy eyes mean tired. Decide their "
    "SINGLE main emotion. Do NOT answer neutral unless the face is truly "
    'expressionless. Return only JSON like {"emotion": "angry"}. Choose exactly '
    "one of: happy, sad, angry, surprised, tired, neutral.")

_EMO_TO_MOOD = {
    "happy":     ("positive", 0.85, "low",    "Customer looks cheerful — keep it warm and light."),
    "surprised": ("positive", 0.65, "low",    "Customer looks surprised — be friendly and clear."),
    "neutral":   ("neutral",  0.5,  "medium", "Neutral customer — be polite and efficient."),
    "tired":     ("neutral",  0.4,  "medium", "Customer looks tired — be gentle and quick."),
    "sad":       ("negative", 0.3,  "medium", "Customer looks down — be gentle and reassuring."),
    "angry":     ("negative", 0.2,  "high",   "Customer seems unhappy — apologise and be quick."),
}
_EMO_ALIASES = {
    "happiness": "happy", "joy": "happy", "smiling": "happy", "smile": "happy",
    "sadness": "sad", "unhappy": "sad", "anger": "angry", "mad": "angry",
    "surprise": "surprised", "shocked": "surprised", "sleepy": "tired",
    "calm": "neutral", "serious": "neutral", "fear": "sad", "disgust": "angry",
}

def _mood_from_emotion(emotion, source):
    emotion = (emotion or "").strip().lower()
    emotion = _EMO_ALIASES.get(emotion, emotion)
    sentiment, score, urgency, note = _EMO_TO_MOOD.get(
        emotion, _EMO_TO_MOOD["neutral"])
    if emotion not in _EMO_TO_MOOD:
        emotion = "neutral"
    return {"emotion": emotion, "sentiment": sentiment,
            "sentiment_score": score, "urgency": urgency,
            "empathy_note": note, "source": source}

def _normalise_mood(raw, source):
    """Extract and normalise the mood JSON from an Ollama `response` string."""
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"no JSON in reply: {raw[:120]}")
    text = re.sub(r"//[^\n]*", "", m.group(0))
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    mood = json.loads(text)
    mood.setdefault("emotion", "neutral")
    mood.setdefault("sentiment", "neutral")
    mood.setdefault("sentiment_score", 0.5)
    mood.setdefault("urgency", "medium")
    mood.setdefault("empathy_note", "")
    mood["source"] = source
    return mood

def _encode_frame(frame):
    """JPEG-encode a BGR frame to base64 (or None if OpenCV/encoding fails)."""
    if not _CV_OK or frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode()

def _vision_emotion(best_frame, speech, timeout=None):
    """Send the face frame to the Ollama VISION model and return a mood dict.

    Returns None (so the caller falls back to the text path) if no vision model
    is configured, the frame can't be encoded, or the call/parse fails/times out.
    """
    if not VISION_MODEL or best_frame is None:
        return None
    b64 = _encode_frame(best_frame)
    if b64 is None:
        return None
    if timeout is None:
        timeout = VISION_TIMEOUT

    payload = {"model": VISION_MODEL, "prompt": _VISION_PROMPT, "images": [b64],
               "stream": False, "format": "json", "keep_alive": "10m",
               "options": {"temperature": 0.2}}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        raw = r.json().get("response", "").strip().lower()
        m = re.search(r'"emotion"\s*:\s*"([a-z ]+)"', raw)
        emotion = m.group(1).strip() if m else None
        if emotion is None or (emotion not in _EMO_TO_MOOD
                               and emotion not in _EMO_ALIASES):
            words = re.findall(r"[a-z]+", raw)
            emotion = next((w for w in words
                            if w in _EMO_TO_MOOD or w in _EMO_ALIASES), None)
        if emotion is None:
            raise ValueError(f"no emotion word in reply: {raw[:80]}")
        mood = _mood_from_emotion(emotion, f"ollama-vision/{VISION_MODEL}")
        print(f"👁️🧠 Vision model {VISION_MODEL} read the face as: {mood['emotion']}")
        return mood
    except Exception as e:
        print(f"Vision model ({VISION_MODEL}) unavailable: {e} "
              "- falling back to text reasoning")
        return None

def quick_mood(facial, speech):
    """Instant, rule-based mood (NO LLM) — used to dispatch the robot without
    waiting on Ollama. The richer analyze_sentiment() runs afterwards for tone."""
    return _fallback_mood(facial or {}, speech)

def analyze_sentiment(facial, speech, timeout=25):
    """Fuse the facial signal and the speech into a mood dict.
    
    Uses EMOTION_VISION_MODEL on the captured frame when set, otherwise the text model
    on the OpenCV cue plus the speech, otherwise a rule-based read."""
    facial = facial or {}

    if VISION_MODEL and facial.get("best_frame") is not None:
        mood = _vision_emotion(facial.get("best_frame"), speech)
        if mood is not None:
            return mood

    prompt = _SENTIMENT_PROMPT.format(
        emotion=facial.get("emotion", facial.get("expression", "no_face")),
        emo_pct=int(round(facial.get("emotion_conf", 0.0) * 100)),
        face_found=str(facial.get("face_found", False)).lower(),
        speech=(speech or "").replace('"', "'") or "(said nothing)")

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "format": "json", "keep_alive": "10m",
               "options": {"temperature": 0.2, "num_predict": 160}}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        return _normalise_mood(raw, f"ollama/{OLLAMA_MODEL}")
    except Exception as e:
        print("emotion LLM error:", e, "- using rule-based mood")
        return _fallback_mood(facial, speech)

def read_mood(speech, seconds=None):
    """Convenience: read the face from the webcam, then fuse with `speech`.

    Returns the mood dict from analyze_sentiment(). Respects EMOTION_ENABLED.
    """
    if not EMOTION_ENABLED:
        return _fallback_mood({}, speech)
    facial = read_facial_expression(seconds=seconds)
    return analyze_sentiment(facial, speech)

if __name__ == "__main__":
    f = read_facial_expression()
    print("facial:", {k: v for k, v in f.items() if k != "best_frame"})
    print("mood:", analyze_sentiment(f, "hi, can I get a coke please"))
