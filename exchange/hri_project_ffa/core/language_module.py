
import json
import os
import re
import requests
import whisper
import tempfile
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

from config import OLLAMA_MODEL, OLLAMA_URL

OBJECT_ALIASES = {
    "phone": "cell phone",
    "cell": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "telephone": "cell phone",
    "cellular": "cell phone",

    "mug": "cup",

    "human": "person",
    "man": "person",
    "woman": "person",
}

whisper_model = whisper.load_model(
    os.environ.get("WHISPER_MODEL", "small"), device="cpu")

def normalize_target_name(target):
    """Strip articles and apply OBJECT_ALIASES, so "the mug" becomes "cup"."""
    if not target:
        return ""

    target = target.lower().strip()

    target = re.sub(
        r"\b(the|a|an|my|your|his|her|its|their|me)\b",
        "",
        target
    )

    target = re.sub(r"\s+", " ", target).strip()

    return OBJECT_ALIASES.get(target, target)

def build_unknown_request():
    return {
        "intent": "unknown",
        "target": "",
        "constraints": [],
        "ambiguous": True
    }

def build_find_request(target):
    return {
        "intent": "find_object",
        "target": target,
        "constraints": [],
        "ambiguous": False
    }

def parse_simple_find_command(text):
    """Deterministic parser for simple commands like "find the cup", so they do not
    depend on LLM generation. Commands containing spatial words are left to the LLM,
    which can see the scene text."""

    if not text:
        return None

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    spatial_words = [
        "left",
        "right",
        "center",
        "middle",
        "near",
        "next to",
        "behind",
        "front"
    ]

    if any(word in text for word in spatial_words):
        return None

    if text in ["find", "locate", "search", "look", "look for", "search for"]:
        return build_unknown_request()

    patterns = [
        r"^(find|locate)\s+(.+)$",
        r"^(search for|look for)\s+(.+)$",
        r"^(can you find|can you locate|please find|please locate)\s+(.+)$",
        r"^(can you search for|can you look for|please search for|please look for)\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text)

        if match:
            raw_target = match.group(2)
            target = normalize_target_name(raw_target)

            if target in ["", "find", "locate", "search", "look"]:
                return build_unknown_request()

            return build_find_request(target)

    return None

class MicUnavailable(RuntimeError):
    """The capture stream would not start. On WSLg the PulseAudio bridge
    times out now and then; that must cost one turn, never the session."""

def _open_stream(seconds, fs, latency):
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1,
                   dtype=np.float32, latency=latency)
    sd.wait()
    return audio

def record_audio(seconds=5, fs=16000):
    """Record from the microphone, cleaning NaN/Inf for Whisper.

    Retries with a relaxed latency before giving up: a stream that will not
    open used to raise through the interaction loop and abort the process."""

    print(chr(10) + "🎤 Recording...")

    audio = None
    for latency in ("low", "high"):
        try:
            audio = _open_stream(seconds, fs, latency)
            break
        except Exception as e:
            print("   (microphone did not start, latency=%s: %s)" % (latency, e))
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
    if audio is None:
        raise MicUnavailable("capture stream would not start")

    audio = np.nan_to_num(audio)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 0.02:
        return np.zeros(1, dtype=np.float32), fs

    audio = audio / peak
    return audio, fs

def listen_user_command():
    """Record, normalise and transcribe the user's command with local Whisper on CPU.
    No cloud STT, no internet dependency."""

    audio, fs = record_audio()

    if len(audio) < fs * 1:
        print("Audio too short, ignoring.")
        return ""

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            wav.write(temp_audio.name, fs, audio)

            result = whisper_model.transcribe(
                temp_audio.name,
                language="en",
                fp16=False
            )

            text = result.get("text", "").strip()

        print("🗣️ You said:", text)
        return text

    except Exception as e:
        print("STT error:", e)
        return ""

def parse_user_command_with_llm(text, scene_text=""):
    """Convert the spoken command into structured JSON via Ollama:
    {intent, target, constraints, ambiguous}."""

    simple_parse = parse_simple_find_command(text)

    if simple_parse is not None:
        print("\nParsed with simple command parser:")
        print(simple_parse)
        return simple_parse

    prompt = f"""
You are a robot assistant.

Scene:
{scene_text}

User command:
{text}

Your task is to understand what object the user wants the robot to find.

Return ONLY valid JSON.

Use this exact format:
{{
  "intent": "find_object",
  "target": "object_name",
  "constraints": [],
  "ambiguous": false
}}

Rules:
- If the command is not about finding an object, use intent "unknown".
- If the object is visible in the scene, use the exact object label from the scene.
- If the user refers to position, such as left, center, or right, choose the object in that position.
- If the user says "mug", use "cup".
- If the user says "human", "man", or "woman", use "person".
- If there is not enough information, set ambiguous to true.
- Remove articles like "the", "a", "an", "my", "your" from the target.
- Do not explain.
- Do not add text outside the JSON.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    response.raise_for_status()

    raw_output = response.json()["response"].strip()

    print("\nRaw LLM output:")
    print(raw_output)

    json_match = re.search(r"\{[\s\S]*?\}", raw_output)

    if not json_match:
        raise ValueError(f"LLM did not return JSON: {raw_output}")

    json_text = json_match.group(0)
    json_text = re.sub(r"//[^\n]*", "", json_text)
    json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)
    parsed = json.loads(json_text)

    if "intent" not in parsed:
        parsed["intent"] = "unknown"

    if "target" not in parsed:
        parsed["target"] = ""

    if "constraints" not in parsed:
        parsed["constraints"] = []

    if "ambiguous" not in parsed:
        parsed["ambiguous"] = False

    parsed["target"] = normalize_target_name(parsed["target"])

    if parsed["intent"] == "find_object" and not parsed["target"]:
        return build_unknown_request()

    return parsed

if __name__ == "__main__":
    command_text = listen_user_command()
    parsed = parse_user_command_with_llm(command_text)

    print("\nParsed command:")
    print(parsed)
