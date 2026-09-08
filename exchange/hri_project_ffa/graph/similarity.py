
import difflib
import math
import os

_W2V = None
if os.environ.get("SG_USE_WORD2VEC", "0") == "1":
    try:
        import gensim.downloader
        _W2V = gensim.downloader.load("word2vec-google-news-300")
        print("[similarity] word2vec-google-news-300 loaded")
    except Exception as e:
        print("[similarity] word2vec unavailable, lexical fallback:", e)

_SYNONYMS = [
    {"bottle", "can", "drink", "soda", "beverage"},
    {"cup", "mug", "glass", "wine glass"},
    {"person", "human", "customer", "man", "woman"},
    {"table", "desk", "counter"},
    {"chips", "crisps", "pringles", "snack"},
]

WEIGHTS = {"label": 0.45, "color": 0.20, "material": 0.10, "description": 0.25}

def _norm(s):
    return (s or "").lower().strip().replace("_", " ")

def word_similarity(a, b):
    """Cosine word similarity with word2vec, else lexical fallback in [0, 1]."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if _W2V is not None:
        try:
            ta = [t for t in a.split() if t in _W2V]
            tb = [t for t in b.split() if t in _W2V]
            if ta and tb:
                return float(max(0.0, _W2V.n_similarity(ta, tb)))
        except Exception:
            pass
    for group in _SYNONYMS:
        if a in group and b in group:
            return 0.9
    if a in b or b in a:
        return 0.85
    return 0.8 * difflib.SequenceMatcher(None, a, b).ratio()

def color_similarity(rgb1, rgb2):
    """1 - normalized Euclidean distance in RGB space (slides' color metric)."""
    if not rgb1 or not rgb2:
        return None
    d = math.dist(rgb1, rgb2)
    return 1.0 - d / math.sqrt(3 * 255 ** 2)

def description_similarity(d1, d2):
    """Token-set Jaccard as a lightweight stand-in for text embeddings."""
    t1 = set(_norm(d1).split())
    t2 = set(_norm(d2).split())
    if not t1 or not t2:
        return None
    return len(t1 & t2) / len(t1 | t2)

def lost_similarity(obj_a, obj_b, weights=WEIGHTS):
    """Combined LOST similarity in [0, 1] between two objects carrying label, color_rgb,
    material and description. Missing components are skipped and the weights
    renormalised."""
    parts = {
        "label": word_similarity(obj_a.label, obj_b.label),
        "color": color_similarity(obj_a.color_rgb, obj_b.color_rgb),
        "material": word_similarity(obj_a.material, obj_b.material)
                    if obj_a.material and obj_b.material else None,
        "description": description_similarity(obj_a.description, obj_b.description),
    }
    score, total_w = 0.0, 0.0
    for name, value in parts.items():
        if value is None:
            continue
        score += weights[name] * value
        total_w += weights[name]
    return score / total_w if total_w > 0 else 0.0
