
from .similarity import word_similarity

def _matches_constraint(kg, sg, node, constraint):
    c = str(constraint).lower().strip()
    if not c:
        return True
    for rel in ("hasColor", "hasMaterial", "hasShape"):
        if any(t == c for _, _, t in kg.query(h=node.node_id, r=rel)):
            return True
    if kg.ask(node.node_id, "isA", c.capitalize()) or kg.ask(node.node_id, "isA", c):
        return True
    for prefix, rel in (("near ", "near"), ("on ", "isOnTopOf")):
        if c.startswith(prefix):
            ref = c[len(prefix):]
            for s, r, t in sg.edges:
                if r != rel and not (rel == "near"):
                    continue
                pair = None
                if s == node.node_id:
                    pair = t
                elif t == node.node_id and rel == "near":
                    pair = s
                if pair and word_similarity(sg.nodes[pair].label, ref) > 0.8:
                    return True
            return False
    return False

def find_target_in_graph(sg, kg, request, exclude=()):
    """Resolve {'target', 'constraints'} to the best ObjectNode, matching by label
    similarity or ontology class (target='drink' matches anything inferred to be a
    Drink) and preferring confirmed, closer objects. `exclude` skips node_ids a previous
    fetch failed on, so a retry re-grounds to a different physical object."""
    target = str(request.get("target", "")).lower().strip()
    if not target:
        return None
    candidates = []
    for node in sg.nodes.values():
        if node.node_id in exclude:
            continue
        label_ok = word_similarity(node.label, target) > 0.75
        class_ok = kg.ask(node.node_id, "isA", target.capitalize()) is True
        if label_ok or class_ok:
            candidates.append(node)
    for c in request.get("constraints", []) or []:
        candidates = [n for n in candidates if _matches_constraint(kg, sg, n, c)]
    if not candidates:
        return None

    def score(n):
        s = n.confidence + 0.1 * min(n.times_seen, 5)
        if n.status == "uncertain":
            s -= 1.0
        if n.distance is not None:
            s -= 0.05 * n.distance
        return s
    return max(candidates, key=score)

def answer_location(sg, node):
    """Human-readable answer about where an object is (uses graph edges)."""
    if node is None:
        return "I don't have that object in my scene graph."
    parts = [f"The {node.label} ({node.node_id})"]
    if node.centroid:
        x, y, z = node.centroid
        if getattr(node, "frame", "camera") == "map":
            parts.append(f"is at map position x={x}, y={y}")
        else:
            parts.append(f"is {z} m in front of me"
                         + (", to my left" if x < -0.1 else
                            ", to my right" if x > 0.1 else ", straight ahead"))
    rels = {"near": "near the", "isOnTopOf": "on top of the",
            "isLeftOf": "left of the"}
    for s, r, t in sg.edges:
        if s == node.node_id and t in sg.nodes:
            parts.append(f"{rels.get(r, r)} {sg.nodes[t].label}")
    if node.status == "uncertain":
        parts.append("(but I have not seen it recently)")
    return " ".join(parts) + "."
