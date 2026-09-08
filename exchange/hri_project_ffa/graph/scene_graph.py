
import json
import os
import time

NEAR_THRESHOLD = 0.7

class SceneGraph3D:
    def __init__(self, frame_id="camera"):
        self.frame_id = frame_id
        self.nodes = {}
        self.edges = []

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def remove_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if node_id not in (e[0], e[2])]

    SPATIAL_RELATIONS = ("near", "isOnTopOf", "isLeftOf", "isRightOf")

    def refresh_relations(self):
        """Recompute spatial edges from the current geometry. Frame-aware:
        camera frame (x right, y DOWN, z forward) vs map frame (z UP).
        Semantic edges injected from outside (e.g. isAtTable) are not
        geometric — keep them, dropping only those with a deleted endpoint."""
        self.edges = [e for e in self.edges
                      if e[1] not in self.SPATIAL_RELATIONS
                      and e[0] in self.nodes and e[2] in self.nodes]
        items = [n for n in self.nodes.values() if n.centroid]
        for a in items:
            for b in items:
                if a.node_id >= b.node_id:
                    continue
                if getattr(a, "frame", "camera") != getattr(b, "frame", "camera"):
                    continue
                ax, ay, az = a.centroid
                bx, by, bz = b.centroid
                d = ((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2) ** 0.5
                lateral = False
                if getattr(a, "frame", "camera") == "map":
                    horiz = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
                    up = bz - az
                    lower, upper = (a, b) if up > 0 else (b, a)
                    on_r = 1.2 if getattr(lower, "shape", "") == "surface" \
                        else 0.25
                    if (horiz < on_r and 0.10 < abs(up) < 1.5
                            and getattr(upper, "label", "") != "person"):
                        self.edges.append((upper.node_id, "isOnTopOf",
                                           lower.node_id))
                    if (getattr(a, "shape", "") == "bottle"
                            and getattr(b, "shape", "") == "bottle"
                            and abs(by - ay) < 0.6 and abs(bz - az) < 0.4):
                        dx = bx - ax
                        left, right = ((a, b) if 0.15 < dx < 0.8 else
                                       (b, a) if -0.8 < dx < -0.15 else
                                       (None, None))
                        if left is not None:
                            self.edges.append((left.node_id, "isLeftOf",
                                               right.node_id))
                            self.edges.append((right.node_id, "isRightOf",
                                               left.node_id))
                            lateral = True
                else:
                    dx, dy = bx - ax, by - ay
                    if abs(dx) < 0.25 and abs(bz - az) < 0.4:
                        if dy > 0.10:
                            self.edges.append((a.node_id, "isOnTopOf", b.node_id))
                        elif dy < -0.10:
                            self.edges.append((b.node_id, "isOnTopOf", a.node_id))
                    elif abs(dx) > 0.15:
                        left, right = (a, b) if dx > 0 else (b, a)
                        self.edges.append((left.node_id, "isLeftOf",
                                           right.node_id))
                        self.edges.append((right.node_id, "isRightOf",
                                           left.node_id))
                        lateral = True
                if not lateral and d < NEAR_THRESHOLD:
                    self.edges.append((a.node_id, "near", b.node_id))

    def objects_by_label(self, label):
        from .similarity import word_similarity
        return [n for n in self.nodes.values()
                if word_similarity(n.label, label) > 0.8]

    def to_triples(self):
        """Export the graph as (h, r, t) triples for the KnowledgeGraph ABox."""
        triples = []
        for n in self.nodes.values():
            triples.append((n.node_id, "hasLabel", n.label))
            if n.color_name:
                triples.append((n.node_id, "hasColor", n.color_name))
            if n.material and n.material != "unknown":
                triples.append((n.node_id, "hasMaterial", n.material))
            if n.shape and n.shape != "unknown":
                triples.append((n.node_id, "hasShape", n.shape))
            triples.append((n.node_id, "isMovable", "yes" if n.is_movable else "no"))
            triples.append((n.node_id, "hasStatus", n.status))
            for a in getattr(n, "affordances", ()):
                triples.append((n.node_id, "canBe", a))
        triples.extend(self.edges)
        return triples

    def to_text(self):
        """Readable scene description for the LLM brain (grounding context)."""
        if not self.nodes:
            return "The scene graph is empty."
        lines = [f"Scene graph ({len(self.nodes)} objects, frame '{self.frame_id}'):"]
        for n in self.nodes.values():
            attrs = [a for a in (n.color_name, n.material if n.material != "unknown" else "")
                     if a and a not in n.label]
            desc = f"- {n.node_id}: {' '.join(attrs)} {n.label}".rstrip()
            if n.centroid:
                x, y, z = n.centroid
                where = "at map position" if getattr(n, "frame", "camera") == "map" \
                    else "at (camera frame)"
                desc += f" {where} (x={x}, y={y}, z={z}) m"
                if n.depth_source == "prior":
                    desc += " (depth assumed from the bar layout)"
            if n.distance is not None:
                desc += f", {n.distance} m away"
            if n.status == "uncertain":
                desc += " [uncertain: not seen recently]"
            lines.append(desc)
        rel_names = {"near": "is near", "isOnTopOf": "is on top of",
                     "isLeftOf": "is left of", "isRightOf": "is right of"}
        for s, r, t in self.edges:
            lines.append(f"- {s} {rel_names.get(r, r)} {t}")
        return "\n".join(lines)

    def save(self, path):
        data = {"frame_id": self.frame_id, "stamp": time.time(),
                "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": self.edges}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        from .object_node import ObjectNode
        with open(path) as f:
            data = json.load(f)
        sg = cls(frame_id=data.get("frame_id", "camera"))
        for nd in data.get("nodes", []):
            sg.add_node(ObjectNode.from_dict(nd))
        sg.edges = [tuple(e) for e in data.get("edges", [])]
        return sg
