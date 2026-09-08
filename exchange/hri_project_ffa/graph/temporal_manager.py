
import math

from .object_node import ObjectNode
from .similarity import lost_similarity

SIM_THRESHOLD = 0.55
DIST_THRESHOLD = 0.50
IOU_THRESHOLD = 0.20

def _dist(a, b):
    if a is None or b is None:
        return None
    return math.dist(a, b)

def _iou(a, b):
    if a is None or b is None:
        return None
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area

class TemporalManager:
    def __init__(self, scene_graph, sim_th=SIM_THRESHOLD,
                 dist_th=DIST_THRESHOLD, iou_th=IOU_THRESHOLD, verbose=True):
        self.sg = scene_graph
        self.sim_th = sim_th
        self.dist_th = dist_th
        self.iou_th = iou_th
        self.verbose = verbose
        self.phase = "exploration"
        self._next_id = 0
        self.ops_log = []

    def start_tracking(self):
        """Exploration ended: switch to the tracking phase."""
        self.phase = "tracking"

    def _new_id(self, label):
        self._next_id += 1
        return f"{label.replace(' ', '_')}_{self._next_id}"

    def _log(self, op, node, detail=""):
        entry = (op, node.node_id, node.label, detail)
        self.ops_log.append(entry)
        if self.verbose:
            print(f"[scene-graph] {op:<10} {node.node_id:<20} {detail}")

    def _best_match(self, det, unmatched_nodes):
        """Highest LOST similarity, with 3D distance as tie-breaker.
        
        Two identical-looking objects a short distance apart score almost the same, and
        without the tie-break the matcher can pick the far twin and drag its node across
        the map."""
        best, best_sim, best_d = None, 0.0, None
        for node in unmatched_nodes:
            sim = lost_similarity(det, node)
            if sim <= 0.0:
                continue
            same = getattr(det, "frame", "camera") == \
                getattr(node, "frame", "camera")
            d = _dist(det.centroid, node.centroid) if same else None
            better = sim > best_sim + 0.05
            tie = abs(sim - best_sim) <= 0.05 and best is not None
            closer = (d is not None and
                      (best_d is None or d < best_d))
            if best is None or better or (tie and closer):
                best, best_sim, best_d = node, max(sim, best_sim), d
        return best, best_sim

    def _add(self, det):
        node = ObjectNode.from_detection(self._new_id(det.label), det)
        self.sg.add_node(node)
        self._log("ADD", node, f"at {det.centroid}")
        return node

    def _update_in(self, node, det):
        node.update_from(det)
        self._log("UPDATE_IN", node, f"-> {det.centroid}")

    def _update_out(self, node, det):
        new_node = self._add(det)
        node.status = "uncertain"
        self._log("UPDATE_OUT", node, f"moved? old pose {node.centroid} kept as uncertain")
        return new_node

    def _delete(self, node):
        self.sg.remove_node(node.node_id)
        self._log("DELETE", node, "not found at expected location")

    def update(self, detections, fov_check=None):
        """Integrate this frame's detections into the scene graph.
        
        fov_check(centroid) -> bool tells whether a stored object should be visible now.
        Without it nothing is ever deleted, which is the open-world default."""
        self.ops_log.clear()
        unmatched = set(self.sg.nodes)
        for det in detections:
            candidates = [self.sg.nodes[i] for i in unmatched]
            node, sim = self._best_match(det, candidates)
            if node is None or sim <= self.sim_th:
                self._add(det)
                continue
            unmatched.discard(node.node_id)
            if self.phase == "exploration":
                same_frame = getattr(det, "frame", "camera") == \
                    getattr(node, "frame", "camera")
                d = _dist(det.centroid, node.centroid) if same_frame else None
                if d is not None and d > self.dist_th:
                    unmatched.add(node.node_id)
                    self._add(det)
                else:
                    self._update_in(node, det)
                continue
            same_frame = getattr(det, "frame", "camera") == \
                getattr(node, "frame", "camera")
            d = _dist(det.centroid, node.centroid) if same_frame else None
            if d is None:
                iou = _iou(det.bbox, node.bbox)
                close = iou is not None and iou >= self.iou_th
            else:
                close = d < self.dist_th
            if close:
                self._update_in(node, det)
            else:
                self._update_out(node, det)

        if self.phase == "tracking" and fov_check is not None:
            for node_id in list(unmatched):
                node = self.sg.nodes.get(node_id)
                if node is None:
                    continue
                if getattr(node, "source", "") in ("ground_truth", "patrol",
                                                   "causal_rule"):
                    continue
                try:
                    visible = fov_check(node.centroid, frame=node.frame)
                except TypeError:
                    visible = fov_check(node.centroid)
                if not visible:
                    continue
                if node.status == "uncertain":
                    self._delete(node)
                else:
                    node.status = "uncertain"
                    self._log("UNCERTAIN", node, "expected in FOV but not detected")

        self.sg.refresh_relations()
        return list(self.ops_log)
