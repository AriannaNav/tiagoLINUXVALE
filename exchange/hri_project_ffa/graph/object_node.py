
import time
from dataclasses import dataclass, field, asdict

@dataclass
class Detection:
    """A single-frame observation produced by the PerceptionModule."""
    label: str
    confidence: float = 1.0
    bbox: tuple = None
    centroid: tuple = None
    frame: str = "camera"
    distance: float = None
    color_name: str = ""
    color_rgb: tuple = None
    material: str = ""
    shape: str = ""
    is_movable: bool = True
    description: str = ""
    source: str = "camera"
    depth_source: str = "measured"
    affordances: tuple = ()

@dataclass
class ObjectNode:
    """A node of the persistent 3D scene graph."""
    node_id: str
    label: str
    confidence: float = 1.0
    bbox: tuple = None
    centroid: tuple = None
    frame: str = "camera"
    distance: float = None
    color_name: str = ""
    color_rgb: tuple = None
    material: str = ""
    shape: str = ""
    is_movable: bool = True
    description: str = ""
    source: str = "camera"
    depth_source: str = "measured"
    affordances: tuple = ()
    is_minor: bool = False
    status: str = "confirmed"
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    times_seen: int = 1

    @classmethod
    def from_detection(cls, node_id, det: Detection):
        return cls(node_id=node_id, label=det.label, confidence=det.confidence,
                   bbox=det.bbox, centroid=det.centroid, frame=det.frame,
                   distance=det.distance,
                   color_name=det.color_name, color_rgb=det.color_rgb,
                   material=det.material, shape=det.shape,
                   is_movable=det.is_movable, description=det.description,
                   source=det.source, depth_source=det.depth_source,
                   affordances=tuple(det.affordances))

    def update_from(self, det: Detection):
        """UPDATE_IN: refresh pose + attributes from a matched detection."""
        self.bbox = det.bbox or self.bbox
        better = (det.depth_source == "measured"
                  or self.depth_source != "measured")
        frame_ok = det.frame == self.frame or det.frame == "map"
        if det.centroid and ((better and frame_ok) or self.centroid is None):
            self.centroid = det.centroid
            self.frame = det.frame
            self.distance = det.distance
            self.depth_source = det.depth_source
        self.confidence = max(self.confidence, det.confidence)
        if det.color_rgb:
            self.color_rgb, self.color_name = det.color_rgb, det.color_name
        if det.affordances:
            self.affordances = tuple(sorted(set(self.affordances)
                                            | set(det.affordances)))
        self.status = "confirmed"
        self.last_seen = time.time()
        self.times_seen += 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("bbox", "centroid", "color_rgb", "affordances"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)
