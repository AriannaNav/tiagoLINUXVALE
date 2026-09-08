
import json
import math
import os
import time

import cv2
import numpy as np

from .object_node import Detection

HERE = os.path.dirname(os.path.abspath(__file__))
EXCHANGE = os.path.abspath(os.path.join(HERE, "..", ".."))
RGB_FRAME_PATH = os.environ.get("SG_RGB", os.path.join(EXCHANGE, "robot_frame.jpg"))
CAMERA_INFO_PATH = os.environ.get("SG_CAMINFO", os.path.join(EXCHANGE, "camera_info.json"))
POSE_PATH = os.environ.get("SG_POSE", os.path.join(EXCHANGE, "robot_pose.json"))
POSE_MAX_AGE = float(os.environ.get("SG_POSE_MAX_AGE", "5.0"))

SEMANTIC_PRIORS = {
    "bottle":      dict(material="plastic", shape="cylinder", is_movable=True,
                        description="a drink bottle that can be grasped and served",
                        affordances=("grasped", "carried", "pouredFrom", "served")),
    "cup":         dict(material="ceramic", shape="cylinder", is_movable=True,
                        description="a cup for drinks",
                        affordances=("grasped", "carried", "filled")),
    "wine glass":  dict(material="glass", shape="cylinder", is_movable=True,
                        description="a glass for drinks",
                        affordances=("grasped", "carried", "filled")),
    "person":      dict(material="organic", shape="human", is_movable=True,
                        description="a customer in the bar",
                        affordances=("greeted", "askedForOrder", "servedTo")),
    "table":       dict(material="wood", shape="box", is_movable=False,
                        description="a table where customers sit",
                        affordances=("approached", "usedAsSupport")),
    "chair":       dict(material="wood", shape="box", is_movable=False,
                        description="a chair",
                        affordances=("satOn",)),
    "book":        dict(material="paper", shape="box", is_movable=True,
                        description="a book",
                        affordances=("grasped", "carried")),
    "vase":        dict(material="ceramic", shape="cylinder", is_movable=True,
                        description="a decorative vase",
                        affordances=("grasped",)),
}
_DEFAULT_PRIOR = dict(material="unknown", shape="unknown", is_movable=True,
                      description="", affordances=())

_COLOR_NAMES = [
    ((0, 14), "red"), ((14, 22), "orange"), ((22, 38), "yellow"),
    ((38, 85), "green"), ((85, 130), "blue"), ((130, 160), "purple"),
    ((160, 180), "red"),
]

class DepthCamera:
    """Pinhole back-projection using the shared depth image + intrinsics
    (same conventions as exchange/codice/detect_and_plan.py)."""

    def __init__(self):
        self.fx = self.fy = 460.0
        self.cx, self.cy = 320.0, 240.0
        if os.path.exists(CAMERA_INFO_PATH):
            try:
                with open(CAMERA_INFO_PATH) as f:
                    info = json.load(f)
                self.fx = info.get("fx", self.fx)
                self.fy = info.get("fy", self.fy)
                self.cx = info.get("cx", self.cx)
                self.cy = info.get("cy", self.cy)
            except (OSError, json.JSONDecodeError):
                pass

    def in_fov(self, centroid, margin=20, max_range=8.0):
        """POV-volume check for the disappearance detector: does a stored 3D
        point project inside the current image (i.e. should we see it)?"""
        if centroid is None:
            return False
        x, y, z = centroid
        if z <= 0.05 or z > max_range:
            return False
        px = self.fx * x / z + self.cx
        py = self.fy * y / z + self.cy
        w, h = int(self.cx * 2), int(self.cy * 2)
        return margin <= px < w - margin and margin <= py < h - margin

def _quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

_FALLBACK_CAM_T = np.array([0.22, 0.0, 1.12])
_FALLBACK_CAM_R = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])

class RobotPose:
    """T_map_camera composed from the robot's ground-truth base pose
    (robot_pose.json, written by frame_grabber_tiago.py) and the
    base->camera extrinsic (TF-captured, else the fixed TIAGo approximation).
    Lets perception express object centroids in the MAP frame, so the graph
    survives robot motion and the bridge can navigate to what was seen."""

    def __init__(self):
        self.R = None
        self.t = None
        self.base_xy_yaw = None

    @property
    def valid(self):
        return self.R is not None

    def refresh(self):
        try:
            with open(POSE_PATH) as f:
                d = json.load(f)
            if time.time() - float(d.get("stamp", 0)) > POSE_MAX_AGE:
                raise ValueError("stale pose")
            yaw = float(d["yaw"])
            Rb = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                           [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.]])
            tb = np.array([float(d["x"]), float(d["y"]), float(d.get("z", 0.0))])
            ext = d.get("camera_extrinsic")
            if ext:
                Rc = _quat_to_mat(*ext["q"])
                tc = np.array(ext["t"], dtype=float)
            else:
                Rc, tc = _FALLBACK_CAM_R, _FALLBACK_CAM_T
            self.R = Rb @ Rc
            self.t = Rb @ tc + tb
            self.base_xy_yaw = (tb[0], tb[1], yaw)
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            self.R = None
            self.t = None
            self.base_xy_yaw = None
            return False

    def map_to_cam(self, p):
        q = self.R.T @ (np.asarray(p, dtype=float) - self.t)
        return (float(q[0]), float(q[1]), float(q[2]))

def dominant_color(frame, bbox):
    """(name, (r, g, b)) of the dominant color inside the bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return "", None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = [np.median(hsv[:, :, i]) for i in range(3)]
    b, g, r = [int(np.median(roi[:, :, i])) for i in range(3)]
    if s >= 60:
        for (lo, hi), name in _COLOR_NAMES:
            if lo <= h < hi:
                return name, (r, g, b)
    if v < 50:
        return "black", (r, g, b)
    if s < 40:
        return "white" if v > 180 else "gray", (r, g, b)
    for (lo, hi), name in _COLOR_NAMES:
        if lo <= h < hi:
            return name, (r, g, b)
    return "", (r, g, b)

def characterize(label, frame, bbox, confidence, pos3d, dist, source,
                 depth_source="measured"):
    """Fuse semantic priors + measured color + geometry into a Detection."""
    base = label.replace("_", " ")
    prior = SEMANTIC_PRIORS.get(base.split()[-1], SEMANTIC_PRIORS.get(base, _DEFAULT_PRIOR))
    cname, crgb = dominant_color(frame, bbox) if bbox is not None else ("", None)
    desc = prior["description"]
    if cname and desc and not base.startswith(cname):
        desc = f"a {cname} {base}: {desc}"
    elif desc:
        desc = f"a {base}: {desc}"
    return Detection(label=base, confidence=confidence, bbox=tuple(bbox) if bbox is not None else None,
                     centroid=pos3d, distance=dist,
                     color_name=cname, color_rgb=crgb,
                     material=prior["material"], shape=prior["shape"],
                     is_movable=prior["is_movable"], description=desc,
                     source=source, depth_source=depth_source,
                     affordances=tuple(prior.get("affordances", ())))

class PerceptionModule:
    """Frame, pose and field-of-view utilities for the scene-graph loop.
    Object identity comes from graph/vlm_perception.py (Gemini), not from here."""

    def __init__(self):
        self.depth_cam = DepthCamera()
        self.pose = RobotPose()

    def read_bridge_frame(self):
        if not os.path.exists(RGB_FRAME_PATH):
            return None
        return cv2.imread(RGB_FRAME_PATH)

    def in_view(self, centroid, frame="map"):
        """POV-volume check for the disappearance detector, frame-aware:
        map-frame points are converted back to the CURRENT camera pose."""
        if centroid is None:
            return False
        if frame == "map":
            if not self.pose.valid:
                return False
            centroid = self.pose.map_to_cam(centroid)
        return self.depth_cam.in_fov(tuple(centroid))

