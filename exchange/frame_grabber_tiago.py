import json
import math
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

try:
    from tf2_ros import Buffer, TransformListener
    TF_OK = True
except ImportError:
    TF_OK = False

OUT_DIR = os.environ.get("FRAME_DIR", "/root/exchange/exchange")
RGB_PATH = os.path.join(OUT_DIR, "robot_frame.jpg")
INFO_PATH = os.path.join(OUT_DIR, "camera_info.json")
POSE_PATH = os.path.join(OUT_DIR, "robot_pose.json")
WRITE_PERIOD = float(os.environ.get("FRAME_PERIOD", "0.4"))
EXTRINSIC_REFRESH = 10.0

_write_failures = [0]

def atomic_write(path, img, params=None):
    """Best-effort write. The bind-mounted Windows filesystem occasionally
    refuses a write (Errno 12) under load; that must cost one frame, never the
    whole node — a dead grabber blinds every perception downstream."""
    tmp = path + ".tmp" + os.path.splitext(path)[1]
    try:
        if cv2.imwrite(tmp, img, params or []):
            os.replace(tmp, path)
        _write_failures[0] = 0
    except OSError as e:
        _write_failures[0] += 1
        if _write_failures[0] % 25 == 1:
            print("[frame_grabber] write failed (%s), continuing" % e, flush=True)

class FrameGrabberTiago(Node):
    def __init__(self):
        super().__init__("frame_grabber_tiago")
        self.bridge = CvBridge()
        self._last_rgb = 0.0
        self._last_pose = 0.0
        self._info_written = False
        self._camera_frame = None
        self._extrinsic = None
        self._extrinsic_stamp = 0.0

        if TF_OK:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, "/head_front_camera/rgb/image_raw",
                                 self.on_rgb, qos_profile_sensor_data)

        self.create_subscription(Odometry, "/ground_truth_odom",
                                 self.on_odom, 10)
        for topic in ("/head_front_camera/rgb/camera_info",
                      "/head_front_camera/camera_info"):
            self.create_subscription(CameraInfo, topic, self.on_info,
                                     qos_profile_sensor_data)
        self.get_logger().info(f"Writing frames+pose to {OUT_DIR} every "
                               f"{WRITE_PERIOD}s")

    def on_rgb(self, msg):
        now = time.time()
        self._camera_frame = msg.header.frame_id
        if now - self._last_rgb < WRITE_PERIOD:
            return
        self._last_rgb = now
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        atomic_write(RGB_PATH, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _refresh_extrinsic(self, now):
        """base_footprint -> camera-optical transform from TF (refreshed
        periodically: the torso lift / head joints move the camera)."""
        if not TF_OK or self._camera_frame is None:
            return
        if now - self._extrinsic_stamp < EXTRINSIC_REFRESH:
            return
        try:
            tr = self.tf_buffer.lookup_transform(
                "base_footprint", self._camera_frame, rclpy.time.Time())
            t, q = tr.transform.translation, tr.transform.rotation
            self._extrinsic = {"t": [t.x, t.y, t.z],
                               "q": [q.x, q.y, q.z, q.w],
                               "child_frame": self._camera_frame}
            if self._extrinsic_stamp == 0.0:
                self.get_logger().info(
                    f"camera extrinsic from TF: base_footprint -> "
                    f"{self._camera_frame} t={self._extrinsic['t']}")
            self._extrinsic_stamp = now
        except Exception:
            pass

    def on_odom(self, msg):
        now = time.time()
        if now - self._last_pose < WRITE_PERIOD:
            return
        self._last_pose = now
        self._refresh_extrinsic(now)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        data = {"x": p.x, "y": p.y, "z": p.z, "yaw": yaw,
                "frame": "map", "stamp": now}
        if self._extrinsic:
            data["camera_extrinsic"] = self._extrinsic
        tmp = POSE_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, POSE_PATH)
        except OSError as e:
            _write_failures[0] += 1
            if _write_failures[0] % 25 == 1:
                print("[frame_grabber] pose write failed (%s), continuing" % e,
                      flush=True)

    def on_info(self, msg):
        if self._info_written:
            return
        self._info_written = True
        k = msg.k
        info = {"fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
                "width": msg.width, "height": msg.height}
        tmp = INFO_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(info, f)
            os.replace(tmp, INFO_PATH)
        except OSError as e:
            self._info_written = False
            print("[frame_grabber] camera_info write failed (%s)" % e, flush=True)
            return
        self.get_logger().info(f"Camera intrinsics saved: fx={k[0]:.1f} "
                               f"fy={k[4]:.1f} cx={k[2]:.1f} cy={k[5]:.1f}")

def main():
    rclpy.init()
    node = FrameGrabberTiago()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
