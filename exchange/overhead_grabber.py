#!/usr/bin/env python3
import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

OUT = "/root/exchange/exchange/overhead_frame.jpg"
TOPIC = "/overhead_camera/overhead/image_raw"

_write_failures = [0]

class Grab(Node):
    def __init__(self):
        super().__init__("overhead_grabber")
        self.create_subscription(Image, TOPIC, self.on_img, 10)
        self.n = 0

    def on_img(self, msg):
        try:
            h, w = msg.height, msg.width
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            arr = arr.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
            if msg.encoding == "rgb8":
                arr = arr[:, :, ::-1]
            ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return
            _write_failures[0] = _write_failures[0]
            tmp = OUT + ".tmp"
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, OUT)
        except Exception as e:
            self.get_logger().warn(f"decode failed: {e}")

def main():
    rclpy.init()
    n = Grab()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
