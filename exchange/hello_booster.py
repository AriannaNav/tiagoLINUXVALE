"""
Hello World example - reads RGB image, depth image, IMU and joint states from the Booster T1 simulation.

Run inside the container:
    python3 /app/exchange/hello_booster.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState


class HelloBooster(Node):
    def __init__(self):
        super().__init__('hello_booster')

        self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_cb, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_cb, 10)
        self.create_subscription(Imu, '/booster/ros2_k2_imu', self.imu_cb, 10)
        self.create_subscription(JointState, '/booster/ros2_k2_joint_states', self.joints_cb, 10)

        self.get_logger().info('HelloBooster node started — waiting for messages...')

    def rgb_cb(self, msg: Image):
        self.get_logger().info(
            f'[RGB]   {msg.width}x{msg.height}  encoding={msg.encoding}'
        )

    def depth_cb(self, msg: Image):
        self.get_logger().info(
            f'[DEPTH] {msg.width}x{msg.height}  encoding={msg.encoding}'
        )

    def imu_cb(self, msg: Imu):
        a = msg.linear_acceleration
        self.get_logger().info(
            f'[IMU]   accel=({a.x:.2f}, {a.y:.2f}, {a.z:.2f})'
        )

    def joints_cb(self, msg: JointState):
        self.get_logger().info(
            f'[JOINTS] {len(msg.position)} joints  — first={msg.position[0]:.3f} rad'
            if msg.position else '[JOINTS] received (no positions)'
        )


def main():
    rclpy.init()
    rclpy.spin(HelloBooster())


main()
