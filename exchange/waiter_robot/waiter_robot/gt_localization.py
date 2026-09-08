#!/usr/bin/env python3
"""
gt_localization.py — perfect map->odom localization from Gazebo ground truth.

WHY THIS EXISTS
---------------
In this simulation (running without a GPU, software-rendered Gazebo classic) the
pmb2 base veers noticeably when driving, which violates AMCL's differential
motion model and makes AMCL localization diverge — the robot then believes it
is already at the goal and spins forever. To meet the project goal (reach exact
(x, y, yaw) targets with NO drift and NO accumulating error, reliably), this
node provides "perfect localization": it broadcasts the map->odom transform
computed from the zero-drift ground-truth pose, so the full Nav2 stack
(global/local costmaps, planner, controller) operates in a correct map frame.

It replaces AMCL's TF broadcast. AMCL should be launched with tf_broadcast:=false
(see nav2_params.yaml) so the two do not fight over map->odom.

MATH
----
  map_T_base  : robot pose in map frame  == ground-truth pose
                (the map was built with the robot starting at the world origin,
                 so the Nav2 'map' frame coincides with the Gazebo world frame)
  odom_T_base : wheel-odometry pose, read from the existing TF tree
  map_T_odom  = map_T_base * inverse(odom_T_base)   <-- what we broadcast
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformBroadcaster, TransformListener


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GtLocalization(Node):

    def __init__(self):
        super().__init__("gt_localization")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("ground_truth_topic", "/ground_truth_odom")
        self.declare_parameter("rate", 30.0)

        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        topic = self.get_parameter("ground_truth_topic").value
        rate = self.get_parameter("rate").value

        # Publish map->odom slightly into the future so consumers never see it
        # as stale relative to the high-rate odom->base transform (the same
        # trick nav2's fake_localization / amcl transform_tolerance uses).
        self.declare_parameter("transform_tolerance", 0.1)
        self._tol = float(self.get_parameter("transform_tolerance").value)

        self._gt = None     # (x, y, yaw) of base in map frame
        self._gt_stamp = None  # header stamp of the latest ground-truth msg
        self.create_subscription(Odometry, topic, self._on_gt, 20)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._br = TransformBroadcaster(self)

        self.get_logger().info(
            f"gt_localization: broadcasting {self.map_frame}->{self.odom_frame} "
            f"from {topic} (one transform per ground-truth message)")

    def _on_gt(self, msg: Odometry):
        p = msg.pose.pose
        self._gt = (p.position.x, p.position.y, quat_to_yaw(p.orientation))
        self._gt_stamp = msg.header.stamp
        self._broadcast()

    def _broadcast(self):
        if self._gt is None or self._gt_stamp is None:
            return
        # odom_T_base from the TF tree (wheel odometry).
        try:
            tf = self._tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return
        ox = tf.transform.translation.x
        oy = tf.transform.translation.y
        oyaw = quat_to_yaw(tf.transform.rotation)

        gx, gy, gyaw = self._gt

        # map_T_odom = map_T_base * inv(odom_T_base)
        yaw_mo = gyaw - oyaw
        c, s = math.cos(yaw_mo), math.sin(yaw_mo)
        tx = gx - (c * ox - s * oy)
        ty = gy - (s * ox + c * oy)

        t = TransformStamped()
        # Stamp = ground-truth (sim) time + transform_tolerance, so the
        # transform is valid slightly ahead of the latest odom->base.
        total_ns = (self._gt_stamp.sec * 1_000_000_000 + self._gt_stamp.nanosec
                    + int(self._tol * 1e9))
        t.header.stamp.sec = total_ns // 1_000_000_000
        t.header.stamp.nanosec = total_ns % 1_000_000_000
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(yaw_mo / 2.0)
        t.transform.rotation.w = math.cos(yaw_mo / 2.0)
        self._br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = GtLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
