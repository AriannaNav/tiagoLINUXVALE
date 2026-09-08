#!/usr/bin/env python3
"""TIAGo arm, torso and gripper control, plus a deterministic Gazebo attach.

pymoveit2 is not installed in this image and the simulation is CPU-bound, so
instead of Cartesian IK this uses play_motion2 for named poses and direct
FollowJointTrajectory for the torso, the gripper and a tuned grasp-reach arm
configuration.

Grasping a free object with the pal-gripper is unreliable in Gazebo classic, so a
held bottle is teleported to follow the gripper frame with set_entity_state:
deterministic, and it survives navigation."""

import math
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

try:
    from play_motion2_msgs.action import PlayMotion2
    HAVE_PLAY_MOTION = True
except ImportError:
    HAVE_PLAY_MOTION = False

ARM_JOINTS = [f"arm_{i}_joint" for i in range(1, 8)]
GRIPPER_JOINTS = ["gripper_left_finger_joint", "gripper_right_finger_joint"]

HOME_ARM = [0.50, -1.34, -0.48, 1.94, -1.49, 1.37, 0.0]

GRASP_TORSO = 0.32
GRASP_ARM = [0.27, -0.788, -0.741, 2.286, 0.517, 1.306, 1.496]

GRIPPER_OPEN = 0.044
GRIPPER_CLOSED = 0.012

BOTTLE_NAME = "sprite_bottle"

def yaw_to_quat(yaw):
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q

def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

class ArmController(Node):

    def __init__(self):
        super().__init__(
            "arm_controller",
            parameter_overrides=[Parameter("use_sim_time", value=True)])

        self._arm = ActionClient(self, FollowJointTrajectory,
                                 "/arm_controller/follow_joint_trajectory")
        self._torso = ActionClient(self, FollowJointTrajectory,
                                   "/torso_controller/follow_joint_trajectory")
        self._gripper = ActionClient(self, FollowJointTrajectory,
                                     "/gripper_controller/follow_joint_trajectory")
        self._head = ActionClient(self, FollowJointTrajectory,
                                  "/head_controller/follow_joint_trajectory")
        self._pm = ActionClient(self, PlayMotion2, "/play_motion2") \
            if HAVE_PLAY_MOTION else None

        self._base = None
        self.create_subscription(Odometry, "/ground_truth_odom",
                                 self._on_odom, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._set_state = self.create_client(SetEntityState, "/set_entity_state")
        self._get_state = self.create_client(GetEntityState, "/get_entity_state")

        self._holding = False
        self._attach_timer = None
        self._joint_pos = {}
        from sensor_msgs.msg import JointState
        self.create_subscription(JointState, "/joint_states",
                                 self._on_joints, 10)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        self._base = (p.position.x, p.position.y, quat_to_yaw(p.orientation))

    def _on_joints(self, msg):
        for n, p in zip(msg.name, msg.position):
            self._joint_pos[n] = p

    def _spin(self, secs):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_future(self, fut, timeout):
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def _send_traj(self, client, joints, positions, seconds, timeout=None):
        if not client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(f"trajectory server not available")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start = Duration(sec=int(seconds),
                                      nanosec=int((seconds % 1) * 1e9))
        goal.trajectory.points = [pt]
        send = client.send_goal_async(goal)
        gh = self._wait_future(send, 15.0)
        if gh is None or not gh.accepted:
            self.get_logger().error("trajectory goal rejected")
            return False
        res = self._wait_future(gh.get_result_async(),
                                (timeout or max(40.0, seconds * 8.0)))
        if res is None:
            self.get_logger().error("trajectory result timed out")
        return res is not None

    def play_motion(self, name, skip_planning=True, timeout=90.0):
        if self._pm is None:
            self.get_logger().error("play_motion2 not available")
            return False
        if not self._pm.wait_for_server(timeout_sec=15.0):
            self.get_logger().error("play_motion2 server not available")
            return False
        goal = PlayMotion2.Goal()
        goal.motion_name = name
        goal.skip_planning = skip_planning
        send = self._pm.send_goal_async(goal)
        gh = self._wait_future(send, 15.0)
        if gh is None or not gh.accepted:
            self.get_logger().error(f"play_motion '{name}' rejected")
            return False
        res = self._wait_future(gh.get_result_async(), timeout)
        ok = res is not None and res.result.success
        if not ok:
            self.get_logger().error(f"play_motion '{name}' failed")
        return ok

    def tuck_arm(self):
        """Safe navigation pose. Uses play_motion2 'home'."""
        return self.play_motion("home", skip_planning=True)

    def offer(self):
        """Extend arm to the serving pose. Uses play_motion2 'offer'."""
        return self.play_motion("offer", skip_planning=True)

    def greet(self):
        """Wave hello (#10 — a gesture DISTINCT from offer(), which is reused
        for both serving and pointing). Uses play_motion2 'wave' — an EXISTING
        TIAGo demo motion (tiago_bringup/config/motions/tiago_motions_general.yaml),
        not a newly invented one, so it's known to actually run on this robot."""
        return self.play_motion("wave", skip_planning=True)

    def move_torso(self, height, seconds=3.0):
        height = max(0.0, min(0.35, float(height)))
        return self._send_traj(self._torso, ["torso_lift_joint"], [height], seconds)

    def look_at_customer(self, pan=0.0, tilt=0.05, seconds=1.2):
        """Turn the head to face the customer TIAGo just arrived at (level,
        centred) instead of leaving it wherever the last motion left it."""
        pan = max(-1.5, min(1.5, float(pan)))
        tilt = max(-1.0, min(0.4, float(tilt)))
        return self._send_traj(self._head, ["head_1_joint", "head_2_joint"],
                               [pan, tilt], seconds)

    def open_gripper(self, seconds=1.5):
        return self._send_traj(self._gripper, GRIPPER_JOINTS,
                               [GRIPPER_OPEN, GRIPPER_OPEN], seconds)

    def close_gripper(self, seconds=1.5):
        return self._send_traj(self._gripper, GRIPPER_JOINTS,
                               [GRIPPER_CLOSED, GRIPPER_CLOSED], seconds)

    def move_arm(self, positions, seconds=4.0):
        return self._send_traj(self._arm, ARM_JOINTS, positions, seconds)

    def grasp_reach(self):
        """Torso up + arm forward/down to place the gripper at the bottle."""
        ok = self.move_torso(GRASP_TORSO, seconds=3.0)
        ok = self.move_arm(GRASP_ARM, seconds=4.0) and ok
        self._spin(1.0)
        return ok

    def lift(self, height=0.15, seconds=2.5):
        """Raise the held object by raising the torso."""
        cur = self._joint_pos.get("torso_lift_joint", GRASP_TORSO)
        return self.move_torso(min(0.35, cur + height), seconds=seconds)

    def is_arm_tucked(self, tol=0.30):
        self._spin(0.3)
        for name, target in zip(ARM_JOINTS, HOME_ARM):
            if name not in self._joint_pos:
                return False
            if abs(self._joint_pos[name] - target) > tol:
                return False
        return True

    def _gripper_world_pose(self):
        """World (x, y, z) of gripper_grasping_frame from ground truth + TF."""
        if self._base is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                "base_footprint", "gripper_grasping_frame", rclpy.time.Time())
        except Exception:
            return None
        gx = tf.transform.translation.x
        gy = tf.transform.translation.y
        gz = tf.transform.translation.z
        bx, by, byaw = self._base
        c, s = math.cos(byaw), math.sin(byaw)
        return (bx + c * gx - s * gy, by + s * gx + c * gy, gz)

    def attach_bottle(self):
        """Start rigidly teleporting the bottle to follow the gripper."""
        self._holding = True
        if self._attach_timer is None:
            self._attach_timer = self.create_timer(0.05, self._attach_tick)
        self.get_logger().info("bottle attached to gripper")

    def detach_bottle(self, place_pose=None):
        """Stop following; optionally drop the bottle at place_pose (x,y,z)."""
        self._holding = False
        if place_pose is not None:
            self._teleport_bottle(*place_pose)
        self.get_logger().info("bottle released")

    def drop_bottle_here(self):
        """Stop following the gripper and let the held bottle fall at floor
        level, right at the gripper's CURRENT (x, y) — used when a delivery
        is aborted mid-transit (demo "drop" trigger), as opposed to
        detach_bottle(place_pose=...) which places it deliberately on a
        table."""
        self._holding = False
        p = self._gripper_world_pose()
        if p is not None:
            self._teleport_bottle(p[0], p[1], 0.05)
        self.get_logger().info("bottle dropped mid-delivery")

    def _attach_tick(self):
        if not self._holding:
            return
        p = self._gripper_world_pose()
        if p is not None:
            self._teleport_bottle(*p)

    def _teleport_bottle(self, x, y, z):
        if not self._set_state.wait_for_service(timeout_sec=0.2):
            return
        req = SetEntityState.Request()
        es = EntityState()
        es.name = BOTTLE_NAME
        es.pose.position.x = float(x)
        es.pose.position.y = float(y)
        es.pose.position.z = float(z)
        es.pose.orientation.w = 1.0
        es.reference_frame = "world"
        req.state = es
        self._set_state.call_async(req)

    def gripper_to_bottle_error(self):
        """Debug helper: distance from gripper frame to the bottle (m)."""
        gp = self._gripper_world_pose()
        if gp is None or not self._get_state.wait_for_service(timeout_sec=1.0):
            return None
        req = GetEntityState.Request()
        req.name = BOTTLE_NAME
        req.reference_frame = "world"
        res = self._wait_future(self._get_state.call_async(req), 2.0)
        if res is None or not res.success:
            return None
        b = res.state.pose.position
        return math.dist(gp, (b.x, b.y, b.z)), gp, (b.x, b.y, b.z)

def main(args=None):
    """Standalone smoke test: tuck, then report tucked state."""
    rclpy.init(args=args)
    ac = ArmController()
    ac.get_logger().info("Tucking arm...")
    ok = ac.tuck_arm()
    ac.get_logger().info(f"tuck ok={ok}  is_arm_tucked={ac.is_arm_tucked()}")
    ac.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
