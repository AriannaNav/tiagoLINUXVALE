#!/usr/bin/env python3
"""Grasp-and-serve sequence for the TIAGo waiter.

Tuck, drive to the counter, raise the torso, open the gripper, reach, close, attach
and lift, tuck with the bottle held, drive to the person, extend, release, tuck, go
home. Any step failure tucks the arm and stops the run.

One ArmController node does both the manipulation and the driving, so the attach
timer keeps the bottle on the gripper while the base moves. Navigation goes through
Nav2; the in-place nudges and turns stay closed-loop on ground truth.

Run:  ros2 run waiter_robot grasp_test"""

import math
import os
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

from waiter_robot.arm_controller import ArmController, GRASP_TORSO

W_MIN, W_MAX, W_KP = 0.32, 1.4, 1.7
V_MAX = 0.70
HEAD_REORIENT = math.radians(45.0)
NEAR = 0.18
FACE_GOAL = math.radians(3.0)
POS_GOAL = 0.04
CTRL_DT = 0.05

NAV_SERVER_WAIT = 20.0
NAV_TIMEOUT = 420.0
# How far short of a docking pose Nav2 is asked to stop, leaving the final
# straight-in leg to closed-loop ground-truth driving. See navigate_to().
NAV_PREDOCK = 0.55

GRASP_BASE = (2.0, -2.70, math.radians(-90))
# Default dock/serve = table 1. The bridge overrides both per order from
# waiter.py's TABLES, so these only apply to a command with no table attached —
# but they must still be right. Derived like waiter.py's _table_positions():
# dining_table_1 uses model://bar_table, a ROUND top of radius 0.65 whose centre
# is the include pose (4.0, 2.5) plus the model's own (+0.40, +0.75) offset, so
# (4.4, 3.25), with the top at z 0.72. The old values (3.45, 2.75) / (4.25,
# 2.75) came from the rectangular kitchen_table this model replaced: they parked
# the robot on the table's diagonal, facing ~28 deg past it and ~23 deg past the
# customer, and put the drink on the far rim.
PERSON_BASE = (3.643, 2.493, -0.187)   # SW of the round top, facing the customer
HOME_BASE = (0.0, 0.0, math.radians(0))

SERVE_PLACE = (4.117, 2.967, 0.74)     # on the top, 0.40 of the 0.65 radius toward the robot
TORSO_NAV = 0.15
TORSO_SERVE = 0.30

ABORT_HOLD_FILE = os.environ.get(
    "HRI_ABORT_HOLD_FILE",
    "/root/exchange/exchange/hri_project_ffa/shared/abort_delivery.request")

def _consume_abort_request():
    """True (and deletes the file) if a mid-delivery drop was requested."""
    if os.path.exists(ABORT_HOLD_FILE):
        try:
            os.remove(ABORT_HOLD_FILE)
        except OSError:
            pass
        return True
    return False

def norm_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

class GraspDemo:
    """Wraps an ArmController node and adds closed-loop base navigation."""

    def __init__(self, ac: ArmController):
        self.ac = ac
        self.vel = ac.create_publisher(Twist, "/key_vel", 10)
        self._nav = ActionClient(ac, NavigateToPose, "/navigate_to_pose")

    def _pose(self):
        for _ in range(2):
            rclpy.spin_once(self.ac, timeout_sec=CTRL_DT)
        return self.ac._base

    def _pub(self, vx=0.0, wz=0.0):
        t = Twist(); t.linear.x = vx; t.angular.z = wz
        self.vel.publish(t)

    def _stop(self):
        for _ in range(6):
            self._pub(0.0, 0.0)
            rclpy.spin_once(self.ac, timeout_sec=CTRL_DT)

    def _check_carry_abort(self):
        """Consume a mid-transit drop request, called every control-loop tick.
        
        The per-step checks only fire between steps, so a request made during the
        multi-second walk to the customer would sit unconsumed until the robot had already
        arrived. Only acts while a bottle is actually held."""
        if self.ac._holding and _consume_abort_request():
            self._stop()
            print("[nav] Drop requested mid-transit — the bottle falls right here.")
            self.ac.drop_bottle_here()
            self.ac.tuck_arm()
            raise _StepFailed("the drink slipped and fell")

    def _rotate_to(self, target, tol=math.radians(1.8), tmax=18.0):
        t0 = time.time()
        while time.time() - t0 < tmax:
            self._check_carry_abort()
            p = self._pose()
            err = norm_angle(target - p[2])
            if abs(err) < tol:
                break
            w = W_KP * err
            s = 1.0 if w > 0 else -1.0
            self._pub(0.0, s * min(W_MAX, max(W_MIN, abs(w))))
        self._stop()

    def _drive_to(self, tx, ty, tol=POS_GOAL, tmax=30.0):
        t0 = time.time()
        while time.time() - t0 < tmax:
            self._check_carry_abort()
            x, y, th = self._pose()
            d = math.hypot(tx - x, ty - y)
            if d < tol:
                break
            head_err = norm_angle(math.atan2(ty - y, tx - x) - th)
            if d > NEAR:
                if abs(head_err) > HEAD_REORIENT:
                    s = 1.0 if head_err > 0 else -1.0
                    self._pub(0.0, s * max(W_MIN, min(W_MAX, abs(W_KP * head_err))))
                else:
                    self._pub(min(V_MAX, 0.9 * d + 0.12),
                              max(-1.0, min(1.0, 1.7 * head_err)))
            else:
                if abs(head_err) > math.radians(12) and d > 0.05:
                    s = 1.0 if head_err > 0 else -1.0
                    self._pub(0.0, s * W_MIN)
                else:
                    self._pub(0.10, 0.0)
        self._stop()
        x, y, th = self._pose()
        return math.hypot(tx - x, ty - y)

    def navigate_to(self, tx, ty, tyaw, predock=NAV_PREDOCK):
        """Drive to (tx, ty, tyaw): Nav2 for the journey, ground truth for the dock.

        Nav2 alone cannot deliver this pose. Both docking poses sit INSIDE the
        global costmap's 0.50 m inflation — the counter grasp base is 0.31 m
        from the counter edge (the distance the arm sequence is tuned for) and a
        table dock is 0.42 m from the round rim — so the planner rejects them
        outright: "GridBased: failed to create plan with tolerance 0.50 ...
        failed to generate a valid path to (3.64, 2.49)", seen for the counter
        at (-1.80, -2.70) too. Widening the docks to clear the inflation would
        put the drink out of the arm's reach; shrinking the inflation below the
        0.28 m robot radius would let long paths clip the furniture.

        So Nav2 is given a PRE-DOCK point `predock` metres short of the target
        along the final heading — out in open space, where planning is easy and
        obstacle avoidance actually matters — and the last straight leg runs
        closed-loop on ground truth, driving in along the heading the arm
        expects. That final stretch is short, aimed at the furniture the robot
        is about to work on, and needs no planning.

        Returns the final distance from the goal in metres, or None if Nav2
        could not execute the approach at all."""
        px = tx - predock * math.cos(tyaw)
        py = ty - predock * math.sin(tyaw)
        self._back_out_of_dock(predock)
        if self._nav_goal(px, py, tyaw) is None:
            return None
        self._rotate_to(tyaw, tol=FACE_GOAL)
        self._drive_to(tx, ty, tol=POS_GOAL)
        self._rotate_to(tyaw)
        x, y, _ = self._pose()
        return math.hypot(tx - x, ty - y)

    # Furniture the robot docks against, from waiter_scene.world. The four
    # dining tables are round (model://bar_table, r 0.65, centre = the include
    # pose plus the model's own (0.40, 0.75) offset). The two counter tops are
    # long slabs, so they stay rectangles: an enclosing circle on a 5.2 m
    # counter would have a 2.6 m radius and report "docked" from halfway across
    # the room. Only used to answer "am I parked against something right now?".
    _DOCK_CIRCLES = [(4.4, 3.25, 0.65), (-1.6, 3.25, 0.65),
                     (-1.6, 7.75, 0.65), (4.4, 7.75, 0.65)]
    _DOCK_RECTS = [(-0.6, 4.6, -3.85, -3.01),    # bar_counter_top
                   (-2.6, -1.0, -3.85, -3.01)]   # side_counter_top

    def _dock_clearance(self):
        """Distance from the base centre to the nearest docking surface."""
        x, y, _ = self._pose()
        d = [math.hypot(cx - x, cy - y) - r for cx, cy, r in self._DOCK_CIRCLES]
        for x0, x1, y0, y1 in self._DOCK_RECTS:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            d.append(math.hypot(dx, dy))
        return min(d)

    def _at_a_dock(self, margin=0.60):
        """True if the base is parked closer than `margin` to a table rim.

        Nav2 will not plan out of a pose its costmap calls occupied, and both
        docking poses are inside the global 0.50 m inflation by design (see
        navigate_to). So after a delivery the robot sits somewhere the planner
        refuses to start from: seen live as a goal accepted, then 420 s of the
        robot inching 0.3 m before the timeout cancelled it, while it stood
        0.27 m off table 1's rim."""
        return self._dock_clearance() < margin

    def _back_out_of_dock(self, distance):
        """Reverse straight out of a dock so Nav2 starts from free space.

        Straight back along the current heading, no rotation: it retraces the
        line the robot drove in on, which is known clear because it just came
        along it. Docking is two-stage, so undocking has to be too."""
        if not self._at_a_dock():
            return
        print(f"[nav] parcheggiato contro un mobile — arretro {distance:.2f} m "
              "prima di passare a Nav2")
        self.nudge(-distance)

    def _nav_goal(self, tx, ty, tyaw):
        """Send one Nav2 goal and wait for it. Distance from it, or None."""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.ac.get_clock().now().to_msg()
        goal.pose.position.x = float(tx)
        goal.pose.position.y = float(ty)
        goal.pose.orientation.z = math.sin(tyaw / 2.0)
        goal.pose.orientation.w = math.cos(tyaw / 2.0)

        # wait_for_server() da solo non aggiorna il grafo se il nodo non e'
        # spinnato da un executor: qui spinniamo ac mentre aspettiamo, cosi' la
        # discovery del server /navigate_to_pose si completa (senza questo il
        # bridge non "vedeva" Nav2 anche se attivo -> goal mai inviati).
        _srv_deadline = time.time() + NAV_SERVER_WAIT
        while not self._nav.server_is_ready() and time.time() < _srv_deadline:
            rclpy.spin_once(self.ac, timeout_sec=0.1)
        if not self._nav.server_is_ready():
            print("[nav] Nav2 non risponde — il goal non parte")
            return None

        send = self._nav.send_goal_async(NavigateToPose.Goal(pose=goal))
        if not self._spin_until(send, 15.0):
            print("[nav] Nav2 non ha accettato il goal in tempo")
            return None
        handle = send.result()
        if handle is None or not handle.accepted:
            print("[nav] Nav2 ha rifiutato il goal")
            return None

        result = handle.get_result_async()
        deadline = time.time() + NAV_TIMEOUT
        while not result.done() and time.time() < deadline:
            rclpy.spin_once(self.ac, timeout_sec=CTRL_DT)
            self._check_carry_abort()
        if not result.done():
            handle.cancel_goal_async()
            print(f"[nav] goal non concluso entro {NAV_TIMEOUT:.0f}s, annullato")
            return None

        self._stop()
        x, y, _ = self._pose()
        err = math.hypot(tx - x, ty - y)
        status = result.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            print(f"[nav] Nav2 ha terminato con stato {status} "
                  f"(a {err*100:.0f} cm dal punto)")
        return err

    def _spin_until(self, future, timeout):
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(self.ac, timeout_sec=CTRL_DT)
        return future.done()

    def nudge(self, distance, tol=0.02, tmax=6.0):
        """Step forward or backward by `distance` metres along the current heading, with no
        rotation, for small proxemics adjustments.
        
        navigate_to() is the wrong tool for a few centimetres: it is a goal-directed
        approach, and on a tiny target the robot ends up turning around and back instead of
        taking one small step."""
        x0, y0, yaw0 = self._pose()
        tx = x0 + distance * math.cos(yaw0)
        ty = y0 + distance * math.sin(yaw0)
        sign = 1.0 if distance >= 0 else -1.0
        t0 = time.time()
        while time.time() - t0 < tmax:
            x, y, th = self._pose()
            d = math.hypot(tx - x, ty - y)
            if d < tol:
                break
            self._pub(sign * min(V_MAX, 0.6 * d + 0.08), 0.0)
        self._stop()

    def angle_by(self, delta_yaw, tol=math.radians(3.0), tmax=6.0):
        """Rotate IN PLACE by a small RELATIVE angle (radians) — a deliberate,
        bounded turn (e.g. angling slightly away from a distressed customer
        instead of facing them dead-on, HRI4 proxemics), not the accidental
        near-180° spin navigate_to() caused for tiny proxemics moves before
        nudge() existed. A few tens of degrees, quick and intentional."""
        x, y, yaw0 = self._pose()
        self._rotate_to(norm_angle(yaw0 + delta_yaw), tol=tol, tmax=tmax)

class _StepFailed(Exception):
    """Raised internally when a step fails, so the sequence can stop cleanly."""

def run_grasp_and_serve(ac, demo):
    """Execute the full sequence. True on success. The caller owns the ac/demo
    lifecycle."""
    def fail(step, msg):
        print(f"[STEP {step}] FAILED: {msg}")
        print("  tucking arm for safety...")
        ac.detach_bottle()
        ac.tuck_arm()
        raise _StepFailed(msg)

    try:
        print("[STEP 1] Tucking arm for safe navigation...")
        if not ac.tuck_arm():
            fail(1, "tuck_arm")

        print("[STEP 2] Navigating to table grasping position...")
        e = demo.navigate_to(*GRASP_BASE)
        print(f"         at grasp pose (pos err {e*100:.1f} cm)")

        print("[STEP 3] Raising torso...")
        ac.move_torso(GRASP_TORSO)

        print("[STEP 4] Opening gripper...")
        if not ac.open_gripper():
            fail(4, "open_gripper")

        print("[STEP 5] Reaching to the bottle...")
        if not ac.grasp_reach():
            fail(5, "grasp_reach")

        print("[STEP 6] Settling on grasp...")
        ac._spin(1.0)
        res = ac.gripper_to_bottle_error()
        if res:
            print(f"         gripper {res[0]*100:.1f} cm from bottle")

        print("[STEP 7] Closing gripper...")
        if not ac.close_gripper():
            fail(7, "close_gripper")

        print("[STEP 8] Attaching + lifting bottle...")
        ac.attach_bottle()
        ac._spin(0.5)
        ac.lift(0.12)

        def _dropped_mid_carry(step):
            print(f"[STEP {step}] Drop requested — the bottle falls right here, "
                  "delivery aborted.")
            ac.drop_bottle_here()
            ac.tuck_arm()
            raise _StepFailed("the drink slipped and fell")

        print("[STEP 9] Tucking arm with bottle in hand...")
        if _consume_abort_request():
            _dropped_mid_carry(9)
        if not ac.tuck_arm():
            fail(9, "tuck_arm")

        print("[STEP 10] Navigating to the person...")
        if _consume_abort_request():
            _dropped_mid_carry(10)
        e = demo.navigate_to(*PERSON_BASE)
        print(f"          at person (pos err {e*100:.1f} cm)")

        print("[STEP 11] Extending arm to serving position...")
        if _consume_abort_request():
            _dropped_mid_carry(11)
        ac.move_torso(TORSO_SERVE)
        if not ac.offer():
            fail(11, "offer")

        if _consume_abort_request():
            _dropped_mid_carry(12)
        print("[STEP 12] Releasing bottle on the table...")
        ac.detach_bottle(place_pose=SERVE_PLACE)
        ac.open_gripper()

        print("[STEP 13] Tucking arm (staying at the customer's table)...")
        if not ac.tuck_arm():
            fail(13, "tuck_arm")

        print("[COMPLETE] Grasp-and-serve sequence finished (at the customer).")
        return True
    except _StepFailed:
        return False
    except KeyboardInterrupt:
        demo._stop()
        return False

def main(args=None):
    rclpy.init(args=args)
    ac = ArmController()
    demo = GraspDemo(ac)

    t0 = time.time()
    while ac._base is None and time.time() - t0 < 10:
        rclpy.spin_once(ac, timeout_sec=0.1)
    if ac._base is None:
        print("[FATAL] no /ground_truth_odom — is the sim running?")
        ac.destroy_node()
        rclpy.shutdown()
        return

    try:
        ok = run_grasp_and_serve(ac, demo)
    finally:
        ac.detach_bottle()
        ac.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
