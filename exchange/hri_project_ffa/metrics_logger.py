#!/usr/bin/env python3
"""Motion metrics for the experimental evaluation, logged on the host side.

The behavioural metrics already produced by waiter.py (waiting times, serve
success) come from the event timeline. They say nothing about HOW the robot
moved while producing them, which is exactly what the proxemic part of the
system changes: the approach distance and the approach angle are modulated by
the customer's mood, and the priority reordering changes the route.

This module samples the ground-truth pose the bridge publishes in
shared/robot_pose.json, together with the customer positions in
shared/customers_present.json, and accumulates the four objective quantities
that the social-navigation literature reports for this kind of comparison:

  path_length_m        distance travelled by the base over the whole run
  min_distance_m       closest the base ever came to each customer
  intrusions           number of times the base entered the personal space
                       of a customer (Hall's 1.2 m boundary is commonly
                       rounded to 1.0 m in HRI evaluations)
  duration_s           wall-clock length of the run

An intrusion is counted once per approach, not once per sample: the counter
only re-arms after the robot has moved back beyond RELEASE_RADIUS. Without
that hysteresis a single delivery at a table would be counted dozens of
times, since the sampler runs at 2 Hz and the robot stands still while it
talks.

The file is written continuously, so a run that is interrupted still leaves
usable data behind.
"""
import json
import math
import os
import threading
import time

INTRUSION_RADIUS = 1.0      # m: entering personal space
RELEASE_RADIUS = 1.3        # m: hysteresis, the counter re-arms beyond this
SAMPLE_PERIOD = 0.5         # s
MIN_STEP = 0.01             # m: below this a displacement is pose noise
MAX_STEP = 1.0              # m: above this the pose jumped (respawn/teleport)


class MotionMetrics:
    """Background sampler over the shared pose files."""

    def __init__(self, shared_dir):
        self.shared = shared_dir
        self.pose_file = os.path.join(shared_dir, "robot_pose.json")
        self.customers_file = os.path.join(shared_dir, "customers_present.json")
        self.out_file = os.path.join(shared_dir, "motion_metrics.json")
        self.t0 = time.time()
        self.path_length = 0.0
        self.samples = 0
        self._last_xy = None
        self._min_dist = {}      # entity -> minimum distance seen
        self._inside = {}        # entity -> True while inside personal space
        self._intrusions = {}    # entity -> count
        self._stop = threading.Event()

    # ------------------------------------------------------------------ io
    def _read_pose(self):
        try:
            with open(self.pose_file) as f:
                d = json.load(f)
            return float(d["x"]), float(d["y"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _read_customers(self):
        try:
            with open(self.customers_file) as f:
                present = json.load(f).get("present", [])
        except (OSError, ValueError, TypeError):
            return {}
        out = {}
        for c in present:
            pos = c.get("pos") or []
            if len(pos) >= 2:
                out[str(c.get("entity", c.get("table_id")))] = (float(pos[0]),
                                                                float(pos[1]))
        return out

    # -------------------------------------------------------------- sampling
    def _step(self):
        xy = self._read_pose()
        if xy is None:
            return
        self.samples += 1
        if self._last_xy is not None:
            step = math.dist(xy, self._last_xy)
            # A displacement below MIN_STEP is pose noise while standing still;
            # one above MAX_STEP is a jump (the robot was respawned), and both
            # would inflate the path length if integrated.
            if MIN_STEP <= step <= MAX_STEP:
                self.path_length += step
        self._last_xy = xy

        for entity, cxy in self._read_customers().items():
            d = math.dist(xy, cxy)
            if entity not in self._min_dist or d < self._min_dist[entity]:
                self._min_dist[entity] = d
            if d < INTRUSION_RADIUS and not self._inside.get(entity):
                self._inside[entity] = True
                self._intrusions[entity] = self._intrusions.get(entity, 0) + 1
            elif d > RELEASE_RADIUS:
                self._inside[entity] = False

    # ---------------------------------------------------------------- output
    def snapshot(self):
        dists = list(self._min_dist.values())
        return {
            "duration_s": round(time.time() - self.t0, 1),
            "path_length_m": round(self.path_length, 2),
            "min_distance_m": round(min(dists), 2) if dists else None,
            "mean_min_distance_m": (round(sum(dists) / len(dists), 2)
                                    if dists else None),
            "intrusions": sum(self._intrusions.values()),
            "per_customer": {e: {"min_distance_m": round(self._min_dist[e], 2),
                                 "intrusions": self._intrusions.get(e, 0)}
                             for e in sorted(self._min_dist)},
            "samples": self.samples,
        }

    def save(self):
        tmp = self.out_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.snapshot(), f, indent=2)
            os.replace(tmp, self.out_file)
        except OSError:
            pass

    # ------------------------------------------------------------------ life
    def _loop(self):
        while not self._stop.is_set():
            try:
                self._step()
                self.save()
            except Exception:
                pass
            self._stop.wait(SAMPLE_PERIOD)

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        print("📏 Motion metrics: sampling pose every %.1fs -> %s"
              % (SAMPLE_PERIOD, os.path.basename(self.out_file)))
        return self

    def stop(self):
        self._stop.set()
        self.save()


def start(shared_dir):
    """Convenience entry point used by waiter.py."""
    return MotionMetrics(shared_dir).start()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    m = start(os.path.join(here, "shared"))
    try:
        while True:
            time.sleep(5)
            print(json.dumps(m.snapshot()))
    except KeyboardInterrupt:
        m.stop()
