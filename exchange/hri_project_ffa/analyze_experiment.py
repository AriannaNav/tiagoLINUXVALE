#!/usr/bin/env python3
"""Experimental evaluation of the two manipulated conditions.

Two independent variables can be toggled from the environment, and the same
script analyses either comparison:

  WAITER_PRIORITY_REORDER  urgency-driven reordering of the service queue.
      Expected to shorten the wait of the customer who is in a hurry, at the
      cost of a longer wait for the others.

  WAITER_PROXEMICS         mood-driven adjustment of the approach geometry,
      that is the stopping distance and the angle of approach. Expected to
      affect the closest approach distance, the number of personal-space
      intrusions and the perceived safety, while leaving the waiting times
      essentially unchanged.

Only one variable should be toggled at a time; the configuration actually
used is written into the event log at start-up and reported back here.

Aggregates per-run metrics from shared/events.json and runs the statistical test
comparing the two conditions, plus the Godspeed and RoSAS questionnaire scores if
any were collected.

Procedure:
  1. run the scenario N>=6 times per condition (six is the minimum at which a
     two-sided Wilcoxon signed-rank test can reach p<0.05: with five pairs the
     smallest attainable p-value is 0.0625) (control: WAITER_PRIORITY_REORDER=0),
     copying shared/events.json to runs/<condition>_runN.json and
     shared/motion_metrics.json to runs/<condition>_runN.motion.json after
     each run, where <condition> is e.g. reorder_on / reorder_off or
     proxemics_on / proxemics_off;
  2. fill godspeed_questionnaire.html once per run, appending each result line to
     runs/godspeed_results.jsonl;
  3. python3 analyze_experiment.py "runs/reorder_on_*.json" --vs "runs/reorder_off_*.json"

H1: reordering lowers the urgent customer's wait time and raises Likeability and
Perceived Intelligence."""
import argparse
import glob
import json
import re
import statistics

try:
    from scipy import stats
except ImportError:
    stats = None

CONDITIONS_RE = re.compile(
    r"conditions: priority_reorder=(on|off), proxemics=(on|off)")
ORDER_RE = re.compile(r"serving order: (.+?) \(priority_reorder=(on|off)\)")
ORDER_TABLE_RE = re.compile(r"([\w ]+?) \((high|medium|low)\)")
ORDER_TAKEN_RE = re.compile(r"^order at (.+?): ")
SERVE_START_RE = re.compile(r"^Now serving (.+?) to (.+?)\.$")
SERVE_DONE_RE = re.compile(r"^(.+?) → (.+?): (served|not served)$")

def cohens_d(a, b, paired):
    """Standardised effect size. Paired samples use the standard deviation of
    the differences (Cohen's d_z), independent samples the pooled standard
    deviation. Reported alongside the p-value because, with the small samples
    typical of a course project, significance alone says little about how
    large the difference actually is."""
    if paired:
        diffs = [x - y for x, y in zip(a, b)]
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        return (statistics.mean(diffs) / sd) if sd else float("nan")
    na, nb = len(a), len(b)
    va = statistics.variance(a) if na > 1 else 0.0
    vb = statistics.variance(b) if nb > 1 else 0.0
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5 \
        if na + nb > 2 else 0.0
    return ((statistics.mean(a) - statistics.mean(b)) / pooled) if pooled \
        else float("nan")

def magnitude(d):
    """Conventional reading of |d| (Cohen, 1988)."""
    d = abs(d)
    if d != d:
        return ""
    return ("negligible" if d < 0.2 else "small" if d < 0.5 else
            "medium" if d < 0.8 else "large")

def load_events(path):
    with open(path) as f:
        return json.load(f).get("events", [])

def extract_metrics(events):
    """One run's objective metrics: urgent-table wait time (order-taken ->
    served), mean wait time across all tables, serve success rate, and
    whether priority reorder was active this run."""
    order_taken, served_at = {}, {}
    urgent_table, reorder_on = None, None

    proxemics_on = None
    for e in events:
        text, ts = e.get("text", ""), e.get("ts")
        m = CONDITIONS_RE.search(text)
        if m:
            reorder_on = (m.group(1) == "on")
            proxemics_on = (m.group(2) == "on")
            continue
        m = ORDER_RE.search(text)
        if m:
            reorder_on = (m.group(2) == "on")
            for name, urgency in ORDER_TABLE_RE.findall(m.group(1)):
                if urgency == "high":
                    urgent_table = name.strip()
            continue
        m = ORDER_TAKEN_RE.search(text)
        if m:
            order_taken.setdefault(m.group(1).strip(), ts)
            continue
        m = SERVE_DONE_RE.match(text)
        if m and e.get("icon") in ("✅", "⚠️") and m.group(3) == "served":
            served_at[m.group(2).strip()] = ts

    wait_times = {t: served_at[t] - t0 for t, t0 in order_taken.items()
                 if t in served_at and served_at[t] >= t0}
    n_serve_events = sum(1 for e in events if e.get("icon") in ("✅", "⚠️"))
    n_served_ok = sum(1 for e in events if e.get("icon") == "✅")

    stamps = [e.get("ts") for e in events if e.get("ts")]
    duration = (max(stamps) - min(stamps)) if len(stamps) > 1 else None

    return {
        "reorder_on": reorder_on,
        "proxemics_on": proxemics_on,
        "urgent_table": urgent_table,
        "urgent_wait": wait_times.get(urgent_table) if urgent_table else None,
        "mean_wait": statistics.mean(wait_times.values()) if wait_times else None,
        "serve_success_rate": (n_served_ok / n_serve_events) if n_serve_events else None,
        "duration": duration,
    }

def load_motion(events_path):
    """Motion metrics recorded by metrics_logger.py for the same run.

    Each run leaves two files: the event timeline and a sibling
    '<name>.motion.json' holding path length, closest approach and
    personal-space intrusions. A run copied before the logger existed simply
    has no sibling, and the corresponding metrics are reported as missing
    rather than treated as zero."""
    base = events_path[:-5] if events_path.endswith(".json") else events_path
    for candidate in (base + ".motion.json", events_path + ".motion"):
        try:
            with open(candidate) as f:
                return json.load(f)
        except (OSError, ValueError):
            continue
    return {}

def load_questionnaire(path):
    """*_results.jsonl (same {participant,condition,scores,ts} schema for both
    godspeed_questionnaire.html and rosas_questionnaire.html) ->
    {condition: [{subscale: score}, ...]}"""
    by_condition = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_condition.setdefault(rec["condition"], []).append(rec["scores"])
    except OSError:
        pass
    return by_condition

def run_test(a, b, label, ordinal=False):
    """Compare two conditions and report significance and effect size.

    A paired test is used when the two samples have the same size, since run i
    of condition A and run i of condition B are the same scenario before and
    after the change (within-subject). Otherwise Welch's test is used, which
    does not assume equal variances and is the safer default for small and
    uneven samples.

    `ordinal` selects a rank-based test. Questionnaire subscales are averages
    of Likert items, so the distance between two levels is not guaranteed to
    be constant and a rank-based test is the appropriate choice."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        print(f"  {label}: not enough data (n_a={len(a)}, n_b={len(b)}) "
              "— need >=2 runs per condition")
        return
    if stats is None:
        print(f"  {label}: scipy not installed (`pip install scipy`) — "
              f"raw means: A={statistics.mean(a):.2f}, B={statistics.mean(b):.2f}")
        return
    paired = len(a) == len(b)
    if ordinal and paired:
        stat, p = stats.wilcoxon(a, b)
        test_name = "Wilcoxon signed-rank test (paired, ordinal data)"
    elif ordinal:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "Mann-Whitney U test (independent, ordinal data)"
    elif paired:
        stat, p = stats.ttest_rel(a, b)
        test_name = "paired-samples t-test (within-subject)"
    else:
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        test_name = "Welch's independent-samples t-test (unequal n)"
    d = cohens_d(a, b, paired)
    if p != p:      # NaN: the two conditions produced identical values
        print(f"  {label}: identical in both conditions "
              f"(A=B={statistics.mean(a):.2f}, n={len(a)}) — no test applicable")
        return
    sig = "SIGNIFICANT at alpha=0.05" if p < 0.05 else "not significant at alpha=0.05"
    print(f"  {label}: A={statistics.mean(a):.2f}+-{_sd(a):.2f} (n={len(a)}), "
          f"B={statistics.mean(b):.2f}+-{_sd(b):.2f} (n={len(b)})")
    print(f"    {test_name}: stat={stat:.3f}, p={p:.4f} -> {sig}")
    if d == d:
        print(f"    Cohen's d = {d:.2f} ({magnitude(d)} effect)")
    else:
        print("    Cohen's d: not defined (no variance in the samples)")

def _sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0

def report_questionnaire(path, label):
    """Load one *_results.jsonl file and run_test() every subscale it
    contains, comparing the two conditions found inside it. Shared by
    Godspeed and RoSAS (and any future validated questionnaire dropped in
    the same {participant,condition,scores,ts} format)."""
    data = load_questionnaire(path)
    if not data:
        print(f"\n(No {label} data found at {path} — fill in the "
              f"questionnaire after each run and append the printed JSON "
              f"line there.)")
        return
    print()
    print(f"Subjective metrics ({label} questionnaire)")
    print("-" * 60)
    conditions = list(data.keys())
    if len(conditions) != 2:
        print(f"  need exactly 2 conditions in {path}, found: {conditions}")
        return
    ca, cb = conditions
    subscales = sorted({k for recs in data.values() for r in recs for k in r})
    for sub in subscales:
        run_test([r.get(sub) for r in data[ca]], [r.get(sub) for r in data[cb]],
                 sub, ordinal=True)

def describe_condition(name, metrics):
    """Report which experimental condition the runs of a group were recorded
    under, reading the configuration line the waiter writes at start-up.

    A group whose runs do not share the same configuration cannot be compared
    against anything, and the mistake is silent otherwise: the numbers still
    average, they simply average two different systems."""
    for flag, label in (("reorder_on", "priority_reorder"),
                        ("proxemics_on", "proxemics")):
        values = {m.get(flag) for m in metrics}
        values.discard(None)
        if not values:
            state = "not recorded (runs predate the configuration log)"
        elif len(values) == 1:
            state = "on" if values.pop() else "off"
        else:
            state = "MIXED — the runs in this group are not one condition"
        print(f"    {name}: {label} = {state}")

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("condition_a", nargs="+",
                    help="events.json snapshots for condition A (glob ok)")
    ap.add_argument("--vs", nargs="+", required=True,
                    help="events.json snapshots for condition B")
    ap.add_argument("--godspeed", default="runs/godspeed_results.jsonl")
    ap.add_argument("--rosas", default="runs/rosas_results.jsonl",
                    help="optional secondary questionnaire (RoSAS)")
    args = ap.parse_args()

    # The motion snapshots sit next to the event snapshots and share their
    # stem, so a glob such as "runs/reorder_on_*.json" matches both. They are
    # loaded separately, by load_motion(), and must not be counted as runs.
    def expand(patterns):
        found = sorted({f for pat in patterns for f in glob.glob(pat)})
        found = [f for f in found if not f.endswith(".motion.json")]
        return found or list(patterns)

    files_a = expand(args.condition_a)
    files_b = expand(args.vs)

    metrics_a = [extract_metrics(load_events(f)) for f in files_a]
    metrics_b = [extract_metrics(load_events(f)) for f in files_b]

    print(f"Condition A: {len(files_a)} run(s) — {files_a}")
    print(f"Condition B: {len(files_b)} run(s) — {files_b}")
    describe_condition("A", metrics_a)
    describe_condition("B", metrics_b)
    print()
    print("Objective metrics (from events.json)")
    print("-" * 60)
    run_test([m["urgent_wait"] for m in metrics_a],
             [m["urgent_wait"] for m in metrics_b],
             "Time-to-service for the URGENT customer (s)")
    run_test([m["mean_wait"] for m in metrics_a],
             [m["mean_wait"] for m in metrics_b],
             "Mean time-to-service, all tables (s)")
    run_test([m["serve_success_rate"] for m in metrics_a],
             [m["serve_success_rate"] for m in metrics_b],
             "Serve success rate")
    run_test([m["duration"] for m in metrics_a],
             [m["duration"] for m in metrics_b],
             "Task completion time, whole round (s)")

    motion_a = [load_motion(f) for f in files_a]
    motion_b = [load_motion(f) for f in files_b]
    if any(motion_a) or any(motion_b):
        print()
        print("Motion metrics (from metrics_logger.py)")
        print("-" * 60)
        for key, label in (
                ("path_length_m", "Path length (m)"),
                ("min_distance_m", "Minimum robot-customer distance (m)"),
                ("mean_min_distance_m",
                 "Mean closest approach over customers (m)"),
                ("intrusions", "Personal-space intrusions (< 1.0 m)")):
            run_test([m.get(key) for m in motion_a],
                     [m.get(key) for m in motion_b], label)
    else:
        print()
        print("(No motion metrics found. Copy shared/motion_metrics.json next "
              "to each events snapshot, renamed '<run>.motion.json'.)")

    report_questionnaire(args.godspeed, "Godspeed")
    report_questionnaire(args.rosas, "RoSAS")

if __name__ == "__main__":
    main()
