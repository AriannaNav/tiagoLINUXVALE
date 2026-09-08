#!/usr/bin/env python3
"""Offline tests for the graph features: no simulator, no bridge, no microphone.

Covers the affordance layer (priors -> Detection -> ObjectNode -> KG triples, with
class-level inheritance through inference), failure diagnosis with re-grounding to
a different node, and commonsense substitution including the accept/decline
dialogue.

Run:  python test_graph_features.py"""
import json
import os
import tempfile

import waiter
from graph.knowledge_graph import build_waiter_ontology
from graph.object_node import ObjectNode
from graph.perception import characterize
from graph.scene_graph import SceneGraph3D
from graph.temporal_manager import TemporalManager

SGFILE = os.path.join(tempfile.gettempdir(), "sg_test_features.json")
EPFILE = os.path.join(tempfile.gettempdir(), "episodes_test.json")
waiter.SCENE_GRAPH_FILE = SGFILE
waiter.EPISODES_FILE = EPFILE

def make_bottle(nid, color, x):
    """A bottle node as the HSV pipeline would produce it."""
    d = characterize(f"{color}_bottle", None, None, 0.8, (x, -3.2, 0.85),
                     0.9, "simulation", "measured")
    d.color_name, d.color_rgb = color, (200, 40, 40)
    return ObjectNode.from_detection(nid, d)

def fresh_graphs():
    sg, kg = SceneGraph3D(), build_waiter_ontology()
    waiter._GRAPH["sg"], waiter._GRAPH["kg"] = sg, kg
    return sg, kg

def test_affordance_layer():
    sg, kg = fresh_graphs()
    node = make_bottle("green_bottle_1", "green", 1.0)
    assert "grasped" in node.affordances and "pouredFrom" in node.affordances
    sg.add_node(node)
    kg.sync_scene_graph(sg)
    assert kg.ask("green_bottle_1", "canBe", "pouredFrom") is True
    assert kg.ask("green_bottle_1", "canBe", "served") is True
    assert kg.ask("sprite", "canBe", "grasped") is True
    sg.save(SGFILE)
    assert SceneGraph3D.load(SGFILE).nodes["green_bottle_1"].affordances \
        == node.affordances
    waiter.detected_customers = lambda max_age=600: [
        {"table_id": 1, "entity": "male03", "pos": [4.0, 0.53]}]
    waiter.add_customers_to_graph()
    assert "servedTo" in sg.nodes["customer_male03"].affordances
    assert "usedAsSupport" in sg.nodes["table_1"].affordances
    assert ("customer_male03", "isAtTable", "table_1") in sg.edges
    tm = TemporalManager(sg, verbose=False)
    tm.start_tracking()
    tm.update([], fov_check=lambda c, frame=None: True)
    assert "customer_male03" in sg.nodes
    assert ("customer_male03", "isAtTable", "table_1") in sg.edges
    print("1. affordance layer OK")

def test_failure_diagnosis():
    sg, kg = fresh_graphs()
    node = make_bottle("green_bottle_1", "green", 1.0)
    sg.add_node(node)
    kg.sync_scene_graph(sg)
    msg = waiter.diagnose_serve_failure(node)
    assert "positioning" in msg
    assert sg.nodes["green_bottle_1"].status == "uncertain"
    assert json.load(open(SGFILE))["nodes"]
    assert "lost sight" in waiter.diagnose_serve_failure(node)
    sg.remove_node("green_bottle_1")
    assert "lost track" in waiter.diagnose_serve_failure(node)
    assert "fixed bar layout" in waiter.diagnose_serve_failure(None)
    sg.add_node(make_bottle("green_bottle_2", "green", 1.5))
    sg.add_node(make_bottle("green_bottle_3", "green", 2.0))
    kg.sync_scene_graph(sg)
    n1 = waiter.ground_order("sprite")
    n2 = waiter.ground_order("sprite", exclude={n1.node_id})
    assert n2 is not None and n2.node_id != n1.node_id
    print("2. failure diagnosis + re-grounding OK")

def test_substitution():
    os.path.exists(EPFILE) and os.remove(EPFILE)
    sg, kg = fresh_graphs()
    sg.add_node(make_bottle("green_bottle_1", "green", 1.0))
    kg.sync_scene_graph(sg)
    alt, alt_node = waiter.suggest_substitute("coca cola")
    assert alt == "sprite" and alt_node.node_id == "green_bottle_1"
    assert waiter.suggest_substitute("sprite") == (None, None)

    import graph.vlm_perception as _vlmp
    _saved_layout = _vlmp.DRINK_LAYOUT
    _vlmp.DRINK_LAYOUT = {k: v for k, v in _saved_layout.items() if k != "coca cola"}
    try:
        spoken = []
        waiter.say = lambda t, *a, **k: spoken.append(t)
        waiter.listen_en = lambda seconds=4, text_input=False, mood=None: "yes please"
        assert waiter.serve_one("coca cola", None, waiter.TABLES[1], rehearse=True)
        assert any("Sprite instead" in s for s in spoken)
        assert any("Sprite it is" in s for s in spoken)
        spoken.clear()
        waiter.listen_en = lambda seconds=4, text_input=False, mood=None: "no thanks"
        waiter.serve_one("coca cola", None, waiter.TABLES[1], rehearse=True)
        assert any("still try to find some Coca-Cola" in s for s in spoken)
    finally:
        _vlmp.DRINK_LAYOUT = _saved_layout
    print("3. commonsense substitution (accept + decline) OK")

def test_awareness():
    """HRAI 9: measured P/C, symbolic projection Pr, blended Ft + logging."""
    awfile = os.path.join(tempfile.gettempdir(), "awareness_test.json")
    waiter.AWARENESS_FILE = awfile
    sg, kg = fresh_graphs()
    assert waiter.measured_awareness("sprite") is None

    sg.add_node(make_bottle("green_bottle_1", "green", 1.0))
    waiter.detected_customers = lambda max_age=600: [
        {"table_id": 1, "entity": "male03", "pos": [4.0, 0.53]}]
    waiter.add_customers_to_graph()
    kg.sync_scene_graph(sg)

    m = waiter.measured_awareness("sprite")
    assert m["P"] == 1.0, m["required"]
    assert m["C"] == 1.0, m["checks"]
    assert m["Pr"] == 1.0, m["trace"]
    assert [s["step"] for s in m["trace"]] == \
        ["Navigate", "PickObject", "PlaceObject"]

    m2 = waiter.measured_awareness("coca cola")
    assert m2["P"] < 1.0 and not m2["required"]["item visible"]
    assert m2["Pr"] < 1.0
    blocked = [s for s in m2["trace"] if not s["ok"]]
    assert blocked and blocked[0]["step"] == "PickObject"

    ok, unmet = kg.can_perform("PlaceObject", "green_bottle_1")
    assert not ok and "objectHeldByRobot" in unmet[0]
    pr, _ = kg.project_task("ServeItem", "green_bottle_1")
    assert pr == 1.0

    decision = {"action": "SERVE", "item": "sprite", "reasoning": "test",
                "perception_score": 0.1, "comprehension_score": 0.1,
                "projection_score": 0.1, "feasibility_score": 0.1}
    meas = waiter.refine_decision_with_graphs(decision)
    assert decision["scores_source"] == "graph"
    assert decision["feasibility_score"] > 0.9 and decision["action"] == "SERVE"
    waiter.log_awareness("a sprite please", decision, meas)
    logged = json.load(open(awfile))
    assert logged["latest"]["Ft"] == decision["feasibility_score"]
    assert logged["latest"]["details"]["trace"][0]["step"] == "Navigate"
    bad = {"action": "SERVE", "item": "", "reasoning": "",
           "feasibility_score": 0.9}
    sg.nodes.clear()
    sg.add_node(make_bottle("green_bottle_1", "green", 1.0))
    kg2 = build_waiter_ontology()
    waiter._GRAPH["kg"] = kg2
    kg2.sync_scene_graph(sg)
    waiter.refine_decision_with_graphs(bad)
    assert bad["feasibility_score"] < 0.3, bad
    assert bad["action"] == "SERVE", "refine_decision_with_graphs must not override the action"
    os.remove(awfile)
    print("4. measured awareness + projection + logging OK")

def test_commonsense():
    """HRAI 8: causal chains, case-based memory, intention-aware ranking."""
    import contextlib
    import io
    os.path.exists(EPFILE) and os.remove(EPFILE)
    sg, kg = fresh_graphs()
    waiter.say = lambda *a, **k: None

    assert kg.propagate_causes("spilledDrink") == \
        ["floorWet", "floorSlippery", "hazardForHumans"]
    assert kg.propagate_causes("nonEvent") == []

    node = make_bottle("green_bottle_1", "green", 1.0)
    sg.add_node(node)
    kg.sync_scene_graph(sg)
    waiter.report_possible_spill(node)
    assert "hazard_green_bottle_1" in sg.nodes
    assert sg.nodes["hazard_green_bottle_1"].source == "causal_rule"
    waiter.report_possible_spill(node)
    assert sum(1 for n in sg.nodes if n.startswith("hazard")) == 1
    from graph.temporal_manager import TemporalManager
    tm = TemporalManager(sg, verbose=False)
    tm.start_tracking()
    tm.update([], fov_check=lambda c, frame=None: True)
    tm.update([], fov_check=lambda c, frame=None: True)
    assert "hazard_green_bottle_1" in sg.nodes

    waiter.record_episode("serve", item="sprite", pos=[1.0, -3.2],
                          outcome="failed", valence=-1)
    waiter.record_episode("serve", item="sprite", pos=[1.05, -3.2],
                          outcome="failed", valence=-1)
    waiter.record_episode("serve", item="sprite", pos=[9.0, 9.0],
                          outcome="done", valence=1)
    near = waiter.similar_episodes("serve", item="sprite", near=(1.0, -3.2))
    assert len(near) == 2 and all(e["valence"] < 0 for e in near)

    sg.add_node(make_bottle("green_bottle_1", "green", 1.0))
    sg.add_node(make_bottle("green_bottle_2", "green", 2.0))
    kg.sync_scene_graph(sg)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert waiter.serve_one("sprite", None, waiter.TABLES[1], rehearse=True)
    assert "Case memory: 2 past failures near green_bottle_1" in buf.getvalue()

    os.remove(EPFILE)
    sg2, kg2 = fresh_graphs()
    sg2.add_node(make_bottle("green_bottle_1", "green", 1.0))
    blue = make_bottle("blue_bottle_1", "blue", 2.0)
    blue.color_name, blue.color_rgb = "blue", (0, 120, 255)
    sg2.add_node(blue)
    kg2.sync_scene_graph(sg2)
    alt, _ = waiter.suggest_substitute("juice")
    assert alt == "water", alt
    for _ in range(3):
        waiter.record_episode("substitution", item="juice", substitute="water",
                              accepted=False, valence=-1)
        waiter.record_episode("substitution", item="juice", substitute="sprite",
                              accepted=True, valence=1)
    alt2, _ = waiter.suggest_substitute("juice")
    assert alt2 == "sprite", alt2
    print("5. causal chain + case memory + intention priors OK")

if __name__ == "__main__":
    test_affordance_layer()
    test_failure_diagnosis()
    test_substitution()
    test_awareness()
    test_commonsense()
    for f in (SGFILE, EPFILE):
        os.path.exists(f) and os.remove(f)
    print("\nAll graph-feature tests passed.")
