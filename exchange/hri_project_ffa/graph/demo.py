#!/usr/bin/env python3

from .object_node import Detection
from .scene_graph import SceneGraph3D
from .temporal_manager import TemporalManager
from .knowledge_graph import build_waiter_ontology
from .grounding import find_target_in_graph, answer_location

def det(label, xyz, color=None, rgb=None, material="plastic",
        desc="a drink bottle that can be grasped and served", bbox=None):
    return Detection(label=label, confidence=0.9, centroid=xyz, distance=xyz[2],
                     color_name=color or "", color_rgb=rgb, material=material,
                     description=desc, bbox=bbox)

def main():
    sg = SceneGraph3D(frame_id="demo")
    tm = TemporalManager(sg)
    kg = build_waiter_ontology()

    print("======== EXPLORATION PHASE (initial graph construction) ========")
    frame_t0 = [
        det("bottle", (0.4, 0.1, 1.2), "red", (200, 30, 30)),
        det("bottle", (-0.3, 0.1, 1.3), "green", (30, 180, 40)),
        det("cup", (0.0, 0.2, 1.0), "white", (240, 240, 240),
            material="ceramic", desc="a cup for drinks"),
        det("table", (0.0, 0.6, 1.4), material="wood",
            desc="a table where customers sit"),
        det("person", (1.0, -0.2, 2.5), material="organic",
            desc="a customer in the bar"),
    ]
    tm.update(frame_t0)

    print("\n(seeing the same scene again: LOST similarity disambiguates, no duplicates)")
    tm.update(frame_t0)
    assert len(sg.nodes) == 5, "duplicate nodes were created!"

    print("\n======== TRACKING PHASE ========")
    tm.start_tracking()

    print("\n-- t+1: red bottle drifted 10 cm (UPDATE_IN), a NEW blue bottle appears (ADD)")
    tm.update([
        det("bottle", (0.45, 0.1, 1.25), "red", (200, 30, 30)),
        det("bottle", (-0.3, 0.1, 1.3), "green", (30, 180, 40)),
        det("bottle", (0.9, 0.1, 1.1), "blue", (30, 40, 220)),
        det("cup", (0.0, 0.2, 1.0), "white", (240, 240, 240),
            material="ceramic", desc="a cup for drinks"),
        det("table", (0.0, 0.6, 1.4), material="wood",
            desc="a table where customers sit"),
        det("person", (1.0, -0.2, 2.5), material="organic",
            desc="a customer in the bar"),
    ])

    print("\n-- t+2: green bottle found 1.5 m away (UPDATE_OUT: new node, old -> uncertain)")
    tm.update([
        det("bottle", (0.45, 0.1, 1.25), "red", (200, 30, 30)),
        det("bottle", (-1.5, 0.1, 2.2), "green", (30, 180, 40)),
    ], fov_check=lambda c: True)

    print("\n-- t+3: cup still missing while its spot is observed (DELETE)")
    tm.update([
        det("bottle", (0.45, 0.1, 1.25), "red", (200, 30, 30)),
        det("bottle", (-1.5, 0.1, 2.2), "green", (30, 180, 40)),
    ], fov_check=lambda c: True)

    print("\n======== RESULTING SCENE GRAPH ========")
    print(sg.to_text())

    print("\n======== KNOWLEDGE GRAPH (HRAI 6) ========")
    kg.sync_scene_graph(sg)
    print(f"{len(kg.triples)} triples after ABox sync + inference.\n")

    print('query(None, "isA", "Drink")   [inferred: bottle isA Drink subClassOf Item]')
    for tr in kg.query(None, "isA", "Drink"):
        print("   ", tr)
    print('\nquery(None, "hasColor", None)')
    for tr in kg.query(None, "hasColor", None):
        print("   ", tr)
    print('\nask(red bottle node, canBe, grasped)  [rule: movable => graspable]')
    red = next(n for n in sg.nodes.values() if n.color_name == "red")
    print("   ", kg.ask(red.node_id, "canBe", "grasped"))
    ok, unmet = kg.can_perform("PickObject", red.node_id)
    print(f'\ncan_perform("PickObject", {red.node_id}) -> {ok} {unmet}')
    ok, unmet = kg.can_perform("ServeItem", "coca cola")
    print(f'can_perform("ServeItem", "coca cola") -> {ok} {unmet}')

    print("\n======== GROUNDING A USER REQUEST ========")
    req = {"intent": "find_object", "target": "bottle", "constraints": ["red"]}
    node = find_target_in_graph(sg, kg, req)
    print(f'request {req}\n -> {node.node_id}')
    print("   ", answer_location(sg, node))
    req2 = {"intent": "find_object", "target": "drink", "constraints": []}
    node2 = find_target_in_graph(sg, kg, req2)
    print(f'request {req2}\n -> {node2.node_id} (matched via ontology class, not label)')

    print("\n======== RDF TURTLE EXPORT (first lines) ========")
    print("\n".join(kg.to_turtle().splitlines()[:8]) + "\n...")

if __name__ == "__main__":
    main()
