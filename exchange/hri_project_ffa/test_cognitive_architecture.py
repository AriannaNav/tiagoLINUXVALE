#!/usr/bin/env python3
"""Unit tests for the Cognitive Architecture.

Covers the representation and reasoning layers: the spatial relations of the
3D scene graph, the LOST similarity function, the four operations of the
temporal manager, the inference and the task preconditions of the knowledge
graph, the grounding of a request onto a perceived object, and the code that
produces the experimental measurements.

Everything runs offline. Run with

    python -m unittest test_cognitive_architecture -v
"""
import json
import os
import tempfile
import unittest

from graph.knowledge_graph import KnowledgeGraph, build_waiter_ontology
from graph.object_node import Detection, ObjectNode
from graph.scene_graph import SceneGraph3D
from graph.similarity import (color_similarity, description_similarity,
                              lost_similarity, word_similarity)
from graph.temporal_manager import TemporalManager
from graph.grounding import find_target_in_graph


def node(nid, label, xyz, frame="map", shape="bottle", color="red",
         rgb=(200, 30, 30), source="vlm", status="confirmed"):
    return ObjectNode(node_id=nid, label=label, centroid=xyz, frame=frame,
                      shape=shape, color_name=color, color_rgb=rgb,
                      material="glass", source=source, status=status,
                      description=f"a {color} {label}")


def det(label, xyz, color="red", rgb=(200, 30, 30), frame="map"):
    return Detection(label=label, confidence=0.9, centroid=xyz, frame=frame,
                     color_name=color, color_rgb=rgb, material="glass",
                     description=f"a {color} {label}")


class SpatialRelations(unittest.TestCase):
    """scene_graph: geometric edges recomputed from the current poses."""

    def setUp(self):
        self.sg = SceneGraph3D(frame_id="map")

    def test_an_object_on_a_surface_gets_is_on_top_of(self):
        self.sg.add_node(node("counter", "counter", (2.0, -3.2, 0.40),
                              shape="surface", source="ground_truth"))
        self.sg.add_node(node("coke", "coca cola", (2.0, -3.2, 0.85)))
        self.sg.refresh_relations()
        self.assertIn(("coke", "isOnTopOf", "counter"), self.sg.edges)

    def test_two_bottles_side_by_side_get_left_and_right(self):
        self.sg.add_node(node("a", "coca cola", (2.0, -3.2, 0.85)))
        self.sg.add_node(node("b", "sprite", (2.5, -3.2, 0.85), color="green"))
        self.sg.refresh_relations()
        self.assertIn(("a", "isLeftOf", "b"), self.sg.edges)
        self.assertIn(("b", "isRightOf", "a"), self.sg.edges)

    def test_close_objects_are_related_as_near(self):
        self.sg.add_node(node("p1", "person", (1.0, 1.0, 0.9), shape="human"))
        self.sg.add_node(node("p2", "person", (1.3, 1.0, 0.9), shape="human"))
        self.sg.refresh_relations()
        self.assertIn(("p1", "near", "p2"), self.sg.edges)

    def test_distant_objects_are_not_related(self):
        self.sg.add_node(node("p1", "person", (0.0, 0.0, 0.9), shape="human"))
        self.sg.add_node(node("p2", "person", (5.0, 5.0, 0.9), shape="human"))
        self.sg.refresh_relations()
        self.assertEqual(self.sg.edges, [])

    def test_objects_in_different_frames_are_never_related(self):
        self.sg.add_node(node("a", "coca cola", (0.1, 0.0, 1.0), frame="camera"))
        self.sg.add_node(node("b", "sprite", (0.2, 0.0, 1.0), frame="map"))
        self.sg.refresh_relations()
        self.assertEqual(self.sg.edges, [])

    def test_semantic_edges_survive_a_refresh(self):
        self.sg.add_node(node("cust", "person", (4.0, 2.5, 0.9), shape="human"))
        self.sg.add_node(node("table_1", "table", (4.4, 3.2, 0.4),
                              shape="surface"))
        self.sg.edges.append(("cust", "isAtTable", "table_1"))
        self.sg.refresh_relations()
        self.assertIn(("cust", "isAtTable", "table_1"), self.sg.edges)

    def test_edges_of_a_removed_node_are_dropped(self):
        self.sg.add_node(node("cust", "person", (4.0, 2.5, 0.9), shape="human"))
        self.sg.add_node(node("table_1", "table", (4.4, 3.2, 0.4), shape="surface"))
        self.sg.edges.append(("cust", "isAtTable", "table_1"))
        self.sg.remove_node("table_1")
        self.sg.refresh_relations()
        self.assertEqual(self.sg.edges, [])

    def test_the_graph_survives_a_save_and_reload(self):
        self.sg.add_node(node("coke", "coca cola", (2.0, -3.2, 0.85)))
        path = os.path.join(tempfile.gettempdir(), "sg_unit_test.json")
        self.sg.save(path)
        loaded = SceneGraph3D.load(path)
        self.assertEqual(loaded.nodes["coke"].label, "coca cola")
        os.remove(path)


class LostSimilarity(unittest.TestCase):
    """similarity: the function that decides whether two observations are the
    same physical object."""

    def test_identical_labels_score_one(self):
        self.assertEqual(word_similarity("sprite", "sprite"), 1.0)

    def test_synonyms_score_high(self):
        self.assertGreaterEqual(word_similarity("bottle", "can"), 0.85)

    def test_unrelated_labels_score_low(self):
        self.assertLess(word_similarity("sprite", "table"), 0.6)

    def test_identical_colours_score_one(self):
        self.assertAlmostEqual(color_similarity((200, 30, 30), (200, 30, 30)), 1.0)

    def test_opposite_colours_score_low(self):
        self.assertLess(color_similarity((0, 0, 0), (255, 255, 255)), 0.1)

    def test_missing_colour_is_skipped(self):
        self.assertIsNone(color_similarity(None, (1, 2, 3)))

    def test_description_similarity_is_symmetric(self):
        a, b = "a red bottle of cola", "a bottle of red cola"
        self.assertAlmostEqual(description_similarity(a, b),
                               description_similarity(b, a))

    def test_same_object_scores_above_the_matching_threshold(self):
        n = node("a", "coca cola", (2.0, -3.2, 0.85))
        d = det("coca cola", (2.05, -3.2, 0.85))
        self.assertGreater(lost_similarity(d, n), 0.55)

    def test_different_colour_lowers_the_score(self):
        n = node("a", "coca cola", (2.0, -3.2, 0.85))
        same = lost_similarity(det("coca cola", (2.0, -3.2, 0.85)), n)
        other = lost_similarity(
            det("coca cola", (2.0, -3.2, 0.85), color="green", rgb=(30, 180, 40)), n)
        self.assertGreater(same, other)


class TemporalReasoning(unittest.TestCase):
    """temporal_manager: ADD, UPDATE_IN, UPDATE_OUT and DELETE."""

    def setUp(self):
        self.sg = SceneGraph3D(frame_id="map")
        self.tm = TemporalManager(self.sg, verbose=False)

    def ops(self, log):
        return [entry[0] for entry in log]

    def test_a_new_detection_is_added(self):
        log = self.tm.update([det("coca cola", (2.0, -3.2, 0.85))])
        self.assertIn("ADD", self.ops(log))
        self.assertEqual(len(self.sg.nodes), 1)

    def test_the_same_object_is_not_duplicated(self):
        d = det("coca cola", (2.0, -3.2, 0.85))
        self.tm.update([d])
        self.tm.update([d])
        self.assertEqual(len(self.sg.nodes), 1)

    def test_a_small_displacement_updates_the_node_in_place(self):
        self.tm.update([det("coca cola", (2.0, -3.2, 0.85))])
        self.tm.start_tracking()
        log = self.tm.update([det("coca cola", (2.1, -3.2, 0.85))])
        self.assertIn("UPDATE_IN", self.ops(log))
        self.assertEqual(len(self.sg.nodes), 1)

    def test_a_large_displacement_creates_a_second_node(self):
        self.tm.update([det("coca cola", (2.0, -3.2, 0.85))])
        self.tm.start_tracking()
        log = self.tm.update([det("coca cola", (0.5, -3.2, 0.85))])
        self.assertIn("UPDATE_OUT", self.ops(log))
        self.assertEqual(len(self.sg.nodes), 2)

    def test_the_previous_pose_is_kept_as_uncertain(self):
        self.tm.update([det("coca cola", (2.0, -3.2, 0.85))])
        self.tm.start_tracking()
        self.tm.update([det("coca cola", (0.5, -3.2, 0.85))])
        self.assertIn("uncertain",
                      [n.status for n in self.sg.nodes.values()])

    def test_nothing_is_deleted_without_a_field_of_view_check(self):
        self.tm.update([det("coca cola", (2.0, -3.2, 0.85))])
        self.tm.start_tracking()
        self.tm.update([])
        self.assertEqual(len(self.sg.nodes), 1)

    def test_a_missing_object_becomes_uncertain_before_being_deleted(self):
        self.tm.update([det("coca cola", (2.0, -3.2, 0.85))])
        self.tm.start_tracking()
        seen = lambda c, frame=None: True
        self.tm.update([], fov_check=seen)
        self.assertEqual(list(self.sg.nodes.values())[0].status, "uncertain")
        self.tm.update([], fov_check=seen)
        self.assertEqual(len(self.sg.nodes), 0)

    def test_furniture_from_ground_truth_is_never_deleted(self):
        self.sg.add_node(node("table_1", "table", (4.4, 3.2, 0.4),
                              shape="surface", source="ground_truth"))
        self.tm.start_tracking()
        seen = lambda c, frame=None: True
        self.tm.update([], fov_check=seen)
        self.tm.update([], fov_check=seen)
        self.assertIn("table_1", self.sg.nodes)


class OntologyAndInference(unittest.TestCase):
    """knowledge_graph: TBox, ABox and the rules that derive new facts."""

    def setUp(self):
        self.kg = build_waiter_ontology()

    def test_subclass_relations_are_transitive(self):
        self.assertTrue(self.kg.ask("Drink", "subClassOf", "PhysicalObject"))

    def test_class_membership_is_inherited(self):
        self.assertTrue(self.kg.ask("sprite", "isA", "Item"))

    def test_a_movable_object_is_inferred_to_be_graspable(self):
        kg = KnowledgeGraph()
        kg.add("bottle_1", "isMovable", "yes")
        kg.infer()
        self.assertTrue(kg.ask("bottle_1", "canBe", "grasped"))

    def test_affordances_are_inherited_from_the_class(self):
        self.assertTrue(self.kg.ask("wine", "canBe", "pouredFrom"))

    def test_an_unknown_fact_is_unknown_not_false(self):
        self.assertIsNone(self.kg.ask("sprite", "isA", "Furniture"))

    def test_location_propagates_through_support(self):
        kg = KnowledgeGraph()
        kg.add("counter", "isLocatedIn", "Bar")
        kg.add("bottle_1", "isOnTopOf", "counter")
        kg.infer()
        self.assertTrue(kg.ask("bottle_1", "isLocatedIn", "Bar"))

    def test_a_removed_fact_is_no_longer_entailed(self):
        kg = KnowledgeGraph()
        kg.add("x", "isA", "Drink")
        self.assertTrue(kg.ask("x", "isA", "Drink"))
        kg.remove("x", "isA", "Drink")
        self.assertIsNone(kg.ask("x", "isA", "Drink"))

    def test_a_query_with_wildcards_returns_every_match(self):
        drinks = {h for h, _, _ in self.kg.query(None, "isA", "Drink")}
        self.assertIn("sprite", drinks)
        self.assertIn("wine", drinks)

    def test_the_graph_exports_to_turtle(self):
        ttl = self.kg.to_turtle()
        self.assertTrue(ttl.startswith("@prefix bar:"))
        self.assertIn("bar:sprite", ttl)


class TaskPreconditions(unittest.TestCase):
    """knowledge_graph: what the robot is allowed to attempt."""

    def setUp(self):
        self.kg = build_waiter_ontology()

    def test_a_drink_satisfies_the_serve_precondition(self):
        ok, unmet = self.kg.can_perform("ServeItem", "sprite")
        self.assertTrue(ok)
        self.assertEqual(unmet, [])

    def test_a_snack_also_satisfies_it(self):
        self.assertTrue(self.kg.can_perform("ServeItem", "pringles")[0])

    def test_a_non_servable_target_is_vetoed_with_a_reason(self):
        ok, unmet = self.kg.can_perform("ServeItem", "chair")
        self.assertFalse(ok)
        self.assertTrue(any("servable" in reason for reason in unmet))

    def test_placing_requires_holding_the_object_first(self):
        ok, unmet = self.kg.can_perform("PlaceObject", "sprite")
        self.assertFalse(ok)
        self.assertTrue(any("objectHeldByRobot" in reason for reason in unmet))


class TaskProjection(unittest.TestCase):
    """knowledge_graph: symbolic look-ahead over the subtask chain."""

    def setUp(self):
        self.kg = build_waiter_ontology()
        self.kg.add("sprite_1", "hasLabel", "sprite", abox=True)
        self.kg.add("sprite_1", "isA", "Drink", abox=True)
        self.kg.add("sprite_1", "isMovable", "yes", abox=True)
        self.kg.infer()

    def test_a_perceived_graspable_drink_projects_to_completion(self):
        pr, trace = self.kg.project_task("ServeItem", "sprite_1")
        self.assertEqual(pr, 1.0)
        self.assertEqual([s["step"] for s in trace],
                         ["Navigate", "PickObject", "PlaceObject"])

    def test_an_unperceived_target_blocks_the_chain_at_the_pick(self):
        pr, trace = self.kg.project_task("ServeItem", "never_seen")
        self.assertLess(pr, 1.0)
        blocked = [s for s in trace if not s["ok"]]
        self.assertEqual(blocked[0]["step"], "PickObject")

    def test_the_effects_of_a_step_enable_the_next_one(self):
        _, trace = self.kg.project_task("ServeItem", "sprite_1")
        self.assertTrue(all(s["ok"] for s in trace))


class CausalReasoning(unittest.TestCase):
    """knowledge_graph: few causes, large effects."""

    def setUp(self):
        self.kg = build_waiter_ontology()

    def test_a_spill_propagates_to_a_hazard(self):
        self.assertEqual(self.kg.propagate_causes("spilledDrink"),
                         ["floorWet", "floorSlippery", "hazardForHumans"])

    def test_an_unknown_event_has_no_consequences(self):
        self.assertEqual(self.kg.propagate_causes("nothingHappened"), [])

    def test_the_chain_stops_at_the_last_effect(self):
        self.assertEqual(self.kg.propagate_causes("floorSlippery"),
                         ["hazardForHumans"])


class RequestGrounding(unittest.TestCase):
    """grounding: from a word to a perceived object."""

    def setUp(self):
        self.sg = SceneGraph3D(frame_id="map")
        self.sg.add_node(node("coke_1", "coca cola", (2.5, -3.2, 0.85)))
        self.sg.add_node(node("sprite_1", "sprite", (2.0, -3.2, 0.85),
                              color="green", rgb=(30, 180, 40)))
        self.kg = build_waiter_ontology()
        self.kg.sync_scene_graph(self.sg)

    def test_a_request_is_grounded_by_label(self):
        n = find_target_in_graph(self.sg, self.kg,
                                 {"target": "sprite", "constraints": []})
        self.assertEqual(n.node_id, "sprite_1")

    def test_a_request_is_grounded_by_ontology_class(self):
        n = find_target_in_graph(self.sg, self.kg,
                                 {"target": "drink", "constraints": []})
        self.assertIn(n.node_id, ("coke_1", "sprite_1"))

    def test_a_colour_constraint_selects_the_right_object(self):
        n = find_target_in_graph(self.sg, self.kg,
                                 {"target": "drink", "constraints": ["green"]})
        self.assertEqual(n.node_id, "sprite_1")

    def test_an_excluded_node_is_not_returned_again(self):
        n = find_target_in_graph(self.sg, self.kg,
                                 {"target": "drink", "constraints": []},
                                 exclude={"coke_1", "sprite_1"})
        self.assertIsNone(n)

    def test_an_absent_item_grounds_to_nothing(self):
        n = find_target_in_graph(self.sg, self.kg,
                                 {"target": "wine", "constraints": []})
        self.assertIsNone(n)

    def test_a_confirmed_object_is_preferred_over_an_uncertain_one(self):
        self.sg.nodes["sprite_1"].status = "uncertain"
        self.sg.add_node(node("sprite_2", "sprite", (1.5, -3.2, 0.85),
                              color="green", rgb=(30, 180, 40)))
        self.kg.sync_scene_graph(self.sg)
        n = find_target_in_graph(self.sg, self.kg,
                                 {"target": "sprite", "constraints": []})
        self.assertEqual(n.node_id, "sprite_2")


class MotionMetrics(unittest.TestCase):
    """metrics_logger: the objective quantities of the evaluation."""

    def setUp(self):
        import metrics_logger
        self.mod = metrics_logger
        self.dir = tempfile.mkdtemp()
        with open(os.path.join(self.dir, "customers_present.json"), "w") as f:
            json.dump({"present": [{"table_id": 1, "entity": "male03",
                                    "pos": [4.0, 0.0]}]}, f)
        self.m = metrics_logger.MotionMetrics(self.dir)

    def move_to(self, x, y):
        with open(os.path.join(self.dir, "robot_pose.json"), "w") as f:
            json.dump({"x": x, "y": y, "yaw": 0.0, "ts": 0}, f)
        self.m._step()

    def test_path_length_accumulates_real_motion(self):
        self.move_to(0.0, 0.0)
        self.move_to(0.5, 0.0)
        self.move_to(0.9, 0.0)
        self.assertAlmostEqual(self.m.path_length, 0.9, places=2)

    def test_pose_noise_while_standing_still_is_ignored(self):
        self.move_to(0.0, 0.0)
        for _ in range(20):
            self.move_to(0.002, 0.0)
        self.assertEqual(self.m.path_length, 0.0)

    def test_a_pose_jump_is_not_counted_as_travel(self):
        self.move_to(0.0, 0.0)
        self.move_to(9.0, 9.0)
        self.assertEqual(self.m.path_length, 0.0)

    def test_the_closest_approach_is_recorded(self):
        self.move_to(0.0, 0.0)
        self.move_to(3.4, 0.0)
        self.move_to(2.0, 0.0)
        self.assertAlmostEqual(self.m.snapshot()["min_distance_m"], 0.6, places=2)

    def test_one_approach_counts_as_one_intrusion(self):
        self.move_to(0.0, 0.0)
        for _ in range(30):          # the robot stands at the table and talks
            self.move_to(3.4, 0.0)
        self.assertEqual(self.m.snapshot()["intrusions"], 1)

    def test_leaving_and_returning_counts_twice(self):
        self.move_to(0.0, 0.0)
        self.move_to(3.4, 0.0)       # inside personal space
        self.move_to(0.0, 0.0)       # well beyond the release radius
        self.move_to(3.4, 0.0)       # back again
        self.assertEqual(self.m.snapshot()["intrusions"], 2)

    def test_staying_outside_personal_space_counts_nothing(self):
        self.move_to(0.0, 0.0)
        self.move_to(2.0, 0.0)
        self.assertEqual(self.m.snapshot()["intrusions"], 0)

    def test_a_missing_pose_file_is_tolerated(self):
        m = self.mod.MotionMetrics(tempfile.mkdtemp())
        m._step()
        self.assertEqual(m.samples, 0)


class EffectSize(unittest.TestCase):
    """analyze_experiment: the statistic reported next to every p-value."""

    def setUp(self):
        import analyze_experiment
        self.ae = analyze_experiment

    def test_a_consistent_difference_gives_a_large_effect(self):
        # The differences must vary: the paired d divides by their standard
        # deviation, so a constant difference leaves it undefined.
        a, b = [10, 11, 12, 13], [21, 20, 23, 22]
        self.assertGreater(abs(self.ae.cohens_d(a, b, paired=True)), 0.8)

    def test_a_constant_difference_leaves_the_paired_effect_undefined(self):
        d = self.ae.cohens_d([1, 2, 3], [11, 12, 13], paired=True)
        self.assertNotEqual(d, d)

    def test_identical_samples_give_no_effect(self):
        d = self.ae.cohens_d([5, 5, 5], [5, 5, 5], paired=True)
        self.assertNotEqual(d, d)      # NaN: undefined, not zero

    def test_the_unpaired_formula_uses_the_pooled_deviation(self):
        d = self.ae.cohens_d([10, 12, 14], [20, 22], paired=False)
        self.assertLess(d, 0)

    def test_the_conventional_reading_of_the_magnitude(self):
        self.assertEqual(self.ae.magnitude(0.1), "negligible")
        self.assertEqual(self.ae.magnitude(0.3), "small")
        self.assertEqual(self.ae.magnitude(0.6), "medium")
        self.assertEqual(self.ae.magnitude(1.5), "large")


if __name__ == "__main__":
    unittest.main(verbosity=2)
