# Graph layer — Scene Graph + Knowledge Graph

Implements the architecture from the course decks **"HRAI 5 – Scene Graphs"**
(LOST-3DSG-style lightweight temporal scene graph) and **"HRAI 6 – Graph
Operations"** (knowledge graphs, ontologies, queries, embeddings) on top of the
existing waiter pipeline.

## Mapping slides → code

| Slide concept | File |
|---|---|
| Perception module: semantic + geometric characterization (label, color, material, shape, isMovable, description) | `perception.py` |
| Object identity from a vision-language model (counter stock, table occupancy) | `vlm_perception.py` |
| LOST Similarity Function (label / color / material / description) | `similarity.py` |
| Temporal Reasoning module: exploration & tracking phases, **ADD / UPDATE_IN / UPDATE_OUT / DELETE**, uncertain-object set, POV-volume disappearance detector | `temporal_manager.py` |
| 3D Scene Graph (semantic map ⟨R, M, P⟩, spatial-relation edges) | `scene_graph.py`, `object_node.py` |
| Knowledge Graph G=(V,E) of (h, r, t) triples, TBox/ABox, RDFS inference, Open World Assumption, SPARQL-like queries, RDF Turtle export, optional pykeen KGE link prediction | `knowledge_graph.py` |
| Top-level / domain / task / application ontology of the bar | `build_waiter_ontology()` |
| Language-grounded queries over the graph ("the red bottle near the cup", "a drink") | `grounding.py` |

## How to run

```bash
cd exchange/hri_project_ffa

# Offline demo of every graph operation (no camera, sim or LLM needed):
python -m graph.demo

# Offline tests of the affordance, failure-diagnosis and substitution paths:
python test_graph_features.py
```

The live graph has no separate entry point: `waiter.py` runs it in its own
thread over the bridge camera frames, so starting the waiter starts the graph.

## Integration with the existing pipeline

- Detections reach the `TemporalManager`, so the LLM reasons over the
  **persistent scene graph** — objects survive occlusions, duplicates are merged
  by the LOST similarity, moved objects are tracked — instead of the raw last
  frame. User requests are grounded with `graph.grounding.find_target_in_graph`
  (ontology classes + constraints), falling back to label matching.
- `waiter.py` **relies on the graph** (no separate process needed): at start it
  spawns its own scene-graph thread over the bridge camera frames
  (`start_scene_graph()`, disable with `WAITER_GRAPH=0`), then runs an
  **active-exploration patrol** (S-AVE idea, HRAI 5): it sends the bridge the
  new `explore` action, TIAGo drives past both counters (`EXPLORE_VIEWPOINTS`
  in `ros_nodes/hri_bridge.py`, dwelling at each so frames flow to the host)
  while the temporal manager is in its exploration phase; when the bridge
  reports `explored`, `end_exploration()` switches the graph to tracking.
  Disable the patrol with `WAITER_EXPLORE=0`. The brain's SCENE
  context comes from the live graph; before dispatching an order the knowledge
  graph must approve it (`kg.can_perform("ServeItem", item)` — ontology
  precondition `targetIsDrinkOrSnack`), and the item is grounded to the
  perceived scene-graph node (`ground_order()`), which is forwarded to the
  container bridge inside `command.json` as `"grounding"` and logged by
  `ros_nodes/hri_bridge.py`.
- The knowledge graph is exported to `shared/knowledge_graph.ttl` (RDF Turtle).
- **Map-frame positions**: `../frame_grabber_tiago.py` writes `robot_pose.json`
  (ground-truth base pose + TF camera extrinsic) next to the frames. Counter
  drinks are stored directly in the map frame at the known counter positions —
  identity is perceived by the VLM, geometry comes from the layout — and
  `RobotPose` converts back to the camera frame for the FOV/DELETE check.
  Matching and spatial relations are frame-aware. The bridge then **navigates to
  where the bottle was actually seen** (`grasp_base_from_grounding`, clamped to
  the grasp lane) when `HRI_RESTOCK=0`; `BOTTLE_HOME` remains the fallback for
  never-perceived items or camera-frame-only groundings.

## Optional extras

- `SG_USE_WORD2VEC=1` + `pip install gensim` → real word2vec label/material
  cosine similarity (as in the slides) instead of the lexical fallback.
- `pip install pykeen` → `kg.train_embeddings()` / `kg.predict_missing(h, r)`
  for open-world knowledge-graph completion (TransE & co., HRAI 6).
