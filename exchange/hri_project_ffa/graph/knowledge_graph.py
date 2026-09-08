
from collections import defaultdict

class KnowledgeGraph:
    def __init__(self):
        self.triples = set()
        self._abox_marks = set()
        self._kge = None

    def add(self, h, r, t, abox=False):
        triple = (str(h), str(r), str(t))
        self.triples.add(triple)
        if abox:
            self._abox_marks.add(triple)

    def remove(self, h, r, t):
        self.triples.discard((h, r, t))
        self._abox_marks.discard((h, r, t))

    def query(self, h=None, r=None, t=None):
        """Pattern matching over triples; None is a wildcard (like SPARQL vars).
        Example: query(None, 'isA', 'Drink') -> all drinks in the scene."""
        return [(h_, r_, t_) for (h_, r_, t_) in sorted(self.triples)
                if (h is None or h_ == h)
                and (r is None or r_ == r)
                and (t is None or t_ == t)]

    def ask(self, h, r, t):
        """Truth of a fact under the Open World Assumption:
        True if stated/inferred, otherwise None ("we simply do not know")."""
        return True if (h, r, t) in self.triples else None

    def infer(self):
        """Materialize inferred triples; returns how many facts were added."""
        added = 0
        changed = True
        while changed:
            changed = False
            subs = self.query(r="subClassOf")
            index = defaultdict(set)
            for a, _, b in subs:
                index[a].add(b)
            for a, _, b in list(subs):
                for c in index.get(b, ()):
                    if (a, "subClassOf", c) not in self.triples:
                        self.add(a, "subClassOf", c); added += 1; changed = True
            for x, _, klass in self.query(r="isA"):
                for _, _, parent in self.query(h=klass, r="subClassOf"):
                    if (x, "isA", parent) not in self.triples:
                        self.add(x, "isA", parent); added += 1; changed = True
            for x, _, y in self.query(r="isOnTopOf"):
                for _, _, z in self.query(h=y, r="isLocatedIn"):
                    if (x, "isLocatedIn", z) not in self.triples:
                        self.add(x, "isLocatedIn", z); added += 1; changed = True
            for x, _, _ in self.query(r="isMovable", t="yes"):
                if (x, "canBe", "grasped") not in self.triples:
                    self.add(x, "canBe", "grasped"); added += 1; changed = True
            for klass, _, aff in self.query(r="affords"):
                for x, _, _ in self.query(r="isA", t=klass):
                    if (x, "canBe", aff) not in self.triples:
                        self.add(x, "canBe", aff); added += 1; changed = True
        return added

    def sync_scene_graph(self, sg, instance_classes=None):
        """Replace the previous scene-derived facts with the current scene
        graph, then classify each instance against the ontology and re-infer."""
        for triple in list(self._abox_marks):
            self.remove(*triple)
        classes = instance_classes or LABEL_TO_CLASS
        for h, r, t in sg.to_triples():
            self.add(h, r, t, abox=True)
            if r == "hasLabel":
                klass = classes.get(t)
                if klass:
                    self.add(h, "isA", klass, abox=True)
        self.infer()

    def _unmet(self, pre, target, state=()):
        """Why precondition `pre` fails for `target`, or None if it holds.
        `state` is the set of effects already achieved (symbolic projection)."""
        if pre in state:
            return None
        if pre == "targetIsGraspable":
            if self.ask(target, "canBe", "grasped") is not True:
                return f"{target} is not known to be graspable"
        elif pre == "targetPerceived":
            if not self.query(h=target, r="hasLabel"):
                return f"{target} is not in the scene graph"
        elif pre == "targetIsDrinkOrSnack":
            if (self.ask(target, "isA", "Drink") is not True
                    and self.ask(target, "isA", "Snack") is not True):
                return f"{target} is not a servable drink/snack"
        elif pre in ("objectHeldByRobot", "objectOnTargetSurface"):
            return f"{pre} not yet achieved"
        return None

    def can_perform(self, task, target):
        """Check a task's preconditions against the KG.
        Returns (ok, [unmet precondition descriptions])."""
        unmet = [w for _, _, pre in self.query(h=task, r="hasPrecondition")
                 for w in [self._unmet(pre, target)] if w]
        return (len(unmet) == 0), unmet

    def propagate_causes(self, event):
        """Chase the 'causes' chains from an asserted event: few causes, large
        effects (spilledDrink -> floorWet -> floorSlippery -> hazardForHumans).
        Returns the ordered list of derived effects."""
        effects, frontier, seen = [], [event], {event}
        while frontier:
            cur = frontier.pop(0)
            for _, _, eff in self.query(h=cur, r="causes"):
                if eff not in seen:
                    seen.add(eff)
                    effects.append(eff)
                    frontier.append(eff)
        return effects

    TASK_PLANS = {"ServeItem": ("Navigate", "PickObject", "PlaceObject")}

    def project_task(self, task, target):
        """Projection of future status (HRAI 9): symbolically execute the task's subtask
        chain one step ahead, checking each step's preconditions against the KG facts plus
        the effects accumulated so far, so a broken step blocks the rest. Returns (Pr,
        trace) with Pr the fraction of steps still feasible."""
        steps = list(self.TASK_PLANS.get(task, ())) or \
            [t for _, _, t in self.query(h=task, r="hasSubtask")] or [task]
        state, trace, ok_steps = set(), [], 0
        for step in steps:
            unmet = [w for _, _, pre in self.query(h=step, r="hasPrecondition")
                     for w in [self._unmet(pre, target, state)] if w]
            if unmet:
                trace.append({"step": step, "ok": False, "unmet": unmet})
                break
            for _, _, eff in self.query(h=step, r="hasEffect"):
                state.add(eff)
            ok_steps += 1
            trace.append({"step": step, "ok": True, "unmet": []})
        return ok_steps / len(steps), trace

    def train_embeddings(self, model="TransE", epochs=100):
        """Train a KGE model on the current triples with pykeen (optional dep).
        Enables predict_missing() to score unseen facts (open-world KGC)."""
        import numpy as np
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
        arr = np.array(sorted(self.triples), dtype=str)
        tf = TriplesFactory.from_labeled_triples(arr)
        self._kge = pipeline(model=model, training=tf, testing=tf,
                             training_kwargs=dict(num_epochs=epochs,
                                                  use_tqdm_batch=False))
        return self._kge

    def predict_missing(self, h, r, top_k=5):
        """Rank candidate tails for (h, r, ?) with the trained KGE model."""
        if self._kge is None:
            raise RuntimeError("call train_embeddings() first")
        from pykeen.predict import predict_target
        pred = predict_target(model=self._kge.model, head=h, relation=r,
                              triples_factory=self._kge.training)
        return pred.df.head(top_k)[["tail_label", "score"]].values.tolist()

    def to_turtle(self, prefix="bar"):
        """Export as RDF Turtle (Linked Data-style serialization)."""
        lines = [f"@prefix {prefix}: <http://example.org/{prefix}#> ."]
        for h, r, t in sorted(self.triples):
            def term(x):
                return f'{prefix}:{str(x).replace(" ", "_")}'
            lines.append(f"{term(h)} {term(r)} {term(t)} .")
        return "\n".join(lines)

LABEL_TO_CLASS = {
    "person": "Agent", "bottle": "Drink", "cup": "Tableware",
    "wine glass": "Tableware", "table": "Furniture", "dining table": "Furniture",
    "chair": "Furniture", "book": "Item", "vase": "Item",
    "red bottle": "Drink", "green bottle": "Drink", "blue bottle": "Drink",
    "yellow bottle": "Drink", "purple bottle": "Drink",
    "coca cola": "Drink", "sprite": "Drink", "water": "Drink",
    "juice": "Drink", "wine": "Drink", "pringles": "Snack",
    "counter east": "Furniture", "counter west": "Furniture",
}

def build_waiter_ontology():
    """TBox for the TIAGo bar: top-level + domain + task ontology (HRAI 6,
    'Ontology - Types': top-level / domain / task / application)."""
    kg = KnowledgeGraph()

    for sub, sup in [("PhysicalObject", "Entity"), ("Place", "Entity"),
                     ("Task", "Entity"), ("State", "Entity"),
                     ("Agent", "PhysicalObject"), ("Item", "PhysicalObject"),
                     ("Furniture", "PhysicalObject"), ("Container", "Item")]:
        kg.add(sub, "subClassOf", sup)

    for sub, sup in [("Drink", "Item"), ("Snack", "Item"),
                     ("Tableware", "Container")]:
        kg.add(sub, "subClassOf", sup)
    for place in ("Bar", "Kitchen", "CustomerTable", "Counter", "Bathroom",
                  "Reception"):
        kg.add(place, "isA", "Place")
        kg.add(place, "subClassOf", "Place")
    for drink in ("coca cola", "sprite", "water", "juice", "wine"):
        kg.add(drink, "isA", "Drink")
        kg.add(drink, "isLocatedIn", "Counter")
    kg.add("pringles", "isA", "Snack")
    kg.add("pringles", "isLocatedIn", "Counter")

    for klass, affs in {
        "Drink":     ("grasped", "carried", "served", "pouredFrom"),
        "Snack":     ("grasped", "carried", "served"),
        "Tableware": ("grasped", "filled"),
        "Furniture": ("approached", "usedAsSupport"),
        "Agent":     ("greeted", "askedForOrder", "servedTo"),
        "Place":     ("approached", "indicated"),
    }.items():
        for a in affs:
            kg.add(klass, "affords", a)

    for task in ("Navigate", "PickObject", "PlaceObject", "ServeItem", "TakeOrder",
                 "IndicatePlace"):
        kg.add(task, "isA", "Task")
    kg.add("IndicatePlace", "hasPrecondition", "placeKnown")
    kg.add("ServeAlcohol", "isA", "Task")
    kg.add("ServeAlcohol", "hasPrecondition", "customerAgeVerified")
    kg.add("AgeCheck", "isA", "Task")
    kg.add("AgeCheck", "isPerformedAt", "Reception")
    kg.add("PickObject", "hasPrecondition", "targetPerceived")
    kg.add("PickObject", "hasPrecondition", "targetIsGraspable")
    kg.add("PickObject", "hasEffect", "objectHeldByRobot")
    kg.add("PlaceObject", "hasPrecondition", "objectHeldByRobot")
    kg.add("PlaceObject", "hasEffect", "objectOnTargetSurface")
    kg.add("ServeItem", "hasPrecondition", "targetIsDrinkOrSnack")
    kg.add("ServeItem", "hasSubtask", "Navigate")
    kg.add("ServeItem", "hasSubtask", "PickObject")
    kg.add("ServeItem", "hasSubtask", "PlaceObject")
    kg.add("TakeOrder", "hasSubtask", "Navigate")

    kg.add("spilledDrink", "causes", "floorWet")
    kg.add("floorWet", "causes", "floorSlippery")
    kg.add("floorSlippery", "causes", "hazardForHumans")

    kg.add("tiago", "isA", "Agent")
    kg.add("tiago", "performs", "ServeItem")
    kg.add("tiago", "performs", "TakeOrder")
    kg.add("tiago", "performs", "IndicatePlace")

    kg.infer()
    return kg
