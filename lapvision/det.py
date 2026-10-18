"""Detector abstraction: COCO YOLO (vehicle classes), open-vocabulary YOLO-World, or a trained student.

All return class names in {'truck', 'car', 'motorcycle', 'person'} style so downstream code is unchanged:
YOLO-World's "off-road buggy" is reported as 'buggy' (treated as a vehicle everywhere: cls != 'person').
A student (e.g. students/train_det.py, single class `car`, or `car` + `number`) keeps its own class names;
its vehicle classes are every class except 'person' and 'number', and 'number' boxes are never returned.

mode: None = auto ('world' in the file name -> world; a YOLO with the 80 COCO classes -> coco; otherwise student),
or force 'coco' / 'world' / 'student'.
"""
from ultralytics import YOLO, YOLOWorld

COCO_VEH = {2: "car", 3: "motorcycle", 7: "truck", 0: "person"}
WORLD_PROMPTS = ["off-road buggy", "person"]
NON_VEHICLE = {"person", "number"}


class Detector:
    def __init__(self, name="yolo11x.pt", mode=None):
        if mode is None and "world" in str(name):
            mode = "world"
        self.world = mode == "world"
        if self.world:
            self.model = YOLOWorld(name)
            self.model.set_classes(WORLD_PROMPTS)
            self.names = {0: "buggy", 1: "person"}
            self.classes = None
            self.veh = [0]
        else:
            self.model = YOLO(name)
            names = dict(self.model.names)
            if mode is None:
                mode = "coco" if len(names) == 80 and names.get(7) == "truck" else "student"
            if mode == "coco":
                self.names = COCO_VEH
                self.classes = list(COCO_VEH)
                self.veh = [2, 3, 7]
            else:
                self.names = names
                self.veh = [i for i, n in names.items() if n not in NON_VEHICLE]
                self.classes = self.veh + [i for i, n in names.items() if n == "person"]
        self.mode = mode

    def predict(self, frames, **kw):
        return self.model(frames, classes=self.classes, verbose=False, **kw)

    def track(self, frame, **kw):
        return self.model.track(frame, classes=self.veh, persist=True, tracker="bytetrack.yaml",
                                verbose=False, **kw)

    def reset(self):
        self.model.predictor = None
        if self.world:
            self.model.set_classes(WORLD_PROMPTS)
