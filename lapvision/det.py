"""Detector abstraction: COCO YOLO (vehicle classes) or open-vocabulary YOLO-World.

Both return class names in {'truck', 'car', 'motorcycle', 'person'} style so downstream code is unchanged:
YOLO-World's "off-road buggy" is reported as 'buggy' (treated as a vehicle everywhere: cls != 'person').
"""
from ultralytics import YOLO, YOLOWorld

COCO_VEH = {2: "car", 3: "motorcycle", 7: "truck", 0: "person"}
WORLD_PROMPTS = ["off-road buggy", "person"]


class Detector:
    def __init__(self, name="yolo11x.pt"):
        self.world = "world" in name
        if self.world:
            self.model = YOLOWorld(name)
            self.model.set_classes(WORLD_PROMPTS)
            self.names = {0: "buggy", 1: "person"}
            self.classes = None
            self.veh = [0]
        else:
            self.model = YOLO(name)
            self.names = COCO_VEH
            self.classes = list(COCO_VEH)
            self.veh = [2, 3, 7]

    def predict(self, frames, **kw):
        return self.model(frames, classes=self.classes, verbose=False, **kw)

    def track(self, frame, **kw):
        return self.model.track(frame, classes=self.veh, persist=True, tracker="bytetrack.yaml",
                                verbose=False, **kw)

    def reset(self):
        self.model.predictor = None
        if self.world:
            self.model.set_classes(WORLD_PROMPTS)
