from autodistill.detection import DetectionTargetModel
import supervision as sv
from ultralytics import YOLO
import torch

class YOLOv8(DetectionTargetModel):
    def __init__(self, model_name):
        self.yolo = YOLO(model_name)

    def predict(self, input:str, confidence=0.5) -> sv.Detections:
        return self.yolo(input, conf=confidence)

    def train(self, dataset_yaml, epochs=300, device="cpu"):
        self.yolo.train(data=dataset_yaml, epochs=epochs, device=device)
