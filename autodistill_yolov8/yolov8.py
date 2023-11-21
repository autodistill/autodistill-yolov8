import numpy as np
import supervision as sv
from autodistill.detection import (CaptionOntology, DetectionBaseModel,
                                   DetectionTargetModel)
from ultralytics import YOLO


class YOLOv8Base(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, weights_path: str):
        self.ontology = ontology
        self.model = YOLO(weights_path)

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        model_results = self.model(input, conf=confidence)[0]

        inference_results = sv.Detections.from_ultralytics(model_results)

        model_classes_to_id = [
            class_item
            for class_item in self.model.names
            if self.model.names[class_item] in self.ontology.prompts()
        ]

        model_classes_to_id = np.unique(model_classes_to_id)

        inference_results = inference_results[
            np.isin(inference_results.class_id, model_classes_to_id)
        ]

        return inference_results


class YOLOv8(DetectionTargetModel):
    def __init__(self, model_name):
        self.yolo = YOLO(model_name)

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        return self.yolo(input, conf=confidence)

    def train(self, dataset_yaml, epochs=200, device="cpu"):
        self.yolo.train(data=dataset_yaml, epochs=epochs, device=device)
