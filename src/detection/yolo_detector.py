from __future__ import annotations

from dataclasses import dataclass
import time

from ultralytics import YOLO

from src.utils.config import AppConfig


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None


@dataclass(frozen=True)
class DetectionBatch:
    detections: list[Detection]
    detection_latency_ms: float
    tracking_latency_ms: float
    model_latency_ms: float


class YOLODetector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.allowed_classes = set(config.classes)
        self.model = YOLO(config.model_path)

    def infer(self, frame) -> DetectionBatch:
        started_at = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )
        wall_clock_ms = (time.perf_counter() - started_at) * 1000.0
        return self._parse_results(results[0], wall_clock_ms=wall_clock_ms, tracking_enabled=False)

    def track(self, frame) -> DetectionBatch:
        started_at = time.perf_counter()
        results = self.model.track(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            tracker=self.config.tracker_config,
            persist=True,
            verbose=False,
        )
        wall_clock_ms = (time.perf_counter() - started_at) * 1000.0
        return self._parse_results(results[0], wall_clock_ms=wall_clock_ms, tracking_enabled=True)

    def _parse_results(
        self,
        result,
        *,
        wall_clock_ms: float,
        tracking_enabled: bool,
    ) -> DetectionBatch:
        boxes = result.boxes
        speed = result.speed or {}
        detection_latency_ms = float(speed.get("inference", 0.0))
        tracking_latency_ms = float(speed.get("postprocess", 0.0)) if tracking_enabled else 0.0

        if boxes is None or len(boxes) == 0:
            return DetectionBatch(
                detections=[],
                detection_latency_ms=detection_latency_ms,
                tracking_latency_ms=tracking_latency_ms,
                model_latency_ms=wall_clock_ms,
            )

        detections: list[Detection] = []
        names = result.names

        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

        for coords, confidence, class_id, track_id in zip(
            xyxy,
            confidences,
            class_ids,
            track_ids,
        ):
            class_index = int(class_id)
            class_name = names[class_index]
            if class_name not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = (int(value) for value in coords)
            detections.append(
                Detection(
                    class_id=class_index,
                    class_name=class_name,
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    track_id=int(track_id) if track_id is not None else None,
                )
            )

        return DetectionBatch(
            detections=detections,
            detection_latency_ms=detection_latency_ms,
            tracking_latency_ms=tracking_latency_ms,
            model_latency_ms=wall_clock_ms,
        )
