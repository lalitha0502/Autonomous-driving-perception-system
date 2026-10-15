from __future__ import annotations

import cv2

from src.detection.yolo_detector import Detection


class DetectionAnnotator:
    def __init__(self) -> None:
        self.palette = {
            "person": (0, 255, 255),
            "bicycle": (255, 255, 0),
            "car": (0, 255, 0),
            "motorcycle": (255, 0, 0),
            "bus": (0, 165, 255),
            "truck": (255, 0, 255),
        }

    def draw(self, frame, detections: list[Detection], metrics: dict[str, float | int] | None = None):
        annotated = frame.copy()
        for detection in detections:
            color = self._resolve_color(detection)
            cv2.rectangle(
                annotated,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                color,
                2,
            )

            label = self._build_label(detection)
            text_origin = (detection.x1, max(detection.y1 - 10, 20))
            cv2.putText(
                annotated,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        if metrics is not None:
            self._draw_metrics_panel(annotated, metrics)

        return annotated

    def _build_label(self, detection: Detection) -> str:
        if detection.track_id is None:
            return f"{detection.class_name} {detection.confidence:.2f}"
        return f"ID {detection.track_id} {detection.class_name} {detection.confidence:.2f}"

    def _resolve_color(self, detection: Detection) -> tuple[int, int, int]:
        if detection.track_id is None:
            return self.palette.get(detection.class_name, (255, 255, 255))

        base_color = self.palette.get(detection.class_name, (255, 255, 255))
        shift = detection.track_id * 29
        return (
            (base_color[0] + shift) % 256,
            (base_color[1] + shift * 2) % 256,
            (base_color[2] + shift * 3) % 256,
        )

    def _draw_metrics_panel(self, frame, metrics: dict[str, float | int]) -> None:
        lines = [
            f"FPS: {float(metrics['fps']):.2f}",
            f"Detection: {float(metrics['detection_latency_ms']):.1f} ms",
            f"Tracking: {float(metrics['tracking_latency_ms']):.1f} ms",
            f"E2E: {float(metrics['end_to_end_latency_ms']):.1f} ms",
            f"Tracked: {int(metrics['tracked_objects'])}",
        ]

        panel_width = 270
        panel_height = 28 + (len(lines) * 24)
        cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), (25, 25, 25), -1)
        cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), (180, 180, 180), 1)

        for index, line in enumerate(lines):
            y = 35 + (index * 24)
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
