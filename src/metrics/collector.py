from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
import threading


@dataclass(frozen=True)
class FrameMetrics:
    frame_index: int
    detection_latency_ms: float
    tracking_latency_ms: float
    annotation_latency_ms: float
    write_latency_ms: float
    end_to_end_latency_ms: float
    tracked_objects: int


class MetricsCollector:
    def __init__(self, rolling_window: int = 30) -> None:
        self.frames: list[FrameMetrics] = []
        self._recent_frame_times_ms: deque[float] = deque(maxlen=rolling_window)
        self._lock = threading.Lock()

    def add(self, frame_metrics: FrameMetrics) -> None:
        with self._lock:
            self.frames.append(frame_metrics)
            self._recent_frame_times_ms.append(frame_metrics.end_to_end_latency_ms)

    def live_snapshot(self) -> dict[str, float | int]:
        with self._lock:
            if not self.frames:
                return {
                    "fps": 0.0,
                    "detection_latency_ms": 0.0,
                    "tracking_latency_ms": 0.0,
                    "end_to_end_latency_ms": 0.0,
                    "tracked_objects": 0,
                }

            latest = self.frames[-1]
            rolling_average_ms = mean(self._recent_frame_times_ms) if self._recent_frame_times_ms else 0.0
            fps = 1000.0 / rolling_average_ms if rolling_average_ms > 0 else 0.0

            return {
                "fps": fps,
                "detection_latency_ms": latest.detection_latency_ms,
                "tracking_latency_ms": latest.tracking_latency_ms,
                "end_to_end_latency_ms": latest.end_to_end_latency_ms,
                "tracked_objects": latest.tracked_objects,
            }

    def summary(
        self,
        *,
        elapsed_seconds: float,
        output_path: str | None,
        pipeline_mode: str,
    ) -> dict[str, float | int | str | None]:
        with self._lock:
            if not self.frames:
                return {
                    "pipeline_mode": pipeline_mode,
                    "frames_processed": 0,
                    "elapsed_seconds": elapsed_seconds,
                    "fps": 0.0,
                    "avg_detection_latency_ms": 0.0,
                    "avg_tracking_latency_ms": 0.0,
                    "avg_end_to_end_latency_ms": 0.0,
                    "avg_tracked_objects": 0.0,
                    "max_tracked_objects": 0,
                    "output_path": output_path,
                }

            frames_processed = len(self.frames)
            return {
                "pipeline_mode": pipeline_mode,
                "frames_processed": frames_processed,
                "elapsed_seconds": elapsed_seconds,
                "fps": frames_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                "avg_detection_latency_ms": mean(frame.detection_latency_ms for frame in self.frames),
                "avg_tracking_latency_ms": mean(frame.tracking_latency_ms for frame in self.frames),
                "avg_end_to_end_latency_ms": mean(frame.end_to_end_latency_ms for frame in self.frames),
                "avg_tracked_objects": mean(frame.tracked_objects for frame in self.frames),
                "max_tracked_objects": max(frame.tracked_objects for frame in self.frames),
                "output_path": output_path,
            }
