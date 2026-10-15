from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

import cv2

from src.detection.yolo_detector import DetectionBatch, YOLODetector
from src.metrics.collector import FrameMetrics, MetricsCollector
from src.utils.config import AppConfig
from src.visualization.annotator import DetectionAnnotator

_STOP = object()


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    frame_started_at: float
    frame: object


@dataclass(frozen=True)
class InferencePacket:
    frame_index: int
    frame_started_at: float
    frame: object
    detection_batch: DetectionBatch
    tracked_objects: int


@dataclass(frozen=True)
class AnnotatedPacket:
    frame_index: int
    frame_started_at: float
    annotated_frame: object
    detection_latency_ms: float
    tracking_latency_ms: float
    annotation_latency_ms: float
    tracked_objects: int


class VideoDetectionPipeline:
    def __init__(
        self,
        config: AppConfig,
        detector: YOLODetector,
        annotator: DetectionAnnotator,
    ) -> None:
        self.config = config
        self.detector = detector
        self.annotator = annotator

    def run(self) -> dict[str, float | int | str | None]:
        capture = cv2.VideoCapture(self.config.input_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {self.config.input_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = self._build_writer(fps, frame_width, frame_height)

        try:
            if self.config.pipeline_mode == "async":
                return self._run_async(capture, writer)
            return self._run_sync(capture, writer)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if self.config.display:
                cv2.destroyAllWindows()

    def _run_sync(
        self,
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter | None,
    ) -> dict[str, float | int | str | None]:
        metrics = MetricsCollector()
        started_at = time.perf_counter()
        frame_index = 0

        while True:
            frame_started_at = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break

            detection_batch = self._run_detection(frame)
            tracked_objects = sum(
                detection.track_id is not None for detection in detection_batch.detections
            )

            annotation_started_at = time.perf_counter()
            live_metrics = self._build_live_metrics(metrics, detection_batch, tracked_objects)
            annotated_frame = self.annotator.draw(
                frame,
                detection_batch.detections,
                metrics=live_metrics if self.config.metrics_overlay else None,
            )
            annotation_latency_ms = (time.perf_counter() - annotation_started_at) * 1000.0

            write_started_at = time.perf_counter()
            self._emit_frame(writer, annotated_frame)
            write_latency_ms = (time.perf_counter() - write_started_at) * 1000.0

            if self._display_frame(annotated_frame):
                break

            end_to_end_latency_ms = (time.perf_counter() - frame_started_at) * 1000.0
            metrics.add(
                FrameMetrics(
                    frame_index=frame_index,
                    detection_latency_ms=detection_batch.detection_latency_ms,
                    tracking_latency_ms=detection_batch.tracking_latency_ms,
                    annotation_latency_ms=annotation_latency_ms,
                    write_latency_ms=write_latency_ms,
                    end_to_end_latency_ms=end_to_end_latency_ms,
                    tracked_objects=tracked_objects,
                )
            )
            frame_index += 1

        elapsed_seconds = time.perf_counter() - started_at
        return metrics.summary(
            elapsed_seconds=elapsed_seconds,
            output_path=self.config.output_path if self.config.save_output else None,
            pipeline_mode="sync",
        )

    def _run_async(
        self,
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter | None,
    ) -> dict[str, float | int | str | None]:
        metrics = MetricsCollector()
        started_at = time.perf_counter()
        stop_event = threading.Event()
        exceptions: list[BaseException] = []

        raw_queue: queue.Queue[FramePacket | object] = queue.Queue(maxsize=self.config.queue_size)
        inference_queue: queue.Queue[InferencePacket | object] = queue.Queue(maxsize=self.config.queue_size)
        annotated_queue: queue.Queue[AnnotatedPacket | object] = queue.Queue(maxsize=self.config.queue_size)

        def record_exception(exc: BaseException) -> None:
            if not exceptions:
                exceptions.append(exc)
            stop_event.set()

        def reader_worker() -> None:
            frame_index = 0
            try:
                while not stop_event.is_set():
                    frame_started_at = time.perf_counter()
                    ok, frame = capture.read()
                    if not ok:
                        break

                    packet = FramePacket(
                        frame_index=frame_index,
                        frame_started_at=frame_started_at,
                        frame=frame,
                    )
                    if not self._queue_put(raw_queue, packet, stop_event):
                        break
                    frame_index += 1
            except BaseException as exc:
                record_exception(exc)
            finally:
                self._signal_stop(raw_queue, stop_event)

        def inference_worker() -> None:
            try:
                while True:
                    item = self._queue_get(raw_queue, stop_event)
                    if item is _STOP:
                        break

                    packet = item
                    detection_batch = self._run_detection(packet.frame)
                    tracked_objects = sum(
                        detection.track_id is not None for detection in detection_batch.detections
                    )
                    output_packet = InferencePacket(
                        frame_index=packet.frame_index,
                        frame_started_at=packet.frame_started_at,
                        frame=packet.frame,
                        detection_batch=detection_batch,
                        tracked_objects=tracked_objects,
                    )
                    if not self._queue_put(inference_queue, output_packet, stop_event):
                        break
            except BaseException as exc:
                record_exception(exc)
            finally:
                self._signal_stop(inference_queue, stop_event)

        def annotation_worker() -> None:
            try:
                while True:
                    item = self._queue_get(inference_queue, stop_event)
                    if item is _STOP:
                        break

                    packet = item
                    annotation_started_at = time.perf_counter()
                    live_metrics = self._build_live_metrics(
                        metrics,
                        packet.detection_batch,
                        packet.tracked_objects,
                    )
                    annotated_frame = self.annotator.draw(
                        packet.frame,
                        packet.detection_batch.detections,
                        metrics=live_metrics if self.config.metrics_overlay else None,
                    )
                    annotation_latency_ms = (time.perf_counter() - annotation_started_at) * 1000.0

                    output_packet = AnnotatedPacket(
                        frame_index=packet.frame_index,
                        frame_started_at=packet.frame_started_at,
                        annotated_frame=annotated_frame,
                        detection_latency_ms=packet.detection_batch.detection_latency_ms,
                        tracking_latency_ms=packet.detection_batch.tracking_latency_ms,
                        annotation_latency_ms=annotation_latency_ms,
                        tracked_objects=packet.tracked_objects,
                    )
                    if not self._queue_put(annotated_queue, output_packet, stop_event):
                        break
            except BaseException as exc:
                record_exception(exc)
            finally:
                self._signal_stop(annotated_queue, stop_event)

        def writer_worker() -> None:
            try:
                while True:
                    item = self._queue_get(annotated_queue, stop_event)
                    if item is _STOP:
                        break

                    packet = item
                    write_started_at = time.perf_counter()
                    self._emit_frame(writer, packet.annotated_frame)
                    write_latency_ms = (time.perf_counter() - write_started_at) * 1000.0

                    if self._display_frame(packet.annotated_frame):
                        stop_event.set()

                    end_to_end_latency_ms = (time.perf_counter() - packet.frame_started_at) * 1000.0
                    metrics.add(
                        FrameMetrics(
                            frame_index=packet.frame_index,
                            detection_latency_ms=packet.detection_latency_ms,
                            tracking_latency_ms=packet.tracking_latency_ms,
                            annotation_latency_ms=packet.annotation_latency_ms,
                            write_latency_ms=write_latency_ms,
                            end_to_end_latency_ms=end_to_end_latency_ms,
                            tracked_objects=packet.tracked_objects,
                        )
                    )

                    if stop_event.is_set():
                        break
            except BaseException as exc:
                record_exception(exc)

        threads = [
            threading.Thread(target=reader_worker, name="frame-reader"),
            threading.Thread(target=inference_worker, name="inference-worker"),
            threading.Thread(target=annotation_worker, name="annotation-worker"),
            threading.Thread(target=writer_worker, name="writer-worker"),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        if exceptions:
            raise exceptions[0]

        elapsed_seconds = time.perf_counter() - started_at
        return metrics.summary(
            elapsed_seconds=elapsed_seconds,
            output_path=self.config.output_path if self.config.save_output else None,
            pipeline_mode="async",
        )

    def _run_detection(self, frame) -> DetectionBatch:
        if self.config.tracking_enabled:
            return self.detector.track(frame)
        return self.detector.infer(frame)

    def _build_live_metrics(
        self,
        metrics: MetricsCollector,
        detection_batch: DetectionBatch,
        tracked_objects: int,
    ) -> dict[str, float | int]:
        return {
            **metrics.live_snapshot(),
            "detection_latency_ms": detection_batch.detection_latency_ms,
            "tracking_latency_ms": detection_batch.tracking_latency_ms,
            "tracked_objects": tracked_objects,
        }

    def _emit_frame(self, writer: cv2.VideoWriter | None, frame) -> None:
        if writer is not None:
            writer.write(frame)

    def _display_frame(self, frame) -> bool:
        if not self.config.display:
            return False

        cv2.imshow(self.config.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def _build_writer(
        self,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> cv2.VideoWriter | None:
        if not self.config.save_output:
            return None

        if not self.config.output_path:
            raise ValueError("save_output is enabled but no output_path was provided.")

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*self.config.codec),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")
        return writer

    def _queue_get(self, work_queue: queue.Queue, stop_event: threading.Event):
        while True:
            try:
                return work_queue.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    return _STOP

    def _queue_put(
        self,
        work_queue: queue.Queue,
        item,
        stop_event: threading.Event,
    ) -> bool:
        while True:
            if stop_event.is_set():
                return False
            try:
                work_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue

    def _signal_stop(self, work_queue: queue.Queue, stop_event: threading.Event) -> None:
        while True:
            try:
                work_queue.put(_STOP, timeout=0.1)
                return
            except queue.Full:
                if stop_event.is_set():
                    try:
                        work_queue.get_nowait()
                    except queue.Empty:
                        continue
