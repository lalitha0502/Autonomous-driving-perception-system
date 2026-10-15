from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    input_path: str
    output_path: str
    classes: list[str]
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    device: str
    tracking_enabled: bool
    tracker_config: str
    metrics_overlay: bool
    pipeline_mode: str
    queue_size: int
    display: bool
    save_output: bool
    codec: str
    window_name: str


def load_config(config_path: str) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return AppConfig(
        model_path=data.get("model_path", "yolov8n.pt"),
        input_path=data.get("input_path", ""),
        output_path=data.get("output_path", "outputs/detections.mp4"),
        classes=list(data.get("classes", [])),
        confidence_threshold=float(data.get("confidence_threshold", 0.3)),
        iou_threshold=float(data.get("iou_threshold", 0.45)),
        image_size=int(data.get("image_size", 640)),
        device=data.get("device", "cpu"),
        tracking_enabled=bool(data.get("tracking_enabled", True)),
        tracker_config=data.get("tracker_config", "bytetrack.yaml"),
        metrics_overlay=bool(data.get("metrics_overlay", True)),
        pipeline_mode=data.get("pipeline_mode", "sync"),
        queue_size=int(data.get("queue_size", 8)),
        display=bool(data.get("display", False)),
        save_output=bool(data.get("save_output", True)),
        codec=data.get("codec", "mp4v"),
        window_name=data.get("window_name", "Tracking Output"),
    )


def apply_overrides(config: AppConfig, overrides: dict[str, object | None]) -> AppConfig:
    data = asdict(config)
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return AppConfig(**data)
