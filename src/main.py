from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.yolo_detector import YOLODetector
from src.pipeline.detection_pipeline import VideoDetectionPipeline
from src.utils.config import AppConfig, apply_overrides, load_config
from src.visualization.annotator import DetectionAnnotator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the detection and tracking video pipeline with YOLOv8."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to an input video file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Path to the annotated output video.",
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        help="Path or model name for the YOLOv8 weights.",
    )
    parser.add_argument(
        "--conf",
        dest="confidence_threshold",
        type=float,
        help="Minimum confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        dest="iou_threshold",
        type=float,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--imgsz",
        dest="image_size",
        type=int,
        help="Inference image size passed to YOLOv8.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        help="Inference device, such as cpu, mps, or cuda:0.",
    )
    parser.add_argument(
        "--tracker",
        dest="tracker_config",
        help="Tracker config to use, such as bytetrack.yaml or botsort.yaml.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable multi-object tracking.",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable tracking and run detection only.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames while processing.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("sync", "async"),
        help="Choose between the synchronous baseline and async queue-based pipeline.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        help="Maximum number of frames buffered between async stages.",
    )
    parser.add_argument(
        "--hide-metrics",
        action="store_true",
        help="Disable on-frame metrics overlay.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable output video writing.",
    )
    return parser


def validate_config(config: AppConfig) -> None:
    if not config.input_path:
        raise ValueError(
            "No input video was provided. Pass --input or update configs/default.yaml."
        )

    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(
        config,
        {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "model_path": args.model_path,
            "confidence_threshold": args.confidence_threshold,
            "iou_threshold": args.iou_threshold,
            "image_size": args.image_size,
            "device": args.device,
            "tracker_config": args.tracker_config,
            "tracking_enabled": True if args.track else False if args.no_track else None,
            "metrics_overlay": False if args.hide_metrics else None,
            "pipeline_mode": args.pipeline_mode,
            "queue_size": args.queue_size,
            "display": True if args.show else None,
            "save_output": False if args.no_save else None,
        },
    )

    validate_config(config)

    detector = YOLODetector(config)
    annotator = DetectionAnnotator()
    pipeline = VideoDetectionPipeline(config=config, detector=detector, annotator=annotator)
    summary = pipeline.run()

    print("Finished processing video.")
    print(f"Pipeline mode: {summary['pipeline_mode']}")
    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Elapsed time (s): {summary['elapsed_seconds']:.2f}")
    print(f"Effective FPS: {summary['fps']:.2f}")
    print(f"Average detection latency (ms): {summary['avg_detection_latency_ms']:.2f}")
    print(f"Average tracking latency (ms): {summary['avg_tracking_latency_ms']:.2f}")
    print(f"Average end-to-end latency (ms): {summary['avg_end_to_end_latency_ms']:.2f}")
    print(f"Peak tracked objects: {summary['max_tracked_objects']}")
    if summary["output_path"]:
        print(f"Output saved to: {summary['output_path']}")


if __name__ == "__main__":
    main()
