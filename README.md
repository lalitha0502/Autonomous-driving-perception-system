# Autonomous Driving Perception & Tracking System

**Tech Stack:** Python, PyTorch, OpenCV, YOLOv8

## Overview
Build a real-time computer vision pipeline for dashcam footage that detects road users and tracks them across frames.

The system should:
- Detect vehicles and pedestrians
- Maintain stable IDs over time
- Process video with low latency
- Measure FPS and end-to-end runtime performance

## Core Pipeline

```text
Video Input
→ Frame Reader
→ YOLOv8 Detector
→ Multi-Object Tracker
→ Visualization / Output
→ Metrics
```

## Main Components

### 1. Object Detection
Use YOLOv8 to detect:
- person
- bicycle
- car
- motorcycle
- bus
- truck

Use a lightweight model (such as YOLOv8n) for faster real-time inference.

### 2. Multi-Object Tracking
Assign persistent IDs to detected objects across frames.

Common tracker choices:
- SORT
- DeepSORT
- ByteTrack

### 3. Real-Time Processing
Use an asynchronous pipeline so frame reading, inference, and output run independently.

Benefits:
- Lower latency
- Better throughput
- Reduced idle time

### 4. Optimization
Improve CPU performance using:
- Smaller model sizes
- Lower input resolution
- Frame buffering
- Limited batching
- Frame skipping when overloaded

## Metrics to Measure

Track the following:

- Detection latency (ms)
- Tracking latency (ms)
- End-to-end latency (ms)
- FPS
- Number of tracked objects

## Build Plan

### Phase 1
Detection on video input.

### Phase 2
Add tracking with stable IDs.

### Phase 3
Add performance metrics.

### Phase 4
Convert to asynchronous pipeline.

### Phase 5
Optimize for real-time CPU execution.

## Resume Version

**Autonomous Driving Perception & Tracking System**  
- Built a real-time perception system using YOLOv8 to detect vehicles and pedestrians from dashcam streams.  
- Added multi-object tracking to preserve identities across frames.  
- Designed an asynchronous pipeline for low-latency execution.  
- Optimized CPU throughput and benchmarked FPS / latency.  

## Nice Future Improvements

- ONNX / OpenVINO deployment
- Lane detection
- Collision warning logic
- Night / weather testing
- Tracker comparisons

## Summary

This project demonstrates:
- Computer vision
- Real-time systems engineering
- Performance optimization
- Applied machine learning

## Current Status

The repository now includes the first working perception loop:
- YOLOv8 video inference
- ByteTrack-based multi-object tracking
- Filtering for road-user classes
- Bounding boxes with stable track IDs
- Live FPS and latency reporting
- Queue-based async pipeline mode
- Annotated output video export
- Simple CLI and YAML config flow

## Project Structure

```text
.
├── configs/default.yaml
├── data/samples/
├── outputs/
├── src/
│   ├── detection/
│   ├── pipeline/
│   ├── utils/
│   ├── visualization/
│   └── main.py
├── PLAN.md
├── README.md
└── requirements.txt
```

## Setup

Use Python 3.10 for the cleanest compatibility with `ultralytics`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Pipeline

Update `configs/default.yaml` with an input video path, or pass one at runtime:

```bash
python -m src.main --input path/to/dashcam.mp4
```

Disable tracking and run detection-only mode:

```bash
python -m src.main --input path/to/dashcam.mp4 --no-track
```

Run the async pipeline:

```bash
python -m src.main --input path/to/dashcam.mp4 --pipeline-mode async
```

Write to a custom output file:

```bash
python -m src.main --input path/to/dashcam.mp4 --output outputs/run.mp4
```

Display frames while processing:

```bash
python -m src.main --input path/to/dashcam.mp4 --show
```

Override the model or thresholds:

```bash
python -m src.main \
  --input path/to/dashcam.mp4 \
  --model yolov8n.pt \
  --tracker bytetrack.yaml \
  --pipeline-mode async \
  --queue-size 8 \
  --conf 0.35 \
  --iou 0.45 \
  --imgsz 640
```

## Tracking Notes

- Tracking is enabled by default through `ByteTrack`
- Use `--tracker botsort.yaml` to compare against a different built-in tracker
- Use `--no-track` if you want to fall back to plain detection mode

## Metrics

The pipeline now reports:
- Detection latency per frame
- Tracking latency per frame
- End-to-end latency per frame
- Effective FPS
- Number of tracked objects

For the built-in YOLOv8 + ByteTrack path, tracking latency is taken from the tracker/postprocess stage exposed by Ultralytics.

The output video includes an on-frame metrics panel by default. Disable it with:

```bash
python -m src.main --input path/to/dashcam.mp4 --hide-metrics
```

## Async Notes

- `--pipeline-mode sync` keeps the simple baseline path
- `--pipeline-mode async` splits frame reading, inference/tracking, annotation, and writing into separate stages
- `--queue-size` controls the maximum buffered frames between async stages
- Larger queues can improve throughput stability but usually increase end-to-end latency
