# Autonomous Driving Perception & Tracking System Plan

## Objective

Build a real-time computer vision pipeline for dashcam video that:
- Detects road users reliably
- Tracks objects with stable identities across frames
- Measures latency and throughput
- Runs efficiently on CPU with low enough latency for interactive playback

## Tech Stack

- Python
- PyTorch
- OpenCV
- YOLOv8
- Multi-object tracker: ByteTrack first, with SORT as a simpler fallback

## Target Outcome

By the end of the project, the system should be able to:
- Read dashcam video from file or webcam-like stream
- Detect `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`
- Assign persistent track IDs across frames
- Render annotated output video
- Report detection latency, tracking latency, end-to-end latency, FPS, and active track count
- Support a baseline synchronous mode and an optimized asynchronous mode

## Success Criteria

- Stable detections on sample road footage
- Track IDs remain consistent for common short-term motion
- Measured FPS is reported continuously and in a final summary
- Pipeline runs end-to-end on CPU without crashing on a representative test video
- Code is modular enough to swap detector settings, tracker settings, and input sources

## Assumptions

- Initial development will use prerecorded dashcam video rather than a live vehicle feed
- CPU execution is the primary target; GPU support is optional
- Real-time means practical interactive throughput on a normal laptop, not production ADAS-grade latency
- A small set of sample videos will be enough for baseline validation

## Proposed Repository Structure

```text
autonomous-driving-perception-tracking-system/
├── README.md
├── PLAN.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   └── samples/
├── outputs/
├── src/
│   ├── main.py
│   ├── pipeline/
│   ├── detection/
│   ├── tracking/
│   ├── metrics/
│   ├── visualization/
│   └── utils/
└── tests/
```

## Phase Plan

### Phase 0: Project Setup and Baseline

**Goal:** Create a clean starting point for repeatable development.

**Tasks**
- Initialize project structure
- Add dependency list and installation instructions
- Define a simple config file for model path, classes, confidence threshold, and input source
- Add a sample CLI entrypoint for running the pipeline
- Choose one short dashcam clip for baseline testing

**Deliverables**
- `requirements.txt`
- `src/` package structure
- Config file for runtime parameters
- Basic command to run the project

**Exit Criteria**
- Project installs cleanly in a virtual environment
- App starts from the command line and loads configuration successfully

### Phase 1: Detection Pipeline

**Goal:** Run YOLOv8 on video input and visualize detections frame by frame.

**Tasks**
- Load YOLOv8n model
- Filter detections to target road-user classes
- Read frames from video input using OpenCV
- Draw bounding boxes, class names, and confidence scores
- Save or display processed video

**Deliverables**
- Working frame reader
- Detector wrapper around YOLOv8 inference
- Visualization of detections on output frames

**Exit Criteria**
- System processes a sample video from start to finish
- Output shows correct classes and bounding boxes
- Detection-only mode runs without tracking enabled

### Phase 2: Multi-Object Tracking

**Goal:** Add stable object identities across frames.

**Tasks**
- Integrate a tracker, preferably ByteTrack
- Convert detector outputs into tracker input format
- Associate detections to persistent track IDs
- Render track IDs on the output frames
- Handle track creation, updates, and expiration

**Deliverables**
- Tracker integration module
- Detection-to-tracker adapter
- Visualized track IDs in the output stream

**Exit Criteria**
- Moving objects retain IDs for most consecutive visible frames
- Track IDs are displayed correctly in output video
- Tracking module can be toggled on and off

### Phase 3: Metrics and Instrumentation

**Goal:** Measure system performance with enough detail to guide optimization.

**Tasks**
- Time detection latency per frame
- Time tracking latency per frame
- Time total pipeline latency per frame
- Compute rolling and aggregate FPS
- Track active object count over time
- Print final summary metrics after video completion

**Deliverables**
- Metrics collector module
- On-screen or console reporting of live FPS
- End-of-run performance summary

**Exit Criteria**
- Each frame records timing data
- Final report includes average latency, average FPS, total runtime, and object count stats

### Phase 4: Asynchronous Pipeline

**Goal:** Reduce idle time by decoupling video I/O, inference, and output stages.

**Tasks**
- Split the pipeline into frame reader, inference worker, tracking/annotation worker, and output writer
- Introduce queues or bounded buffers between stages
- Preserve frame ordering in final output
- Add graceful shutdown behavior when input ends or exceptions occur
- Compare synchronous and asynchronous throughput

**Deliverables**
- Async pipeline controller
- Queue-based stage communication
- Benchmark comparison against the synchronous baseline

**Exit Criteria**
- Pipeline stages run independently without deadlocks
- Output remains frame-aligned and valid
- Async mode improves throughput or responsiveness over the synchronous baseline

### Phase 5: CPU Optimization

**Goal:** Improve throughput enough for practical real-time use on CPU.

**Tasks**
- Benchmark multiple input resolutions
- Tune confidence and NMS settings if needed
- Add frame skipping under load
- Limit queue sizes to reduce latency buildup
- Evaluate lightweight tracker settings
- Profile bottlenecks in detection, tracking, visualization, and disk I/O

**Deliverables**
- Configurable optimization settings
- Benchmark table for different runtime settings
- Best-known CPU profile for demo use

**Exit Criteria**
- Documented optimized configuration exists
- Measurable improvement over the unoptimized baseline is shown

### Phase 6: Validation and Demo Packaging

**Goal:** Make the project presentable, reproducible, and easy to evaluate.

**Tasks**
- Add example commands for running different modes
- Save sample output videos or screenshots
- Document performance results from at least one reference machine
- Add lightweight tests for config loading, frame processing, and tracker integration boundaries
- Update README with architecture, usage, and results

**Deliverables**
- Finalized README
- Example assets in `outputs/`
- Basic test coverage for critical modules

**Exit Criteria**
- Another developer can run the project from setup docs
- Demo output and benchmark numbers are included in the repo

## Recommended Implementation Order

1. Build the simplest synchronous detector first.
2. Add tracking only after detection output is stable.
3. Add metrics before optimization so improvements are measurable.
4. Introduce asynchronous execution only after the synchronous pipeline is correct.
5. Optimize last, based on actual profiling data rather than guesswork.

## Key Risks

- CPU-only YOLO inference may be slower than expected on high-resolution video
- Tracker quality may degrade with missed detections or occlusions
- Async design can increase complexity, buffering delay, and debugging cost
- Visualization and video writing can become hidden bottlenecks

## Mitigations

- Start with YOLOv8n and lower input resolution
- Use short sample clips for fast iteration
- Keep synchronous and async modes both available
- Log per-stage timings to isolate bottlenecks quickly
- Make tracker and detector settings configurable from the start

## Stretch Goals

- ONNX or OpenVINO deployment
- Lane detection overlay
- Collision-risk heuristics
- Side-by-side comparison of SORT, DeepSORT, and ByteTrack
- Night and rain condition evaluation

## Definition of Done

The project is complete when:
- The full pipeline runs from video input to annotated output
- Detections and track IDs are visible and reasonably stable
- Performance metrics are reported live and at the end of execution
- Async mode is implemented and benchmarked
- CPU optimization results are documented
- The repo is polished enough to show as a portfolio project
