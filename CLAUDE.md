# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

门架式移动平台AI检测服务 (YDPT Service) - A Flask-based real-time object detection and workflow step validation service for scaffold assembly procedures. Uses YOLOv8 (CUDA/CPU) or Ascend NPU for inference, with RTSP video stream processing and multi-frame confirmation logic.

## Running the Service

```bash
# Start the service (default port 5010)
python run_app.py

# The service reads configuration from config.yaml on startup
```

The service auto-detects video source FPS and initializes the inference backend based on `config.yaml` settings.

## Configuration

Configuration is managed via `config.yaml` with the following key sections:
- `inference.backend`: "yolov8" (CUDA/CPU) or "ascend" (NPU)
- `inference.device_id`: "cuda:0", "cpu", or NPU device number
- `video.rtsp_url`: RTSP stream URL or local video file path
- `service.port`: Flask service port (default 5010)
- `detection.recog_area`: Detection region [x1, y1, x2, y2] based on 1920x1080 resolution

Environment variables can override config values (e.g., `INFERENCE_BACKEND`, `DEVICE_ID`, `SERVICE_PORT`).

## Key Dependencies

- Flask, OpenCV (cv2), numpy, requests, PyYAML
- ultralytics (YOLOv8) for CUDA/CPU inference
- ais_bench (Ascend) for NPU inference - optional, only on Ascend servers
- setproctitle for process naming
- psutil for CPU affinity (optional)

## Architecture

### Core Files

- `run_app.py` (~2300 lines): Main Flask application with video processing, step validation, and API endpoints
- `inference_engine.py`: Inference engine abstraction with CUDA and Ascend backends
- `config_loader.py`: YAML configuration loader with environment variable override support
- `export_onnx.py`: Utility to export YOLO models to ONNX format for Ascend conversion

### Key Components

**InferencePipeline**: Manages inference in a separate process (YOLOv8) or thread (Ascend) to bypass GIL. Uses frame/result queues for async communication.

**ObjectTracker**: IoU-based tracker assigning unique IDs to detected objects and tracking their stability across frames (`stable_count`).

**StepValidator**: Validates 14 workflow steps for scaffold assembly procedures. Uses multi-frame confirmation (`min_stable_frames`) to avoid false triggers. Some steps require spatial relations (e.g., casters attached to gantries) or zone checks.

**GlobalState**: Thread-safe singleton managing all runtime state including recording status, detection buffer, step history, and video writer.

### Detection Classes (LABEL_MAP)

0: 安全帽, 1: 安全带, 2: 围栏, 3: 标识牌, 4: 手套,
5: 滑动脚轮, 6: 门架, 7: 交叉杆, 8: 脚手板, 9: 爬梯,
10: 防护栏, 11: 安全带挂钩, 12: 扫把, 13: 人

### API Endpoints

- `POST /start`: Start exam session with `{"userid": "xxx"}`
- `POST /stop`: End exam session, returns video path and step results
- `GET /ydpt_boxes`: Get current frame detections
- `GET /ydpt_sseboxes`: SSE stream for real-time detection updates
- `GET /health`: Health check endpoint
- `GET /config`: Zone configuration UI page
- `POST /save_zones`: Save detection zone configuration

## Development Notes

### Resolution Adaptation

Detection region (`recog_area`) is configured for 1920x1080 and auto-scaled to actual video resolution. Zone coordinates in `config/zones_config.json` follow the same convention.

### FFmpeg Video Recording

The service uses OpenCV with FFmpeg backend. Key fixes for PTS/timestamp issues:
- `OPENCV_FFMPEG_WRITE_NO_AUTO_PTS=1` environment variable
- Frame validity checking before write
- Consistent frame rate from video source

### Step Validation Logic

Steps 2 and 4 pass unconditionally (`always_pass: True`). Steps 12 and 14 (disassembly/cleanup) use state change detection - confirming objects are removed from the recognition area for N consecutive frames.
