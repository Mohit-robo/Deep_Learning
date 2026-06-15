# PaddleOCR on Triton — Indian ANPR Experiment

Serving a **PaddleOCR v6** text-recognition pipeline on **NVIDIA Triton Inference Server** (Python Backend) for Automatic Number Plate Recognition (ANPR) on Indian vehicle plates.

---

## Overview

This experiment migrates an in-process PaddleOCR pipeline to a production-grade inference server. The model runs inside Triton's Python Backend, exposes a simple `IMAGE → RECOGNIZED_TEXT` interface, and is called from a Python client that supports both single-image and concurrent batch inference with per-image timing.

```
┌─────────────────────┐      HTTP/REST       ┌────────────────────────────────┐
│  triton_client.py   │ ──────────────────▶  │  Triton Inference Server        │
│  (async_infer)      │ ◀──────────────────  │  Python Backend → model.py      │
└─────────────────────┘   RECOGNIZED_TEXT    │  PaddleOCR (onnxruntime)        │
                                             └────────────────────────────────┘
```

---

## Project Structure

```
OCR/
├── images/
│   └── Indian_number_plate/
│       └── test/
│           ├── images/          # Raw YOLO-format test images
│           ├── labels/          # YOLO bounding-box annotations
│           └── crops/           # Cropped plate images (generated)
│
├── model_repository/
│   └── paddle_ocr/
│       ├── config.pbtxt         # Triton model config (input/output tensors)
│       └── 1/
│           └── model.py         # TritonPythonModel — PaddleOCR inference logic
│
├── anpr.py                      # Baseline: run PaddleOCR directly (no Triton)
├── crop_plates.py               # Crop plates from YOLO-annotated images
├── triton_client.py             # Triton client: single + async batch inference
└── README.md
```

---

## Dataset

**Indian Number Plate** dataset in YOLO format (sourced from Roboflow).

```
images/Indian_number_plate/
└── test/
    ├── images/    640×640 frames from dashcam video clips
    └── labels/    YOLO-format bounding boxes (class cx cy w h)
```

Plate crops are extracted with `crop_plates.py`, which reads YOLO normalised coordinates and saves individual plate images to `test/crops/`.

---

## Setup

### 1. Prerequisites

| Dependency | Version |
|---|---|
| NVIDIA Triton Inference Server | `nvcr.io/nvidia/tritonserver:24.xx-py3` |
| tritonclient | `pip install tritonclient[http]` |
| PaddleOCR | `pip install paddleocr` |
| OpenCV | `pip install opencv-python` |
| Docker | any recent version |

### 2. Pre-download PaddleOCR weights

The Triton container has no internet access by default. Download weights on the host first:

```bash
python - <<'EOF'
from paddleocr import PaddleOCR
PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    engine="onnxruntime",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
EOF
```

Locate the downloaded cache (typically `~/.paddlex/official_models/`) and note the path — you'll mount it into the container.

### 3. Launch Triton

```bash
docker run --rm -it \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  -v ~/.paddlex:/opt/paddlex \
  nvcr.io/nvidia/tritonserver:24.xx-py3 \
  tritonserver --model-repository=/models
```

> **Why `/opt/paddlex`?** `model.py` sets `PADDLEX_HOME=/opt/paddlex` and `PADDLEX_OFFLINE_MODE=1` so PaddleOCR loads cached weights instead of attempting a download.

### 4. Crop plates

```bash
python crop_plates.py
# → saves crops to images/Indian_number_plate/test/crops/
```

---

## Model Config (`config.pbtxt`)

```protobuf
name: "paddle_ocr"
backend: "python"
max_batch_size: 0          # variable-size images; batching handled client-side

input  [{ name: "IMAGE"           data_type: TYPE_UINT8   dims: [-1, -1, 3] }]
output [{ name: "RECOGNIZED_TEXT" data_type: TYPE_STRING  dims: [-1]        }]

instance_group [{ count: 1  kind: KIND_CPU }]
```

Key decisions:
- `max_batch_size: 0` — plates have different spatial sizes so they cannot be stacked into a single numpy batch.
- `KIND_CPU` — PaddleOCR with `onnxruntime` engine runs on CPU; `KIND_GPU` causes silent load failure if the container has no CUDA device.

---

## Running Inference

### Baseline (no Triton)

```bash
python anpr.py
```

Runs PaddleOCR directly in-process over all crops. Useful for ground-truth comparison.

### Single image

```bash
python triton_client.py path/to/plate.jpg
```

### Batch — entire crop directory (default)

```bash
python triton_client.py
```

### Batch — explicit paths / directory

```bash
python triton_client.py img1.jpg img2.jpg img3.jpg
python triton_client.py images/Indian_number_plate/test/crops/
```

---

## Client Architecture (`triton_client.py`)

### Async concurrent dispatch

All images are sent to Triton simultaneously via `async_infer()`. Since each plate crop has a unique spatial size, each image is its own request (no batching at the tensor level). Triton's Python Backend receives and processes them in its `execute()` loop.

```
Dispatch loop (fast, non-blocking):
  img_0 → async_infer() ──────────────────────────────────┐
  img_1 → async_infer() ──────────────────────────────────┤ all in-flight
  img_N → async_infer() ──────────────────────────────────┘

Collection loop (blocking per result):
  handle_0.get_result() → parse → print
  handle_1.get_result() → parse → print
  handle_N.get_result() → parse → print
```

### Per-image timing

Latency is measured as time blocked **inside `get_result()`**, not time-from-batch-start. This gives true per-image inference + network round-trip time.

```python
t_get     = time.perf_counter()
response  = handle.get_result()   # blocks only for this result
latency   = (time.perf_counter() - t_get) * 1000  # ms
```

### Plate assembly (`_assemble_plate`)

PaddleOCR detects one text region per line of the plate. Multi-line plates (top-row + bottom-row) arrive as separate strings. The assembly logic merges them robustly:

| Step | Action |
|---|---|
| 1 | Clean all tokens — strip everything except `A-Z 0-9` |
| 2 | Drop noise tokens (< 2 alphanumeric chars) |
| 3 | Check if any single token is already a valid plate |
| 4 | Concatenate all tokens in reading order → validate |
| 5 | Try all ordered subsequences of 2+ tokens → return first valid |
| 6 | Fallback: return full concatenation, mark unverified |

**Examples:**

| OCR output | Assembled | Valid? |
|---|---|---|
| `['R', 'MH02AQ2299']` | `MH02AQ2299` | ✓ |
| `['MH04', 'DW 9020']` | `MH04DW9020` | ✓ |
| `['MH47A', 'V6753']` | `MH47AV6753` | ✓ |
| `['MH02 AK 5481']` | `MH02AK5481` | ✓ |

### Indian plate format regex

```
^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$
```

Matches: `MH02MA5324`, `DL3CAB1234`, `KA01AB1234`, `MH47AV6753`

---

## Batch Output Format

```
  #    File                                    Text            ms
──────────────────────────────────────────────────────────────
   0   video11_2210_crop_0.jpg                 ✓ MH47AV6753    78.2
   1   video4_2170_crop_0.jpg                  ✓ MH04DW9020    71.5
   2   video9_40_crop_0.jpg                    ? NHO1DT1917    80.1
──────────────────────────────────────────────────────────────

══════════════════════════════════════════════════════════════
  TRITON OCR — BATCH SUMMARY
══════════════════════════════════════════════════════════════
  Total images submitted ················ 59
  Successful inferences ················· 59
  Failed inferences ····················· 0
  Success rate ·························· 100.0%
  Valid Indian plates ··················· 51 / 59
  Total batch time ······················ 4.512 s
  Throughput ···························· 13.08 img/s
══════════════════════════════════════════════════════════════
```

`✓` — assembled text matches Indian plate regex  
`?` — text returned but format unverified (OCR noise, partial plate, foreign format)

---

## Key Findings

- **Cold-start latency**: First request after Triton startup takes 60–120 s while PaddleOCR loads ONNX models. Subsequent requests are fast (~70–100 ms per plate).
- **Multi-line plates**: The biggest post-processing challenge. Without fragment assembly, ~30% of valid plates showed as `?` with `|`-joined partial strings.
- **Noise tokens**: Short stray OCR detections (1-char tokens like `R`, `8`, `IND` sticker text) must be filtered before assembly — handled by the `len < 2` threshold.
- **Timeout**: Default `tritonclient.http` timeout (60 s) is shorter than PaddleOCR cold-start. Fixed with `network_timeout=180.0` on client construction.
- **`KIND_GPU` vs `KIND_CPU`**: Using `KIND_GPU` without a CUDA-enabled container causes the model to silently fail to reach `READY` state, hanging all requests.

---

## Files Reference

| File | Purpose |
|---|---|
| `crop_plates.py` | Extract plate crops from YOLO-annotated frames |
| `anpr.py` | Baseline PaddleOCR inference (no server) |
| `model_repository/paddle_ocr/config.pbtxt` | Triton tensor interface definition |
| `model_repository/paddle_ocr/1/model.py` | `TritonPythonModel` — OCR execute logic |
| `triton_client.py` | Async batch client with plate assembly + analytics |
