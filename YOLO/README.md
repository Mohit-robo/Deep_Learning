# YOLO Object Detection 🎯

Implementations of the YOLO (You Only Look Once) family of object detection models, demonstrating the evolution of real-time detection architectures from YOLOv4 through YOLOv7.

## 📚 Overview

YOLO is a groundbreaking real-time object detection framework that frames detection as a regression problem. Unlike two-stage detectors (Faster R-CNN, Mask R-CNN), YOLO predicts bounding boxes and class probabilities directly from full images in a single forward pass, achieving unprecedented speed-accuracy trade-offs.

**Learning Objective**: Understand the evolution of YOLO architecture and how to train/deploy state-of-the-art object detectors.

---

## 📂 Directory Structure

```
YOLO/
├── README.md                          # This file
├── Yolov4/                           # YOLOv4 implementations
│   └── Yolov4.ipynb                 # YOLOv4 notebook
├── Yolov5/                          # YOLOv5 implementations
│   ├── Yolov5.ipynb                 # YOLOv5 notebook
│   └── yolo_pytorch.py              # YOLOv5 PyTorch implementation
├── Yolov6/                          # YOLOv6 implementations
│   ├── Yolov6.ipynb                 # YOLOv6 notebook
│   └── yolov6.py                    # YOLOv6 PyTorch implementation
└── Yolov7/                          # YOLOv7 implementations
    └── Yolov7.ipynb                 # YOLOv7 notebook
```

---

## 🔍 YOLO Architecture Overview

### What Makes YOLO Special?

Traditional object detection (R-CNN family):
```
Image → Region Proposals → Classification → Bounding Box Refinement
(Multiple passes, complex pipeline)
```

YOLO's approach:
```
Image → Single Forward Pass → Predictions (boxes + classes + confidence)
(Faster, simpler, end-to-end)
```

### Detection Pipeline

```
Input Image (416×416)
         ↓
    Backbone (extract features)
         ↓
    Feature Pyramid (multi-scale)
         ↓
    Detection Head (predict boxes)
         ↓
    NMS (remove duplicates)
         ↓
    Output Detections
```

---

## 🎯 YOLO Versions

### **YOLOv4** — Optimized for Accuracy
**Year**: 2020 | **Paper**: [YOLOv4: Optimal Speed and Accuracy](https://arxiv.org/abs/2004.10934)

**Key Innovations**:
- **Backbone**: CSPDarknet (Cross Stage Partial)
- **Neck**: Spatial Pyramid Pooling (SPP)
- **Head**: Multiple detection heads
- **Techniques**: IoU loss, CIoU loss, DropBlock regularization

**Characteristics**:
- Good accuracy-speed trade-off
- Suitable for edge devices
- Well-documented ecosystem

**File**: `Yolov4/Yolov4.ipynb`

---

### **YOLOv5** — Easy to Use
**Year**: 2020 | **Repo**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

**Key Improvements**:
- **Efficient Design**: Smaller model sizes (S, M, L, X variants)
- **Auto Learning Rate**: Automatic hyperparameter tuning
- **Mosaic Augmentation**: Combines 4 images during training
- **Better Defaults**: Out-of-the-box training recipes

**Characteristics**:
- Most popular (extensive community)
- Easiest to use and deploy
- Multiple size options for different devices

**Files**:
- `Yolov5/Yolov5.ipynb` - Full notebook tutorial
- `Yolov5/yolo_pytorch.py` - PyTorch implementation

**Quick Example**:
```python
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
results = model('image.jpg')
results.show()
```

---

### **YOLOv6** — Hardware Aware
**Year**: 2022 | **Paper**: [YOLOv6: A Single-Stage Object Detector](https://arxiv.org/abs/2209.02976)

**Key Innovations**:
- **Anchor-Free**: Direct coordinate prediction
- **Hardware-Aware Design**: Optimized for different hardware
- **EfficientRep Backbone**: Efficient feature extraction
- **DFL (Distribution Focal Loss)**: Better loss function

**Characteristics**:
- Anchor-free detection
- Improved efficiency
- Good for various deployment scenarios

**Files**:
- `Yolov6/Yolov6.ipynb` - Tutorial notebook
- `Yolov6/yolov6.py` - Implementation

---

### **YOLOv7** — Latest & Greatest
**Year**: 2022 | **Paper**: [YOLOv7: Trainable Architecture](https://arxiv.org/abs/2207.02696)

**Key Innovations**:
- **E-ELAN**: Enhanced Efficient Layer Aggregation
- **Model Scaling**: Scaling strategy for different sizes
- **Auxiliary Head**: Helps training with additional supervision
- **RepConv**: Reparameterized convolutions for inference speed

**Characteristics**:
- State-of-the-art accuracy
- Fastest inference
- Latest techniques and best practices

**File**: `Yolov7/Yolov7.ipynb`

---

## 📊 Comparison Table

| Aspect | YOLOv4 | YOLOv5 | YOLOv6 | YOLOv7 |
|--------|--------|--------|--------|---------|
| **Year** | 2020 | 2020 | 2022 | 2022 |
| **Anchors** | Yes | Yes | No | Yes |
| **Ease of Use** | 🟡 | 🟢 | 🟡 | 🟡 |
| **Accuracy** | 🟡 | 🟡 | 🟡 | 🟢 |
| **Speed** | 🟡 | 🟡 | 🟢 | 🟢 |
| **Community** | 🟡 | 🟢 | 🟡 | 🟡 |
| **Deployment** | 🟢 | 🟢 | 🟢 | 🟢 |

---

## 🚀 Quick Start

### Using YOLOv5 (Recommended for Beginners)

```bash
# Clone repo
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install dependencies
pip install -r requirements.txt

# Inference on image
python detect.py --source image.jpg

# Train on custom dataset
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt

# Validate
python val.py --weights yolov5s.pt --data data.yaml
```

### Using Local Implementations

```bash
# Run YOLOv5 notebook
cd Yolov5
jupyter notebook Yolov5.ipynb

# Run YOLOv7 notebook
cd ../Yolov7
jupyter notebook Yolov7.ipynb
```

---

## 🎓 Core Concepts

### 1. Detection Output Format

YOLO outputs predictions for each image:
```
[batch_size, num_predictions, (x, y, w, h, confidence, class_probs)]
```

Where:
- `x, y`: Bounding box center
- `w, h`: Width and height
- `confidence`: Objectness score (0-1)
- `class_probs`: Probability for each class

### 2. Loss Function

```
Total Loss = Localization Loss + Classification Loss + Objectness Loss

Localization Loss = IoU loss (bounding box regression)
Classification Loss = Cross-entropy (class prediction)
Objectness Loss = Binary cross-entropy (object presence)
```

### 3. Non-Maximum Suppression (NMS)

Remove overlapping detections:
1. Sort predictions by confidence
2. Select highest confidence prediction
3. Remove all predictions with IoU > threshold
4. Repeat until all processed

### 4. Metrics

- **mAP (mean Average Precision)**: Overall detection quality
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

---

## 📚 Training Custom Model

### Step 1: Prepare Dataset

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Step 2: Create Data Configuration (data.yaml)

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 2  # number of classes
names: ['class1', 'class2']
```

### Step 3: Train Model

```python
from yolov5 import train

model = train(
    img=640,
    batch=16,
    epochs=100,
    data='data.yaml',
    weights='yolov5s.pt'
)
```

### Step 4: Evaluate & Deploy

```python
# Validation
results = model.val()

# Inference
predictions = model.predict('test_image.jpg')

# Save model
model.save('best_model.pt')
```

---

## 🛠️ Common Use Cases

### 1. Real-time Video Detection
```python
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', results.render()[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 2. Traffic Monitoring
- Detect vehicles, pedestrians, cyclists
- Count traffic flow
- Monitor traffic violations

### 3. Security & Surveillance
- Detect people, suspicious behavior
- Alert on unauthorized access
- Person tracking

### 4. Autonomous Driving
- Detect obstacles, lane markers
- Traffic sign recognition
- Pedestrian detection

### 5. Industrial Inspection
- Defect detection
- Quality control
- Object counting

---

## ⚙️ Hyperparameters

Key parameters to tune:
- **Learning Rate**: 0.001-0.01
- **Batch Size**: 8-64 (higher if GPU memory allows)
- **Epochs**: 50-300
- **Image Size**: 320, 416, 512, 640 (larger = more accuracy but slower)
- **Augmentation**: Mosaic, mixup, rotation, scaling

---

## 📊 Benchmarks

**COCO Dataset Performance**:
| Model | mAP | Speed (ms) | Params (M) |
|-------|-----|-----------|-----------|
| YOLOv4 | 43.5 | 62 | 245 |
| YOLOv5s | 36.7 | 2 | 7.2 |
| YOLOv6s | 42.4 | 4 | 17.2 |
| YOLOv7 | 52.9 | 6.3 | 36.9 |

---

## ✅ Exercises

1. **Train on Custom Dataset**: Prepare your own images and labels
2. **Compare Versions**: Benchmark YOLOv5 vs YOLOv7 on same data
3. **Deploy to Edge**: Run on Raspberry Pi or Jetson Nano
4. **Video Processing**: Build real-time detection pipeline
5. **Multi-GPU Training**: Utilize multiple GPUs
6. **Model Optimization**: Convert to ONNX or TensorRT
7. **Integration**: Build REST API for inference
8. **Performance Analysis**: Profile inference time and memory

---

## 🔗 References

**Official Papers**:
- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
- [YOLOv5 (Ultralytics)](https://github.com/ultralytics/yolov5)
- [YOLOv6 Paper](https://arxiv.org/abs/2209.02976)
- [YOLOv7 Paper](https://arxiv.org/abs/2207.02696)

**Useful Resources**:
- [YOLO Official Website](https://www.yolov5.com/)
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Object Detection Explained](https://neptune.ai/blog/object-detection)

---

## 🔜 Next Steps

After mastering YOLO:
- Deploy to production: [OCR](../OCR/) for production inference patterns
- Combine with tracking: [trackers](../trackers/) for multi-frame consistency
- Explore segmentation: [Paper_review_notes/FCN](../Paper_review_notes/) for instance segmentation
- Build end-to-end systems: Combine detection with post-processing

---

**Last Updated**: June 2026
**Difficulty**: 🟡 Intermediate
**Prerequisites**: Python, CNNs ([Pytorch_series](../Pytorch_series/)), basic computer vision