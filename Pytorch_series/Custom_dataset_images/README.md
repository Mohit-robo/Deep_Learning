# Custom Dataset & DataLoaders 📊

Building custom datasets and DataLoaders for non-standard data formats. Learn to handle diverse data types and create efficient data pipelines.

## 📚 Overview

This module demonstrates how to work with custom data formats, specifically face landmark detection from CSV files and image data.

**Key Concepts**:
- Custom `Dataset` class implementation
- Data normalization and preprocessing
- Efficient data loading with `DataLoader`
- Handling heterogeneous data

---

## 📂 Contents

### **face_landmarks_dataset.ipynb** — Complete Walkthrough

End-to-end implementation of a custom dataset for face landmark detection.

**Dataset Structure**:
```
custom_dataset.csv
├── image_name.jpg,x1,y1,x2,y2,...,x68,y68
├── photo.jpg,100.5,50.3,110.2,60.1,...
└── ...
```

**What You'll Learn**:
1. Reading CSV files with Pandas
2. Loading images with OpenCV/PIL
3. Data normalization (landmarks to [0,1])
4. Creating custom `Dataset` class
5. Using `DataLoader` for batching
6. Training loop with custom data

---

## 🔧 Key Implementation

### Custom Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2

class FaceLandmarksDataset(Dataset):
    """Custom dataset for face landmarks."""
    
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to csv file with landmarks.
            img_dir (string): Directory with all the images.
            transform (callable): Optional transform to be applied on images.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        # Read image
        img_name = os.path.join(self.img_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        
        # Read landmarks (all columns except first)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Normalize landmarks to [0, 1]
        landmarks = landmarks / image.shape[0]
        
        return torch.from_numpy(image).float(), torch.from_numpy(landmarks).float()
```

### Using DataLoader

```python
# Create dataset
dataset = FaceLandmarksDataset(
    csv_file='custom_dataset.csv',
    img_dir='images/',
    transform=transforms.ToTensor()
)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate over batches
for images, landmarks in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Landmarks shape: {landmarks.shape}")
    
    # Training code here
    outputs = model(images)
    loss = criterion(outputs, landmarks)
```

---

## 🎯 Common Data Formats

### CSV with Paths
```
image_path,label,value1,value2
data/img1.jpg,cat,0.5,0.3
data/img2.jpg,dog,0.8,0.2
```

### CSV with Image Data (encoded)
```
filename,image_data(base64),landmarks
img1.jpg,iVBORw0KGgo...,x1 y1 x2 y2...
img2.jpg,/9j/4AAQSkZJRgABA...,x1 y1 x2 y2...
```

### JSON Format
```json
{
    "data": [
        {
            "image": "img1.jpg",
            "landmarks": [[100, 50], [110, 55], ...]
        }
    ]
}
```

### Directory Structure
```
train/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
├── class2/
│   ├── img3.jpg
```

---

## 🛠️ Data Preprocessing

### Image Preprocessing

```python
from torchvision import transforms

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomHorizontalFlip(),
])
```

### Landmark Normalization

```python
def normalize_landmarks(landmarks, img_shape):
    """Normalize landmarks to [0, 1] range."""
    height, width = img_shape
    landmarks[:, 0] /= width
    landmarks[:, 1] /= height
    return landmarks

def denormalize_landmarks(landmarks, img_shape):
    """Convert landmarks back to pixel coordinates."""
    height, width = img_shape
    landmarks[:, 0] *= width
    landmarks[:, 1] *= height
    return landmarks
```

---

## 🔍 Best Practices

### 1. Efficient Batch Loading
```python
# Use num_workers for parallel loading
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Load 4 batches in parallel
    pin_memory=True  # For GPU transfer
)
```

### 2. Data Splitting
```python
from torch.utils.data import random_split

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### 3. Data Validation
```python
# Check data before training
for images, landmarks in train_loader:
    print(f"Images shape: {images.shape}")
    print(f"Landmarks shape: {landmarks.shape}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Landmarks range: [{landmarks.min():.2f}, {landmarks.max():.2f}]")
    break
```

---

## ✅ Exercises

1. **Create Custom Dataset**: Implement for your own data format
2. **Data Visualization**: Display batch with landmarks
3. **Augmentation**: Add transformations to Dataset
4. **Memory Optimization**: Profile memory usage
5. **Parallel Loading**: Test different num_workers values
6. **Data Validation**: Check for corrupted files
7. **Stratified Splitting**: Maintain class distribution
8. **Caching**: Implement sample caching for speed

---

**Difficulty**: 🟡 Intermediate
**Prerequisites**: PyTorch basics, Pandas, OpenCV