# urban-object-detection-yolov5
Urban object detection using YOLOv5 with scale-sensitive performance analysis on Pascal VOC 2007.

## Project Overview

This project focuses on urban object detection using YOLOv5 on a filtered subset of the Pascal VOC 2007 dataset.

Instead of training a detector only, the main goal was to perform controlled experiments to analyze:

- Scale sensitivity (small, medium, large objects)
- Per-class performance
- Confusion patterns
- Resolution impact on detection quality

---

## Dataset

Dataset used:

Pascal VOC 2007  
Official website: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

From the original 20 classes, only urban-related classes were selected:

- bicycle  
- bus  
- car  
- motorbike  
- person  
- train  

After filtering:

- 2365 training images  
- 606 validation images  

---

## Experiments

Two controlled experiments were conducted:

| Experiment | Image Size | Epochs |
|------------|------------|--------|
| Exp 1      | 320        | 40     |
| Exp 2      | 640        | 40     |

Only image resolution was changed to ensure a fair comparison.

---

## Overall Performance Comparison

| Metric     | 320   | 640   |
|------------|-------|-------|
| mAP50      | 0.766 | 0.803 |
| mAP50-95   | 0.478 | 0.492 |
| Recall     | 0.646 | 0.716 |
| Precision  | 0.874 | 0.849 |

Increasing input resolution improved recall significantly.

---

## COCO-Style Size-Based Evaluation

Objects were categorized using COCO area thresholds:

- Small  
- Medium  
- Large  

Small object recall:

0.27 → 0.56  
(Over 100% improvement)

This confirms that object detection performance is highly sensitive to input resolution.

---

## Per-Class Performance (Image Size = 640)

| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| car        | 0.78      | 0.78   | 0.78 |
| person     | 0.76      | 0.78   | 0.77 |
| bicycle    | 0.71      | 0.72   | 0.71 |
| motorbike  | 0.80      | 0.68   | 0.74 |
| bus        | 0.62      | 0.76   | 0.68 |
| train      | 0.59      | 0.77   | 0.67 |

---

## Key Insights

- Small object detection improves dramatically with higher resolution.
- Medium-sized objects generate most false positives.
- Class imbalance affects precision in minority classes.
- Resolution scaling is an effective improvement strategy for small objects.

---

## Setup & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/MahdisHassani/urban-object-detection-yolov5.git
cd urban-object-detection-yolov5
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Download YOLOv5

Clone YOLOv5 inside the project directory:

```bash
git clone https://github.com/ultralytics/yolov5.git
```

---

### 4️⃣ Download Dataset

Download Pascal VOC 2007 from:

http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

Place it inside:

```
dataset/
```

---

### 5️⃣ Create Urban Subset

Generate filtered urban dataset:

```bash
python data_processing/create_urban_dataset.py
```

This will create:

```
dataset_urban/
```

---

### 6️⃣ Run Density Analysis

To analyze object density distribution:

```bash
python analysis/density_analysis.py
```

This script computes:
- Average objects per image
- Traffic level classification (low / medium / high)
- Top crowded images

---

### 7️⃣ Train the Model

Example (Image size = 640):

```bash
cd yolov5
python train.py --img 640 --batch 2 --epochs 40 --data ../configs/urban_dataset.yaml --weights yolov5n.pt
```

After training, copy the best model to:

```
weights/best.pt
```

---

### 8️⃣ Run Size-Based Evaluation

```bash
python analysis/size_based_evaluation.py
```

---

### 9️⃣ Run Per-Class Evaluation

```bash
python analysis/per_class_evaluation.py
```

---

## Notes

- Dataset and trained weights are not included due to size limitations.
- Make sure folder structure is preserved.
- Scripts use relative paths for portability.
