# Contamination_Detection
For educational purpose

# Plastic Contamination Detection Using Computer Vision

A waste contamination detection system built with YOLOv11x and transfer learning, trained on the [TACO dataset](http://tacodataset.org/). Achieves **0.518 mAP50-95** on the test set — surpassing the benchmark of 0.463 set by Mewada et al. (IEEE Access, 2024) while using 3.3× less training data.

---

## Results

| Metric | Paper (YOLOv8x) | This project (YOLOv11x) |
|---|---|---|
| mAP50-95 | 0.463 | **0.518 (+11.9%)** |
| mAP50 | 0.605 | **0.613** |
| Precision | — | 0.673 |
| Recall | — | 0.494 |
| Training images | 1,999 | **793 (3.3× fewer)** |

### Per-class mAP50-95

| Class | mAP50-95 |
|---|---|
| foam | 0.615 |
| plastic_bag | 0.472 |
| rigid_plastic | 0.447 |

---

## Background

Recycling contamination — non-recyclable waste mixed into recycling collections — causes entire truckloads to be rejected at processing facilities. The reference paper (Mewada et al., 2024) demonstrated that computer vision can detect contamination in garbage truck hoppers using a proprietary dataset with color-based class labels (black_plastic, white_plastic, clear_plastic, polystyrene, blue_bag).

This project replicates and improves on that result using the publicly available TACO dataset and a key design insight: **color-based class labels are incompatible with TACO's annotation structure**. TACO labels objects by type (e.g. "Garbage bag", "Styrofoam piece"), not by color, so training a model with color-based class names on TACO data gives the model contradictory signals. Switching to material-based classes that match TACO's annotation vocabulary was the most impactful single change across all training runs.

---

## Dataset

**TACO (Trash Annotations in Context)** — 1,500+ images of litter photographed in real outdoor environments. Original annotations span 60 categories.

We map these to 3 material-based classes:

| Our class | TACO categories included |
|---|---|
| `plastic_bag` | Garbage bag, Single-use carrier bag, Polypropylene bag, Plastic film, Crisp packet, Other plastic wrapper, Squeezable tube, Plastified paper bag, Six pack rings |
| `rigid_plastic` | Clear/other plastic bottle, Plastic cup, Plastic lid, Plastic straw, Plastic utensils, Spread tub, Tupperware, Disposable food container, Other plastic container, Other plastic |
| `foam` | Styrofoam piece, Foam cup, Foam food container |

**Split**: 70% train / 20% val / 10% test (random seed 42)

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| plastic_bag | 562 | 213 | 82 | 857 |
| rigid_plastic | 848 | 276 | 131 | 1,255 |
| foam | 104 | 27 | 9 | 140 |

---

## Model

- **Architecture**: YOLOv11x (Ultralytics) — 56.9M parameters, 195.5 GFLOPs
- **Backbone**: CSPDarknet
- **Pretraining**: COCO (transfer learning)
- **Framework**: Ultralytics 8.4.x, PyTorch 2.10, CUDA 12.8

---

## Training Configuration

```python
model = YOLO('yolo11x.pt')   # COCO pretrained weights

model.train(
    data='data.yaml',
    epochs=200,
    patience=50,        # early stopping
    batch=8,
    imgsz=640,
    optimizer='SGD',
    lr0=0.0005,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    cos_lr=True,        # cosine LR decay

    # Augmentation
    mosaic=1.0,
    copy_paste=0.5,
    flipud=0.3,
    fliplr=0.5,
    degrees=15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    mixup=0.15,
)
```

Training stopped at epoch 102 (best at epoch 52) on a single NVIDIA Tesla T4 GPU (Google Colab).

---

## Key Design Decisions

### 1. Material-based classes instead of color-based

The reference paper used color-based labels (black_plastic, white_plastic, etc.) because their proprietary dataset was captured inside garbage trucks where bag color is a meaningful signal. TACO images are outdoor litter photographs — a "garbage bag" in TACO can be any color. Training with color-based labels on TACO gives the model contradictory supervision. Switching to material-based classes (plastic_bag, rigid_plastic, foam) aligned the labels with what the annotations actually represent.

### 2. Dropping the empty class

The reference paper's `blue_bag` class has zero instances in TACO. Keeping it as an empty class wastes model capacity and introduces noise. It was removed entirely.

### 3. Merging weak classes

An early 4-class version included `plastic_container` as a separate class with only 213 training instances. It achieved mAP of only 0.073, dragging down the overall score. Merging it into `rigid_plastic` produced a stronger, better-represented class.

### 4. Augmentation to compensate for small dataset

With only 793 training images (vs the paper's 1,999), aggressive augmentation was critical. `copy_paste=0.5` was particularly effective — it copies object instances between images, artificially increasing the frequency of rare classes like `foam`.

---

## Training Run History

| Run | Classes | Train images | Best mAP50-95 | Notes |
|---|---|---|---|---|
| Run 1 | 5 color-based | 610 | 0.363 | Baseline, blue_bag empty |
| Run 2 | 4 material-based | 610 | — | Session disconnected |
| Run 3 | 4 material-based | 793 | 0.307 | plastic_container dragging score |
| Run 4 | 3 material-based | 793 | **0.518** | Merged weak class, beat paper |

---

## Inference

```python
from ultralytics import YOLO

model = YOLO('weights/best.pt')

# On an image
results = model.predict(source='image.jpg', conf=0.20, save=True)

# On a video
results = model.predict(source='video.mp4', conf=0.20, save=True)
```

**Recommended confidence threshold**: `conf=0.20` for best mAP balance. Use `conf=0.10` if you prefer higher recall (catches more objects at the cost of more false positives).

---

## Reproduction

```bash
# 1. Clone TACO
git clone https://github.com/pedropro/TACO.git
cd TACO && python download.py

# 2. Install dependencies
pip install ultralytics pycocotools

# 3. Build dataset (run dataset_builder.py — converts TACO to YOLOv8 format with 3-class mapping)
python dataset_builder.py

# 4. Train
python train.py
```

---

## Limitations

- **Recall is 0.494** — the model detects roughly half the objects in a scene. This is expected given the small dataset size and highly cluttered real-world images. More annotated data would be the most direct fix.
- **Domain gap**: TACO images are outdoor litter; the reference paper's dataset is inside garbage trucks. These are different operational environments. Performance on garbage truck imagery is untested.
- **foam class is small**: Only 104 training instances. Despite achieving the highest per-class mAP (0.615), this class would benefit most from additional data.
- **One corrupt image**: `505.jpg` in the test set is unreadable and is excluded from all evaluations.

---

## Reference

Mewada, D., Agnew, C., Grua, E. M., Eising, C., Denny, P., Heffernan, M., Tierney, K., Van de Ven, P., & Scanlan, A. (2024). Contamination Detection From Highly Cluttered Waste Scenes Using Computer Vision. *IEEE Access*, 12, 129434–129446. https://doi.org/10.1109/ACCESS.2024.3456469

---

## Acknowledgements

Dataset: [TACO — Trash Annotations in Context](http://tacodataset.org/) by Proença and Simões (2020).
Model: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics).
Training infrastructure: Google Colab (Tesla T4 GPU).
