# Tom and Jerry Image Classification Implementation Plan

This plan outlines the steps to adapt the image classification pipeline in [`train.py`](file:///c:/Users/Yasiru%20Sithum/OneDrive/Documents/webprojects/MachineLearning/ImageClassificationModel/train.py) to classify images into four categories: **tom**, **jerry**, **both**, and **none**.

---

## Proposed Changes

We will modify the configuration inside [`train.py`](file:///c:/Users/Yasiru%20Sithum/OneDrive/Documents/webprojects/MachineLearning/ImageClassificationModel/train.py) and prepare the dataset folder structure under [`my_dataset`](file:///c:/Users/Yasiru%20Sithum/OneDrive/Documents/webprojects/MachineLearning/my_dataset).

### Component: Dataset Organization

To use the default `ImageClassificationDataset` loader (which inherits the PyTorch `ImageFolder` design), we need to structure the training dataset with sub-directories corresponding to the target classes.

1. Create a `train` folder inside `my_dataset`.
2. Inside `train`, create four sub-folders representing the four classes:
   - `tom/` (contains images with only Tom)
   - `jerry/` (contains images with only Jerry)
   - `both/` (contains images with both Tom and Jerry)
   - `none/` (contains images with neither)

#### Directory Structure Layout:
```
c:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\
├── train\
│   ├── tom\
│   │   ├── tom_001.jpg
│   │   └── ...
│   ├── jerry\
│   │   ├── jerry_001.jpg
│   │   └── ...
│   ├── both\
│   │   ├── both_001.jpg
│   │   └── ...
│   └── none\
│       ├── none_001.jpg
│       └── ...
└── test\  (Optional: place unlabelled images here for inference predictions)
    ├── test_001.jpg
    └── ...
```

---

### Component: Model Configuration Pipeline

#### [MODIFY] [train.py](file:///c:/Users/Yasiru%20Sithum/OneDrive/Documents/webprojects/MachineLearning/ImageClassificationModel/train.py)
We need to update the `Config` class at the top of the script:
- Set `NUM_CLASSES` to `4`.
- Set `TRAIN_DIR` to point to the local training directory: `r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\train"`.
- Set `TEST_DIR` to point to the local test directory: `r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\test"`. If you do not have test images, we should disable the test inference call at the bottom of the script.
- Set `NUM_WORKERS` to `0`. (PyTorch DataLoader with multithreading on Windows often causes hang issues, setting to 0 ensures smooth execution).

```diff
class Config:
    # ── Dataset details ──────────────────────────────────────────────────────
-   TRAIN_DIR       = r"path/to/train"        # Root of training images (with class sub-folders)
-   TEST_DIR        = r"path/to/test"         # Folder of unlabelled test images
-   NUM_CLASSES     = 10                       # Number of target classes
+   TRAIN_DIR       = r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\train"
+   TEST_DIR        = r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\test"
+   NUM_CLASSES     = 4                        # Number of target classes: tom, jerry, both, none
    CLASS_IMBALANCE = False                    # Set True if classes are imbalanced
    IMAGE_SIZE      = 224                      # Target H×W (224 for ConvNeXt-Tiny, 300 for EfficientNetV2-S)
...
    PHASE1_EPOCHS   = 3                        # Head-only training epochs
    PHASE2_EPOCHS   = 12                       # Full fine-tuning epochs
    BATCH_SIZE      = 32
-   NUM_WORKERS     = 4                        # DataLoader workers (set 0 on Windows if issues)
+   NUM_WORKERS     = 0                        # DataLoader workers (set 0 on Windows to avoid multi-processing issues)
```

At the bottom of the script inside `__main__`, if there are no test images in the `TEST_DIR`, ensure that the test-time augmentation (TTA) step is commented out to avoid errors:
```python
    # ── Step 3: Inference on test set with TTA ────────────────────────────
    # Uncomment the line below when you have a test folder ready:
    # predict_with_tta(cfg)
```

---

## Verification Plan

### Automated/Manual Tests
To verify the configuration is valid:
1. **Mock Dataset Verification**:
   We will create a helper dry-run/mock script in the artifacts/scratch folder to verify that:
   - The directory structure is read properly.
   - The class mapping maps to exactly 4 classes (`['both', 'jerry', 'none', 'tom']` in alphabetical order).
2. **Execute Training**:
   Run the modified training script from the virtual environment:
   ```powershell
   # Activate virtual environment
   .\ImageClassificationModel\.venv\Scripts\Activate.ps1
   # Run train.py
   python ImageClassificationModel\train.py
   ```
