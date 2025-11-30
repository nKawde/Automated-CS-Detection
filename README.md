# Automated-CS-Detection

Automated-CS-Detection is a **three-stage deep learning pipeline** for screening embryo blastocyst images for CS (`CS_present` vs `No_CS`) and, when present, **localizing** CS regions.

Our pipeline:

1. **Stage 1 – Blastocyst segmentation & cropping**  
   Uses the official **Blastocyst-Seg** model (`segmentation.py`) to segment the blastocyst and crop the image around it.  
   We **directly use the open-source Blastocyst-Seg code and HRNet weights** from the original authors:  
   <https://github.com/mavillot/Blastocyst-Seg>

2. **Stage 2 – CS classification (EfficientNet-B2)**  
   An EfficientNet-B2 classifier predicts the probability of CS (`p(CS_present)`).  
   A configurable **probability threshold** determines whether CS is considered present.

3. **Stage 3 – CS localization (YOLO11-s)**  
   If Stage 2 is confident enough (probability ≥ threshold), a YOLO11-s model draws **axis-aligned bounding boxes** around CS regions.
   
> [!WARNING]
> **Research use only – not for clinical decision-making.**

---

## Installation & Demo Setup

### 1. Clone the repository and submodules

```bash
git clone https://github.com/nKawde/Automated-CS-Detection.git
cd Automated-CS-Detection

# Initialize and update submodules (includes Blastocyst-Seg)
git submodule update --init --recursive
```

### 2. Install Python dependencies

Recommended (Python 3.10+):
```bash
pip install torch torchvision timm ultralytics gradio pillow numpy
```

Then install any additional dependencies required by **Blastocyst-Seg** as listed in its README:

https://github.com/mavillot/Blastocyst-Seg

### 3. Pre-trained weights
Make sure you downloaded `hrnet.pth` from the Blastocyst-Seg repository and place it in:

```
Models/Blastocyst-Seg/weights/hrnet.pth
```

## Run the Demo

Once dependencies and weights are set up, you can launch the Web UI demo:
```bash
python demo.py
```

> [!TIP]
> You can change this threshold directly in the Gradio UI using the slider. So if you want more recall (catch more CS cases), decrease the threshold (but it will lower the precision and overall score)

### Repository structure
Key files and folders used by `demo.py`

```
Automated-CS-Detection/
├── demo.py                           # Gradio demo entry point
├── Models/
│   ├── Blastocyst-Seg/               # Submodule / external repo for Stage 1
│   │   ├── segmentation.py           # Called by demo.py
│   │   └── weights/
│   │       └── hrnet.pth             # Blastocyst-Seg HRNet weights
│   ├── Stage2Classifier/
│   │   └── weights/
│   │       └── cs_effb2.pt           # EfficientNet-B2 CS classifier weights
│   └── Stage3Detector/
│       └── weights/
│           └── cs_yolo11s.pt         # YOLO11-s CS detector weights
├── DataSet/                          # (optional) training/validation data etc.
└── .gitmodules                       # Blastocyst-Seg submodule config
```

## Training Stage 2 (Classifier)

Stage 2 is a **binary EfficientNet-B2 classifier** that predicts whether a cropped blastocyst image contains CS (`1`) or not (`0`).  
This section explains how to train (or re-train) this classifier and how to plug a new checkpoint into the demo.

### 1. Prepare the CSV files

The training script expects **three CSV files** for train / val / test splits, each with the columns:

- `path` – absolute or relative path to the image file
- `label` – integer class label (`0` = No_CS, `1` = CS_present)

Example `train.csv`:

```csv
path,label
DataSet/stage2/train/img_0001.png,0
DataSet/stage2/train/img_0002.png,1
DataSet/stage2/train/img_0003.png,0
...
```

> [!NOTE]
> Images are expected to be RGB. The script will resize/augment them internally.

### 2. Run The Training

From the repo root:

```bash
python Models/Stage2Classifier/train_classifier.py \
  --train_csv DataSet/stage2_train.csv \
  --val_csv   DataSet/stage2_val.csv   \
  --test_csv  DataSet/stage2_test.csv  \
  --out_dir   outputs_b2
...
```

Important arguments (with defaults):

* `--image_size` (default: `260`) – input size for EfficientNet-B2.

* `--batch_size` (default: `16`)

* `--epochs` (default: `30`)

* `--lr` (default: `3e-4`)

* `--weight_decay` (default: `1e-4`)

* `--pretrained` – use ImageNet-pretrained EfficientNet-B2 (if weights available locally).

* `--no_amp` – disable mixed precision.

* `--cpu` – force CPU.

* `--patience` (default: `8`) – early-stopping patience on val PR-AUC.

Run `python train_classifier.py -h` for all options.

### 3. Outputs

After training, `--out_dir` will contain:

* `best.pt` – best checkpoint

* `history.json` – list of per-epoch validation metrics

* `val_pr_curve.png`, `val_roc_curve.png`, `val_confusion_matrix.png`

* `test_pr_curve.png`, `test_roc_curve.png`, `test_confusion_matrix.png`

* `val_preds.csv`, `test_preds.csv` – probabilities, labels, and predictions

If you train a new classifier and it performs better, you should update the weights used by `demo.py`.
1. Take the `best.pt` checkpoint from your out_dir.
2. Copy and rename it to:
```
Models/Stage2Classifier/weights/cs_effb2.pt
```




## Acknowledgements
We use the exact open-source implementation and HRNet weights from the authors’ Blastocyst-Seg repository:
https://github.com/mavillot/Blastocyst-Seg
Please cite and acknowledge the original Blastocyst-Seg work if you use this pipeline in research.











