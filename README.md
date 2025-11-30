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

### Run the Demo

Once dependencies and weights are set up, you can launch the Web UI demo:
```bash
python demo.py
```

> [!TIP]
> You can change this threshold directly in the Gradio UI using the slider. So if you want more recall (catch more CS cases), decrease the threshold (but it will lower the precision and overall score)

### 4. Repository structure
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









