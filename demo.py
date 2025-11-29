import sys
from pathlib import Path
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as T

import torch.nn as nn
import torchvision.models as models

from ultralytics import YOLO  # YOLO11 / YOLO11s
import gradio as gr

# -------------------------
# CONFIG
# -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths – change these to your actual paths
PROJECT_ROOT = Path(__file__).resolve().parent
BLASTO_ROOT = PROJECT_ROOT / "Models" / "Blastocyst-Seg"
CS_EFFB2_WEIGHTS = PROJECT_ROOT / "Models" / "Stage2Classifier"/"weights"/ "cs_effb2.pt"

# YOLO11-s bbox model
YOLO11_S_WEIGHTS = PROJECT_ROOT / "Models" / "Stage3Detector" /"weights"/ "cs_yolo11s.pt"

BLASTO_WEIGHTS = PROJECT_ROOT / "Models" /"Blastocyst-Seg" / "weights" / "hrnet.pth"

sys.path.append(str(BLASTO_ROOT))

# -------------------------
# DEFAULT THRESHOLD FROM CHECKPOINT
# -------------------------

def get_default_threshold_from_ckpt(ckpt_path: Path, fallback: float = 0.5) -> float:
    """
    Try to read the best CS threshold stored in the classifier checkpoint as:
        ckpt["val_metrics"]["best_thr"]

    If not found or anything fails, return `fallback`.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            print("[WARN] Checkpoint is not a dict, using fallback threshold.")
            return fallback

        # Your requested pattern:
        thr = float(ckpt.get("val_metrics", {}).get("best_thr", fallback))

        # Basic sanity check
        if 0.0 < thr < 1.0:
            print(f"[INFO] Loaded default CS threshold {thr:.4f} from ckpt['val_metrics']['best_thr'].")
            return thr
        else:
            print(f"[WARN] Loaded threshold {thr} out of [0,1] range, using fallback {fallback}.")
            return fallback

    except Exception as e:
        print(f"[WARN] Could not load default threshold from checkpoint: {e}")
        print(f"[INFO] Using fallback CS threshold={fallback:.2f}.")
        return fallback


DEFAULT_THRESHOLD = get_default_threshold_from_ckpt(CS_EFFB2_WEIGHTS, fallback=0.5)

# -------------------------
# STAGE 1: Blastocyst segmentation + crop
# -------------------------

class BlastocystSegmenter:
    """
    Thin wrapper around the official Blastocyst-Seg 'segmentation.py' script.

    It:
      1. Saves the input numpy RGB image to a temporary PNG.
      2. Calls:  python segmentation.py path_img path_weights
      3. Reads 'prediction.png' produced by the script.
      4. Converts it into a binary mask (H x W, values {0,1}).
    """

    def __init__(
        self,
        repo_root: Path,
        weights_path: Path,
        python_executable: str | None = None,
    ):
        self.repo_root = Path(repo_root)
        self.weights_path = Path(weights_path)
        self.segmentation_script = self.repo_root / "segmentation.py"
        self.python_executable = python_executable or sys.executable

        if not self.segmentation_script.exists():
            raise FileNotFoundError(f"segmentation.py not found at {self.segmentation_script}")

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Blastocyst-Seg weights not found at {self.weights_path}")

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Run Blastocyst-Seg and return a binary mask (H x W) where 1=blastocyst structures.
        If something fails, returns an all-zero mask.
        """
        # 1) Save temporary input image inside repo (easiest for their script)
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            dir=self.repo_root,
            delete=False
        ) as tmp:
            tmp_input_path = Path(tmp.name)
            Image.fromarray(image_rgb).save(tmp_input_path)

        # 2) Call their segmentation script
        try:
            cmd = [
                self.python_executable,
                str(self.segmentation_script),
                str(tmp_input_path),
                str(self.weights_path),
            ]

            subprocess.run(
                cmd,
                cwd=self.repo_root,   # so that prediction.png is written here
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(f"[Blastocyst-Seg] segmentation failed: {e}")
            # Clean up temp file
            try:
                tmp_input_path.unlink(missing_ok=True)
            except Exception:
                pass
            # Return empty mask on failure
            h, w, _ = image_rgb.shape
            return np.zeros((h, w), dtype=np.uint8)

        # Clean up temp file
        try:
            tmp_input_path.unlink(missing_ok=True)
        except Exception:
            pass

        # 3) Read prediction.png that the script generates
        pred_path = self.repo_root / "prediction.png"
        if not pred_path.exists():
            print("[Blastocyst-Seg] prediction.png not found after running script.")
            h, w, _ = image_rgb.shape
            return np.zeros((h, w), dtype=np.uint8)

        mask_img = Image.open(pred_path).convert("L")
        mask = np.array(mask_img)

        # 4) Convert to simple binary mask (anything non-zero = structure)
        binary_mask = (mask > 0).astype(np.uint8)

        return binary_mask


BLASTO_SEGMENTER = BlastocystSegmenter(
    repo_root=BLASTO_ROOT,
    weights_path=BLASTO_WEIGHTS,
)


def crop_from_mask(image_rgb: np.ndarray, mask: np.ndarray, padding: int = 10):
    """
    Given RGB image and binary mask, return cropped RGB image around mask.
    Returns None if no positive pixels.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    h, w = mask.shape
    y_min = max(y_min - padding, 0)
    x_min = max(x_min - padding, 0)
    y_max = min(y_max + padding, h - 1)
    x_max = min(x_max + padding, w - 1)

    cropped = image_rgb[y_min:y_max + 1, x_min:x_max + 1, :]
    return cropped


def stage1_segment_and_crop(image_rgb: np.ndarray):
    """
    Stage 1:
      - Run Blastocyst-Seg to get a mask
      - Crop image to that mask
      - Return cropped image and status text
    """
    mask = BLASTO_SEGMENTER(image_rgb)

    if mask is None or mask.sum() == 0:
        status = "Stage 1 FAILED: no blastocyst found."
        return None, status

    cropped = crop_from_mask(image_rgb, mask)
    if cropped is None:
        status = "Stage 1 FAILED: no blastocyst found (empty crop)."
        return None, status

    status = "Stage 1 OK: blastocyst segmented and cropped."
    return cropped, status


# -------------------------
# STAGE 2: EfficientNet-B2 CS classifier
# -------------------------

CLASS_NAMES = ["No_CS", "CS_present"]  # adjust to your training labels


def load_cs_classifier():
    # Load the training checkpoint (your .pt file)
    ckpt = torch.load(CS_EFFB2_WEIGHTS, map_location="cpu")

    # Many training scripts save a dict with keys:
    #   "epoch", "model_state", "optimizer_state", ...
    state = ckpt.get("model_state", ckpt)

    # Infer num_classes from the classifier weights in the checkpoint
    num_classes = state["classifier.1.weight"].shape[0]

    # Build torchvision EfficientNet-B2 backbone
    model = models.efficientnet_b2(weights=None)

    # Replace classifier head to match checkpoint's number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Now load the state dict
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    # (Optional) sanity check vs your CLASS_NAMES
    if len(CLASS_NAMES) != num_classes:
        print(
            f"[WARN] CLASS_NAMES has length {len(CLASS_NAMES)}, "
            f"but checkpoint has {num_classes} output neurons."
        )

    return model


CS_MODEL = None  # lazy load, to avoid errors if weights missing


cs_transform = T.Compose([
    T.Resize((260, 260)),  # or whatever input size you used
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def stage2_classify_cs(cropped_rgb: np.ndarray, threshold: float | None = None):
    """
    Stage 2:
      - Classify whether the blastocyst contains CS or not.
      - Supports:
          * 1-logit binary model (BCEWithLogitsLoss style)
          * multi-logit model (softmax)
      - Uses a configurable probability threshold for the 'CS_present' class.
      - If `threshold` is None, uses DEFAULT_THRESHOLD (loaded from weights).
      - Returns: status text, dict of probabilities, and a boolean "failed" flag.
    """
    global CS_MODEL
    if CS_MODEL is None:
        CS_MODEL = load_cs_classifier()

    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    pil_img = Image.fromarray(cropped_rgb)
    x = cs_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = CS_MODEL(x)

    # --- Handle 1-logit vs multi-class cases ---
    if logits.shape[1] == 1:
        # Binary classifier with a single logit: logit -> sigmoid
        logit = logits[0, 0]
        p_cs = torch.sigmoid(logit).item()
        # Build 2-class probabilities so we can still use CLASS_NAMES
        probs = np.array([1.0 - p_cs, p_cs], dtype=np.float32)
    else:
        # Standard multi-class softmax
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        # Assume CLASS_NAMES are ordered as in probs

    # Map to labels / probabilities dict
    prob_dict = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    # Extract CS probability (by label name, not index assumption)
    try:
        cs_index = CLASS_NAMES.index("CS_present")
    except ValueError:
        # Fallback: assume last class is CS_present
        cs_index = len(CLASS_NAMES) - 1

    p_cs = float(probs[cs_index])

    # Threshold-based decision
    if p_cs >= threshold:
        status = (
            f"Stage 2 OK: CS detected "
            f"(p_CS={p_cs:.3f} ≥ threshold={threshold:.2f})."
        )
        failed = False
    else:
        status = (
            f"Stage 2 FAILED: CS probability too low "
            f"(p_CS={p_cs:.3f} < threshold={threshold:.2f})."
        )
        failed = True

    return status, prob_dict, failed


# -------------------------
# STAGE 3: YOLO11-s CS localization (bounding boxes)
# -------------------------

YOLO_MODEL = None


def load_yolo_bbox():
    """
    Load YOLO11-s model trained for CS localization using standard axis-aligned bounding boxes.
    """
    model = YOLO(str(YOLO11_S_WEIGHTS))
    return model


def stage3_localize_cs(cropped_rgb: np.ndarray):
    """
    Stage 3:
      - Run YOLO11-s on the cropped image
      - Return annotated image + status text
      - Uses standard axis-aligned bounding boxes (no OBB).
    """
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = load_yolo_bbox()

    # YOLO accepts numpy RGB directly
    results = YOLO_MODEL.predict(cropped_rgb, verbose=False)

    if not results:
        return None, "Stage 3 FAILED: no YOLO results."

    r = results[0]

    # Plot annotated image (Ultralytics returns BGR)
    annotated_bgr = r.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB

    # Count detections (axis-aligned bounding boxes)
    n_dets = 0
    if hasattr(r, "boxes") and r.boxes is not None:
        try:
            n_dets = len(r.boxes)
        except TypeError:
            # Fallback in case boxes is a tensor-like structure
            try:
                n_dets = r.boxes.shape[0]
            except Exception:
                n_dets = 0

    if n_dets == 0:
        status = "Stage 3 FAILED: no CS localization (0 bbox detections)."
    else:
        status = f"Stage 3 OK: localized CS region(s) with {n_dets} bounding box detection(s)."

    return annotated_rgb, status


# -------------------------
# PIPELINE FUNCTION FOR GRADIO
# -------------------------

def run_pipeline(image_rgb: np.ndarray, threshold: float | None = None):
    """
    Main function that Gradio will call.
    Input:
      - image_rgb: RGB numpy image from Gradio.
      - threshold: probability threshold for 'CS_present' in Stage 2.
                   If None, uses DEFAULT_THRESHOLD.
    Outputs (in order):
      1. Stage 1 image (cropped blastocyst or None)
      2. Stage 1 status (str)
      3. Stage 2 status (str)
      4. Stage 2 probabilities (dict/JSON)
      5. Stage 3 image (annotated or None)
      6. Stage 3 status (str)
    """

    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    # -------- Stage 1 --------
    cropped, s1_status = stage1_segment_and_crop(image_rgb)

    if cropped is None:
        # Stage 1 failed → everything else is skipped
        s2_status = "Skipped: Stage 1 failed (no blastocyst)."
        s2_probs = {}
        s3_img = None
        s3_status = "Skipped: Stage 1 failed (no blastocyst)."
        return None, s1_status, s2_status, s2_probs, s3_img, s3_status

    # -------- Stage 2 --------
    s2_status, s2_probs, stage2_failed = stage2_classify_cs(cropped, threshold=threshold)

    if stage2_failed:
        # If no CS according to classifier → Stage 3 is skipped
        s3_img = None
        s3_status = (
            "Skipped: Stage 2 below threshold "
            "(no CS detected confidently enough by classifier)."
        )
        return cropped, s1_status, s2_status, s2_probs, s3_img, s3_status

    # -------- Stage 3 --------
    s3_img, s3_status = stage3_localize_cs(cropped)

    return cropped, s1_status, s2_status, s2_probs, s3_img, s3_status


# -------------------------
# GRADIO UI
# -------------------------

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
            # IVF Blastocyst CS Screening (Research Demo)

            **Pipeline:**
            1. Blastocyst-Seg → segment & crop blastocyst.
            2. EfficientNet-B2 → classify CS vs No-CS.
            3. YOLO11-s → localize CS with axis-aligned bounding boxes.

            **Threshold note:**  
            The default Stage 2 CS probability threshold is loaded from the classifier checkpoint  
            (`val_metrics.best_thr` = **{DEFAULT_THRESHOLD:.2f}**), but you can override it with the slider below.

            ⚠️ *For research use only – not for clinical decision-making.*
            """
        )

        with gr.Row():
            input_img = gr.Image(
                label="Upload embryo image",
                type="numpy",
                sources=["upload", "clipboard"],
            )

        # Threshold slider for Stage 2, default from weights
        with gr.Row():
            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=DEFAULT_THRESHOLD,
                step=0.01,
                label="Stage 2 CS probability threshold",
                info=f"Default = {DEFAULT_THRESHOLD:.2f} (loaded from checkpoint val_metrics.best_thr). "
                     "Stage 3 will only run if p(CS_present) ≥ this value.",
            )

        with gr.Row():
            with gr.Column():
                s1_img = gr.Image(label="Stage 1: Cropped blastocyst", type="numpy")
                s1_status = gr.Textbox(label="Stage 1 status")

            with gr.Column():
                s3_img = gr.Image(
                    label="Stage 3: YOLO11-s CS localization (bounding boxes)",
                    type="numpy"
                )
                s3_status = gr.Textbox(label="Stage 3 status")

        with gr.Row():
            s2_status = gr.Textbox(label="Stage 2: CS classifier result")
            s2_probs = gr.JSON(label="Stage 2: class probabilities")

        run_btn = gr.Button("Run 3-stage pipeline")

        run_btn.click(
            fn=run_pipeline,
            inputs=[input_img, threshold_slider],
            outputs=[s1_img, s1_status, s2_status, s2_probs, s3_img, s3_status],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
