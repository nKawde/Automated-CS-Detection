import argparse
from pathlib import Path
import shutil
import pandas as pd
import os
import torch
from ultralytics import YOLO


from pathlib import Path

PROJECT_ROOT = Path.cwd()

# Root of the Dataset + Yaml
DATA_ROOT = PROJECT_ROOT / "Dataset"
YOLO_DATA_YAML = DATA_ROOT / "e2d2_joint_yolo.yaml"



# Clean DataSet to include only positive CS

JOINT_ROOT = PROJECT_ROOT / "Dataset"
ONLYCS_ROOT = PROJECT_ROOT / "DatasetOnlyCS"
CLS_TRAIN_CSV = JOINT_ROOT / "cls_train.csv"
CLS_VAL_CSV   = JOINT_ROOT / "cls_val.csv"
CLS_TEST_CSV  = JOINT_ROOT / "cls_test.csv"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path):
    if not src.is_file():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def process_split(split_name: str, src_csv: Path):
    """
    For a given split (train/val/test):
      - load cls_*.csv from E2D2_joint
      - filter label == 1 (CS present)
      - copy images & labels to E2D2_OnlyCS
      - write new cls_*.csv in E2D2_OnlyCS
    """
    if not src_csv.is_file():
        print(f"[WARN] CSV for split '{split_name}' not found: {src_csv}")
        return

    df = pd.read_csv(src_csv)
    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{src_csv} must contain 'image_path' and 'label' columns")

    # Keep only rows where CS exists
    df_cs = df[df["label"] == 1].copy()

    n_total = len(df)
    n_cs = len(df_cs)
    print(f"[{split_name.upper()}] total images={n_total}, CS_only={n_cs}")

    # Output CSV (same name, but under E2D2_OnlyCS)
    out_csv = ONLYCS_ROOT / f"cls_{split_name}.csv"

    # Paths for images and labels in new dataset
    img_cs_count = 0
    lbl_cs_count = 0

    new_rows = []

    for _, row in df_cs.iterrows():
        rel_path = row["image_path"]
        label_val = int(row["label"])

        # Source and destination for image
        src_img = JOINT_ROOT / rel_path
        dst_img = ONLYCS_ROOT / rel_path

        img_copied = copy_if_exists(src_img, dst_img)
        if not img_copied:
            print(f"[WARN] Missing image for {split_name}: {src_img}")
            continue

        img_cs_count += 1

        img_name = os.path.basename(rel_path)
        stem, _ = os.path.splitext(img_name)

        src_lbl = JOINT_ROOT / "labels" / split_name / f"{stem}.txt"
        dst_lbl = ONLYCS_ROOT / "labels" / split_name / f"{stem}.txt"

        lbl_copied = copy_if_exists(src_lbl, dst_lbl)
        if lbl_copied:
            lbl_cs_count += 1
        else:
            print(f"[WARN] No YOLO label found for image: {src_lbl}")

        new_rows.append({"image_path": rel_path, "label": label_val})

    # Save new CSV
    out_df = pd.DataFrame(new_rows)
    ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)

    print(
        f"[{split_name.upper()}] COPIED: CS_images={img_cs_count}, "
        f"with_YOLO_labels={lbl_cs_count}"
    )
    print("-" * 60)


def write_yolo_yaml():
    """
    Write a YOLO data yaml pointing to the E2D2_OnlyCS dataset.
    """
    yaml_path = ONLYCS_ROOT / "e2d2_onlycs_yolo.yaml"

    lines = [
        f"path: {ONLYCS_ROOT.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        "  0: CS",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[YAML] Wrote YOLO data file: {yaml_path}")


def onlyCSSamples():
    print(f"JOINT_ROOT:  {JOINT_ROOT}")
    print(f"ONLYCS_ROOT: {ONLYCS_ROOT}")
    print("Creating CS-only dataset...\n")

    # Process each split
    process_split("train", CLS_TRAIN_CSV)
    process_split("val",   CLS_VAL_CSV)
    process_split("test",  CLS_TEST_CSV)

    # Create YOLO data yaml for the CS-only dataset
    write_yolo_yaml()

    print("\nDone building E2D2_OnlyCS (CS-only dataset).")
    print("\nUsing the New E2D2_OnlyCS Dataset...\n")
    global DATA_ROOT
    DATA_ROOT = PROJECT_ROOT / "DatasetOnlyCS"
    global YOLO_DATA_YAML
    YOLO_DATA_YAML = DATA_ROOT / "e2d2_onlycs_yolo.yaml"




# Device helpers  (MPS / CUDA / CPU)
def resolve_yolo_device_string(device_str: str | None = None) -> str:
    """
    For Ultralytics YOLO (expects a string like 'mps', 'cpu', '0').

    device_str:
        - "auto" -> prefer MPS, then CUDA, then CPU
        - "mps", "cpu", "0", "cuda:0", ... -> used as-is
    """
    if device_str is not None and device_str != "auto":
        return device_str

    # Auto selection
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        # first GPU
        return "0"
    return "cpu"


#Training Yolo11-s
def train_yolo_CS(
    model_name: str = "yolo11s.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
):

    yolo_device = resolve_yolo_device_string(device)
    print(f"[YOLO] Using device: {yolo_device}")
    print(f"[YOLO] Data yaml: {YOLO_DATA_YAML}")

    # Load base YOLO11s model (pretrained COCO)
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(YOLO_DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(PROJECT_ROOT / "runs_YOLO11S"),
        name="CS_localizer_YOLO11S",
        exist_ok=True,
        task="detect",
        device=yolo_device,
    )

    print("[YOLO] Training finished.")
    print(f"[YOLO] Runs dir: {PROJECT_ROOT / 'runs_YOLO11S'}")
    return results


# simple eval/predict
def eval_yolo(
    weights_path: str,
    imgsz: int = 640,
    device: str = "auto",
):
    """
    Quick validation using the best/last weights on the val set
    """
    yolo_device = resolve_yolo_device_string(device)
    print(f"[YOLO-EVAL] Using device: {yolo_device}")

    model = YOLO(weights_path)
    results = model.val(
        data=str(YOLO_DATA_YAML),
        imgsz=imgsz,
        task="detect",
        device=yolo_device,
    )

    print("[YOLO-EVAL] Validation finished.")
    return results


def predict_folder(
    weights_path: str,
    source: str,
    imgsz: int = 640,
    device: str = "auto",
    conf: float = 0.25,
):

    yolo_device = resolve_yolo_device_string(device)
    print(f"[YOLO-PRED] Using device: {yolo_device}")

    model = YOLO(weights_path)
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        task="detect",
        device=yolo_device,
        save=True,  # saves annotated images in runs/detect/...
    )

    print("[YOLO-PRED] Prediction finished.")
    return results


#Main
def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11s detector For CS Localization."
    )
    sub = parser.add_subparsers(dest="command")

    #Clean dataset to include only CS (Optional: remove it if you want to include the whole dataset)
    onlyCSSamples()


    # ----- train -----
    p_train = sub.add_parser("train", help="Train YOLO11s")
    p_train.add_argument("--model_name", type=str, default="yolo11s.pt",
                         help="Base YOLO11 model (default: yolo11s.pt)")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--device", type=str, default="auto",
                         help="cpu, mps, cuda index (e.g. '0'), or auto")

    # ----- eval -----
    p_eval = sub.add_parser("eval", help="Validate trained weights on val set")
    p_eval.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights (e.g. best.pt)")
    p_eval.add_argument("--imgsz", type=int, default=640)
    p_eval.add_argument("--device", type=str, default="auto")

    # ----- predict -----
    p_pred = sub.add_parser("predict", help="Run detection on a folder of images")
    p_pred.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights (e.g. best.pt)")
    p_pred.add_argument("--source", type=str, required=True,
                        help="Folder or image path to run inference on")
    p_pred.add_argument("--imgsz", type=int, default=640)
    p_pred.add_argument("--device", type=str, default="auto")
    p_pred.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()

    if args.command == "train":
        train_yolo_CS(
            model_name=args.model_name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

    elif args.command == "eval":
        eval_yolo(
            weights_path=args.weights,
            imgsz=args.imgsz,
            device=args.device,
        )

    elif args.command == "predict":
        predict_folder(
            weights_path=args.weights,
            source=args.source,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
