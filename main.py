# ------------------------------- Imports --------------------------------
import os
import argparse
import torch
import numpy as np
import cv2
from rich.console import Console
from rich.progress import track
import time
import glob
import json
from ultralytics import YOLO

from src.model import SegModel as FCN
import src.CategoryDictionary as CatDic
from src.prepare_kaggle_data import prepare_plastic_dataset


# ------------------------------- Setup ----------------------------------
console = Console()

DEFAULT_INPUT_DIR = "data/input_data/"
OUTPUT_DIR = "data/output_data/"
MODEL_PATH = "checkpoints/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"
YOLO_WEIGHTS = "checkpoints/yolo11s.pt"

USE_GPU = True
FREEZE_BN = False
MAX_SIZE = 840  # max image dimension for preprocessing

TARGET_CLASSES = {"cup", "glass", "wine glass"}  # YOLO classes to keep

LEVEL_TO_PERCENT = {
    'Empty': 0, 'Level10': 10, 'Level20': 20, 'Level30': 30,
    'Level40': 40, 'Level50': 50, 'Level60': 60, 'Level70': 70,
    'Level80': 80, 'Level90': 90, 'Full': 100
}


# ------------------------------- Functions -------------------------------
def load_model(model_path: str, use_gpu: bool, freeze_bn: bool) -> FCN:
    net = FCN(CatDic.CatNum)
    if use_gpu and torch.cuda.is_available():
        console.log("[bold green]✓ Using GPU (Segmentation)")
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        console.log("[bold yellow]⚠ Using CPU (Segmentation)")
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        net.cpu()

    if freeze_bn:
        net.eval()
        console.log("[cyan]BatchNorm statistics frozen")
    return net


def load_yolo(weights_path: str = YOLO_WEIGHTS) -> YOLO:
    console.log("[bold cyan]Loading YOLO detector…")
    return YOLO(weights_path)


def preprocess_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w, _ = img.shape
    r = max(h, w)
    if r > MAX_SIZE:
        fr = MAX_SIZE / r
        img = cv2.resize(img, (int(w * fr), int(h * fr)))
    return np.expand_dims(img, axis=0)


def determine_liquid_level(predictions: dict) -> tuple:
    import src.CategoryDictionary as CatDic
    vessel_mask, liquid_mask = None, None

    for name, tensor in predictions.items():
        if name == "Vessel":
            vessel_mask = tensor.data.cpu().numpy()[0].astype(np.uint8)
        elif name in CatDic.CatName.values() and tensor is not None:
            if any(lbl in name for lbl in ["Liquid", "Foam", "Gel"]):
                mask = tensor.data.cpu().numpy()[0].astype(np.uint8)
                liquid_mask = mask if liquid_mask is None else (liquid_mask | mask)

    if vessel_mask is None or liquid_mask is None:
        return "Empty", 0, 0.0, 0.0

    liquid_inside_vessel = liquid_mask * (vessel_mask > 0)
    vessel_pixels, liquid_pixels = vessel_mask.sum(), liquid_inside_vessel.sum()
    if vessel_pixels == 0:
        return "Empty", 0, 0.0, 0.0

    continuous_percent = (liquid_pixels / vessel_pixels) * 100.0
    nearest_level = min(LEVEL_TO_PERCENT.items(),
                        key=lambda kv: abs(continuous_percent - kv[1]))
    return nearest_level[0], nearest_level[1], continuous_percent, (liquid_pixels / (vessel_pixels + 1e-6))


def save_output(out_dir: str, name: str, img: np.ndarray, predictions: dict, ending: str = "") -> None:
    h, w, _ = img.shape
    detected_level, level_percent, cont_percent, ratio = determine_liquid_level(predictions)

    level_dir = os.path.join(out_dir, f"Level_{level_percent}")
    os.makedirs(level_dir, exist_ok=True)

    # --- Extract masks ---
    vessel_mask = predictions.get("Vessel").data.cpu().numpy()[0].astype(np.uint8) if "Vessel" in predictions else None
    liquid_mask = predictions.get("Liquid GENERAL").data.cpu().numpy()[0].astype(np.uint8) if "Liquid GENERAL" in predictions else None
    liquid_inside_vessel = None
    if vessel_mask is not None and liquid_mask is not None:
        liquid_inside_vessel = liquid_mask * (vessel_mask > 0)

    # --- Create overlay ---
    overlay = img.copy()
    if liquid_inside_vessel is not None:
        Lb_resized = cv2.resize(liquid_inside_vessel, (w, h), interpolation=cv2.INTER_NEAREST)
        if Lb_resized.mean() > 0.001:
            color = (0, int(255 * (level_percent / 100)), int(255 * (1 - level_percent / 100)))
            overlay[Lb_resized == 1] = color

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 500.0)
    thickness = max(1, int(font_scale * 2))
    text = f"Liquid Level: {cont_percent:.1f}% → Nearest: {level_percent}%"
    cv2.putText(overlay, text, (10, int(30 * font_scale)), font, font_scale, (255, 255, 255), thickness)

    # --- Build mask visualizations ---
    def to_color_mask(mask, color=(0, 255, 0)):
        if mask is None:
            return np.zeros_like(img)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        color_img = np.zeros_like(img)
        color_img[mask_resized == 1] = color
        return color_img

    vessel_vis = to_color_mask(vessel_mask, color=(255, 0, 0))   # red
    liquid_vis = to_color_mask(liquid_mask, color=(0, 255, 255)) # yellow

    # --- Concatenate all views ---
    fin_img = np.concatenate([img, vessel_vis, liquid_vis, overlay], axis=1)

    # Save image
    out_name = os.path.join(level_dir, f"{os.path.splitext(name)[0]}{ending}.png")
    cv2.imwrite(out_name, fin_img)
    console.log(f"[green]💾 Saved result:[/green] {out_name} "
                f"[cyan](Continuous: {cont_percent:.1f}%, Nearest: {level_percent}%)[/cyan]")

    # --- Save JSON result ---
    result_data = {
        "image_name": name,
        "detected_level": detected_level,
        "nearest_percent": level_percent,
        "continuous_percent": round(cont_percent, 2),
        "liquid_ratio": round(ratio, 4)
    }

    json_path = os.path.join(out_dir, "results.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(result_data)

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)


def get_vessel_bbox(predictions: dict, min_area: int = 500) -> tuple:
    """
    Compute vessel bbox from the segmentation mask (for image/folder/dataset modes).
    """
    if "Vessel" not in predictions:
        return None
    vessel_mask = predictions["Vessel"].data.cpu().numpy()[0].astype(np.uint8)
    if vessel_mask.sum() < min_area:
        return None
    contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    margin = int(0.05 * max(w, h))
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(vessel_mask.shape[1], x + w + margin), min(vessel_mask.shape[0], y + h + margin)
    return (x1, y1, x2, y2)


def run_inference_on_image(net, img: np.ndarray, use_gpu=True, freeze_bn=False) -> dict:
    im_batch = np.expand_dims(img, axis=0)
    with torch.no_grad():
        _, out_lb_dict = net.forward(
            Images=im_batch,
            TrainMode=False,
            UseGPU=use_gpu,
            FreezeBatchNormStatistics=freeze_bn
        )
    return out_lb_dict


def resize_with_aspect_ratio(image, target_size=(224, 224), pad_color=(0, 0, 0)):
    """Resize while keeping aspect ratio, then pad to target size."""
    h, w = image.shape[:2]
    th, tw = target_size

    # scale factor
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # create padded canvas
    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)

    # center the resized image
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def process_image_with_crop(net, img: np.ndarray, path: str, out_dir: str) -> None:
    """
    Image/Folder/Dataset mode:
    1) Run segmentation on full frame
    2) Get vessel bbox from mask
    3) Crop + resize
    4) Segmentation on crop
    5) Save output
    """
    first_pass_preds = run_inference_on_image(net, img, use_gpu=USE_GPU, freeze_bn=FREEZE_BN)
    bbox = get_vessel_bbox(first_pass_preds)
    if bbox is None:
        console.log(f"[yellow]⚠ No vessel detected in {path}")
        return

    x1, y1, x2, y2 = bbox
    cropped = img[y1:y2, x1:x2]
    cropped_resized = resize_with_aspect_ratio(cropped, target_size=(224, 224))

    second_pass_preds = run_inference_on_image(net, cropped_resized, use_gpu=USE_GPU, freeze_bn=FREEZE_BN)
    save_output(out_dir, os.path.basename(path), cropped_resized, second_pass_preds, ending="_cropped")


# ------------------------------- Live / Video ----------------------------
def yolo_best_bbox(yolo_model: YOLO, frame: np.ndarray, target_classes=TARGET_CLASSES, conf_thres: float = 0.25):
    """
    Run YOLO on a frame and return the highest-confidence bbox for target classes.
    Returns (x1, y1, x2, y2) or None.
    """
    results = yolo_model(frame)[0]
    best = None
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        if label.lower() in target_classes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            if (best is None) or (conf > best[0]):
                best = (conf, tuple(xyxy))
    return None if best is None else best[1]


def run_on_stream(net,
                  yolo_model,
                  source=0,
                  save=False,
                  out_path="output_live.mp4",
                  refresh=0):
    """
    Live/Video mode:
    - Runs YOLO once to lock a vessel bbox and reuses it for all frames.
    - Press 'r' to force re-detection.
    - If refresh > 0, YOLO re-runs every N frames to update bbox.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        console.log(f"[bold red]❌ Cannot open video source: {source}")
        return

    # Video writer (optional)
    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        console.log(f"[green]💾 Saving stream to: {out_path}")

    console.log("[cyan]🎥 Live/Video inference started — 'q' to quit, 'r' to re-detect vessel.")
    bbox = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            console.log("[yellow]⚠ No more frames or failed to read.")
            break

        # Refresh logic
        force_redetect = (refresh > 0 and (frame_idx % refresh == 0))
        if bbox is None or force_redetect:
            new_bbox = yolo_best_bbox(yolo_model, frame, target_classes=TARGET_CLASSES)
            if new_bbox is not None:
                bbox = new_bbox
                console.log(f"[green]✓ YOLO bbox: {bbox}")
            else:
                # No bbox yet; show frame and continue
                cv2.imshow("Live Liquid Level", frame)
                if writer: writer.write(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                frame_idx += 1
                continue

        # Crop with stored bbox
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            # invalid bbox; force re-detect next frame
            bbox = None
            frame_idx += 1
            continue

        crop = frame[y1:y2, x1:x2]
        crop_resized = resize_with_aspect_ratio(crop, target_size=(224, 224))

        # Segmentation on the crop
        preds_crop = run_inference_on_image(net, crop_resized, use_gpu=USE_GPU, freeze_bn=FREEZE_BN)

        # Determine level
        _, nearest, cont_percent, _ = determine_liquid_level(preds_crop)

        # Overlay on original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Liquid: {cont_percent:.1f}% (≈{nearest}%)"
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # Show & save
        cv2.imshow("Live Liquid Level", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bbox = None  # force re-detection next loop

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    console.log("[bold green]✅ Stream ended.")


# ------------------------------- Main ------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Liquid Level Detection in Transparent Objects")
    parser.add_argument("--dataset", action="store_true", help="Run inference on full dataset")
    parser.add_argument("--image", type=str, help="Run inference on a single image")
    parser.add_argument("--folder", type=str, help="Run inference on all images in a folder")
    parser.add_argument("--live", action="store_true", help="Run inference on webcam stream")
    parser.add_argument("--video", type=str, help="Run inference on a video file path")
    parser.add_argument("--save", action="store_true", help="Save output video for live/video modes")
    parser.add_argument("--refresh", type=int, default=0, help="Re-run YOLO every N frames in live/video (0=never)")
    args = parser.parse_args()

    console.rule("[bold cyan]Liquid Level Detection")

    # Pick input source(s)
    if args.dataset:
        console.log("[bold cyan]📦 Preparing dataset...")
        prepared_files = prepare_plastic_dataset()
        if not prepared_files:
            console.log("[bold red]❌ No dataset images found. Exiting.")
            return
        console.log(f"[green]✓ Found {len(prepared_files)} dataset images")
    elif args.image:
        if not os.path.exists(args.image):
            console.log(f"[bold red]❌ Image not found: {args.image}")
            return
        prepared_files = [args.image]
        console.log(f"[green]✓ Single image mode: {args.image}")
    elif args.folder:
        if not os.path.exists(args.folder):
            console.log(f"[bold red]❌ Folder not found: {args.folder}")
            return
        prepared_files = sorted(glob.glob(os.path.join(args.folder, "*.png")) +
                                glob.glob(os.path.join(args.folder, "*.jpg")) +
                                glob.glob(os.path.join(args.folder, "*.jpeg")))
        if not prepared_files:
            console.log(f"[bold red]❌ No images found in folder {args.folder}")
            return
        console.log(f"[green]✓ Folder mode: Found {len(prepared_files)} images in {args.folder}")
    elif args.live or args.video:
        prepared_files = None
    else:
        console.log("[red]⚠ Please specify a mode: --dataset, --image <path>, --folder <dir>, --live, or --video <path>")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    console.rule("[bold cyan]Loading Segmentation Model")
    net = load_model(MODEL_PATH, USE_GPU, FREEZE_BN)

    if prepared_files is not None:
        # ---------- Image/Folder/Dataset pipeline (segmentation-first) ----------
        console.rule("[bold cyan]Running Inference")
        start_time = time.time()
        for path in track(prepared_files, description="🔮 Processing images"):
            img = cv2.imread(str(path))
            if img is None:
                console.log(f"[red]⚠ Skipping invalid image:[/red] {path}")
                continue
            process_image_with_crop(net, img, path, OUTPUT_DIR)

        elapsed = time.time() - start_time
        console.rule("[bold green]✅ Inference Complete")
        console.log(f"[bold magenta]⏱ Total inference time: {elapsed:.2f} seconds")
    else:
        # ---------- Live/Video pipeline (YOLO → crop-constant → segmentation) ----------
        yolo = load_yolo(YOLO_WEIGHTS)
        source = 0 if args.live else args.video
        run_on_stream(net,
                      yolo_model=yolo,
                      source=source,
                      save=args.save,
                      out_path=os.path.join(OUTPUT_DIR, "output_live.mp4"),
                      refresh=args.refresh)


if __name__ == "__main__":
    main()
