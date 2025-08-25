import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

def crop_detections(image_path, detections, save_dir="crops"):
    """
    Crop the highest-confidence detected object from the image.
    """
    if not detections:  # no detections
        return []

    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")

    # Only keep the highest-confidence detection
    best_det = max(detections, key=lambda d: d["confidence"])
    x1, y1, x2, y2 = best_det["bbox"]
    crop = img.crop((x1, y1, x2, y2))

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}_{best_det['label']}.jpg")
    crop.save(save_path)
    print(f"✂️ Saved best crop: {save_path}")

    return [crop]

def yolo_detect(image_path,
                weights_path="/home/shreyas/Desktop/Techolution/VESNET/checkpoints/yolo11s.pt",
                save_dir="outputs"):
    """
    Run YOLO detection and save annotated image with only the highest-confidence detection.
    """
    model = YOLO(weights_path)
    target_classes = {"cup", "glass", "wine glass"}

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)

    results = model(img_np)[0]
    detections, annotated_frame = [], img_np.copy()

    # Collect detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label.lower() in target_classes:
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": (x1, y1, x2, y2)
            })

    # Keep only the highest confidence one
    if detections:
        best_det = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best_det["bbox"]
        conf = best_det["confidence"]
        label = best_det["label"]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        print(f"✅ Best detection: {label} ({conf}) at {best_det['bbox']}")
        detections = [best_det]  # overwrite with only best one
    else:
        print("⚠️ No target objects detected.")

    # Save annotated image
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}_det.jpg")
    Image.fromarray(annotated_frame).save(save_path)
    print(f"📸 Saved annotated image as {save_path}")

    return detections

# -------------------- Run on Folder --------------------
if __name__ == "__main__":
    image_dir = "/home/shreyas/Desktop/Techolution/VESNET/data/US_data"
    weights_path = "/home/shreyas/Desktop/Techolution/VESNET/checkpoints/yolo11s.pt"

    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(image_dir, fname)
            print(f"\n🔍 Processing {img_path}")
            detections = yolo_detect(img_path, weights_path, save_dir="outputs")
            crop_detections(img_path, detections, save_dir="crops")
