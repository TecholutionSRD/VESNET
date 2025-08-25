import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------- Setup --------------------
st.set_page_config(page_title="YOLO Object Detection", layout="centered")
st.title("🥤 Cup/Glass Detector with YOLO")
st.markdown("Upload an image to detect **cup, glass, or wine glass** using your YOLO model.")

# -------------------- Load Model --------------------
@st.cache_resource
def load_model(weights_path="/home/shreyas/Desktop/Techolution/VESNET/checkpoints/yolo11s.pt"):
    model = YOLO(weights_path)
    return model

model = load_model("/home/shreyas/Desktop/Techolution/VESNET/checkpoints/yolo11s.pt")

# -------------------- Define Target Classes --------------------
# These should match your model's class labels
target_classes = {"cup", "glass", "wine glass"}

# -------------------- Upload Image --------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil)

    st.image(img_pil, caption="🖼️ Uploaded Image", use_column_width=True)

    # -------------------- Run Inference --------------------
    st.subheader("🔍 Detection Result")
    with st.spinner("Running YOLOv8 inference..."):
        results = model(img_np)[0]  # single image

        # Draw filtered boxes
        annotated_frame = img_np.copy()
        detections_found = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label.lower() in target_classes:
                detections_found = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        if detections_found:
            annotated_pil = Image.fromarray(annotated_frame)
            st.image(annotated_pil, caption="📍 Filtered Detections", use_column_width=True)
        else:
            st.warning("⚠️ No target objects (cup, glass, wine glass) detected.")
