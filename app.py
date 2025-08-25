# # streamlit_app.py
# import streamlit as st
# import cv2
# import asyncio
# import numpy as np
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Camera.functions.camera_receiver import CameraReceiver

# # Page setup
# st.set_page_config(page_title="Camera Live Feed", layout="wide")
# st.title("📹 Live Camera Feed")

# # Sidebar controls
# config_path = st.sidebar.text_input("Config Path", "camera_config.yaml")
# camera_name = st.sidebar.text_input("Camera Name", "D435I")

# # Create camera instance
# camera = CameraReceiver.get_instance(config_path, camera_name)

# # Streamlit placeholders
# frame_placeholder = st.empty()

# async def stream_camera():
#     connected = await camera.connect()
#     if not connected:
#         st.error("❌ Failed to connect to WebSocket server")
#         return

#     camera.running = True
#     async for color_frame, depth_frame in camera.frames():
#         if color_frame is not None:
#             # Convert BGR → RGB for Streamlit
#             rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

#         if not camera.running:
#             break

#     await camera.cleanup()

# def run_streamlit_loop():
#     try:
#         asyncio.run(stream_camera())
#     except Exception as e:
#         st.error(f"⚠️ Error: {e}")

# # Button controls
# col1, col2 = st.columns(2)
# if col1.button("▶️ Start Stream"):
#     run_streamlit_loop()
# if col2.button("⏹️ Stop Stream"):
#     asyncio.run(camera.stop_display_async())
#     st.info("Stream stopped")

# streamlit_liquid_detection.py
import streamlit as st
import cv2
import asyncio
import numpy as np
import os
import sys
import torch
import json
import time
import glob
from datetime import datetime
import tempfile
import base64
from io import BytesIO
from PIL import Image
import pandas as pd

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ultralytics import YOLO
    from Camera.functions.camera_receiver import CameraReceiver
    from src.model import SegModel as FCN
    import src.CategoryDictionary as CatDic
    from src.prepare_kaggle_data import prepare_plastic_dataset
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ------------------------------- Configuration -------------------------------
DEFAULT_INPUT_DIR = "data/input_data/"
OUTPUT_DIR = "data/output_data/"
MODEL_PATH = "checkpoints/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"
YOLO_WEIGHTS = "checkpoints/yolo11s.pt"
CAMERA_CONFIG_PATH = "Config/config.yaml"

USE_GPU = True
FREEZE_BN = False
MAX_SIZE = 840
TARGET_CLASSES = {"cup", "glass", "wine glass"}

LEVEL_TO_PERCENT = {
    'Empty': 0, 'Level10': 10, 'Level20': 20, 'Level30': 30,
    'Level40': 40, 'Level50': 50, 'Level60': 60, 'Level70': 70,
    'Level80': 80, 'Level90': 90, 'Full': 100
}

# ------------------------------- Page Setup -------------------------------
st.set_page_config(
    page_title="🧪 Liquid Level Detection", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Liquid Level Detection in Transparent Objects")
st.markdown("---")

# ------------------------------- Helper Functions -------------------------------
@st.cache_resource
def load_segmentation_model():
    """Load the segmentation model with caching"""
    try:
        net = FCN(CatDic.CatNum)
        if USE_GPU and torch.cuda.is_available():
            net.load_state_dict(torch.load(MODEL_PATH))
            net.cuda()
            st.success("✓ Segmentation model loaded on GPU")
        else:
            net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
            net.cpu()
            st.info("⚠ Segmentation model loaded on CPU")
        
        if FREEZE_BN:
            net.eval()
        return net
    except Exception as e:
        st.error(f"Failed to load segmentation model: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    """Load YOLO model with caching"""
    try:
        yolo = YOLO(YOLO_WEIGHTS)
        st.success("✓ YOLO model loaded")
        return yolo
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

def resize_with_aspect_ratio(image, target_size=(224, 224), pad_color=(0, 0, 0)):
    """Resize while keeping aspect ratio, then pad to target size."""
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas

def run_inference_on_image(net, img: np.ndarray) -> dict:
    """Run segmentation inference on image"""
    if net is None:
        return {}
    
    im_batch = np.expand_dims(img, axis=0)
    with torch.no_grad():
        _, out_lb_dict = net.forward(
            Images=im_batch,
            TrainMode=False,
            UseGPU=USE_GPU,
            FreezeBatchNormStatistics=FREEZE_BN
        )
    return out_lb_dict

def determine_liquid_level(predictions: dict) -> tuple:
    """Determine liquid level from segmentation predictions"""
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

def yolo_best_bbox(yolo_model, frame: np.ndarray, conf_thres: float = 0.25):
    """Get best YOLO detection bbox for target classes"""
    if yolo_model is None:
        return None
        
    try:
        results = yolo_model(frame)[0]
        best = None
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_thres:
                continue
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            if label.lower() in TARGET_CLASSES:
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
                if (best is None) or (conf > best[0]):
                    best = (conf, tuple(xyxy))
        return None if best is None else best[1]
    except Exception as e:
        st.error(f"YOLO inference error: {e}")
        return None

def get_vessel_bbox_from_segmentation(predictions: dict, min_area: int = 500):
    """Get vessel bbox from segmentation mask"""
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

def create_visualization(color_frame, crop_resized, predictions, bbox, cont_percent, nearest):
    """Create comprehensive visualization"""
    if bbox is None:
        return color_frame
    
    h, w = color_frame.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Frame with bbox
    frame_with_bbox = color_frame.copy()
    cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Liquid: {cont_percent:.1f}% (≈{nearest}%)"
    cv2.putText(frame_with_bbox, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Extract masks
    vessel_mask = predictions.get("Vessel")
    liquid_mask = predictions.get("Liquid GENERAL")
    
    if vessel_mask is not None and liquid_mask is not None:
        vessel_mask = vessel_mask.data.cpu().numpy()[0].astype(np.uint8)
        liquid_mask = liquid_mask.data.cpu().numpy()[0].astype(np.uint8)
        
        # Create overlay
        overlay = crop_resized.copy()
        liquid_inside_vessel = liquid_mask * (vessel_mask > 0)
        if liquid_inside_vessel.sum() > 0:
            color = (0, int(255 * (nearest / 100)), int(255 * (1 - nearest / 100)))
            overlay[liquid_inside_vessel == 1] = color
        
        # Add text to overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Level: {cont_percent:.1f}% → {nearest}%"
        cv2.putText(overlay, text, (10, 25), font, 0.6, (255, 255, 255), 2)
        
        return frame_with_bbox, crop_resized, overlay
    
    return frame_with_bbox, crop_resized, crop_resized

def save_results(results_data, filename="results.json"):
    """Save results to JSON file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append(results_data)
    
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    return json_path

# ------------------------------- Sidebar Controls -------------------------------
st.sidebar.title("🎛️ Controls")

# Model loading status
with st.sidebar.expander("📦 Model Status", expanded=True):
    if st.button("🔄 Reload Models"):
        st.cache_resource.clear()
        st.rerun()

# Mode selection
mode = st.sidebar.selectbox(
    "🔧 Select Mode",
    ["📷 Live Camera", "🎬 Video File", "🖼️ Image Upload", "📁 Batch Processing", "📸 Single Capture"]
)

# Camera settings
if mode in ["📷 Live Camera", "📸 Single Capture"]:
    st.sidebar.subheader("📹 Camera Settings")
    config_path = st.sidebar.text_input("Config Path", CAMERA_CONFIG_PATH)
    camera_name = st.sidebar.text_input("Camera Name", "D435I")
    
    # Live camera specific settings
    if mode == "📷 Live Camera":
        refresh_rate = st.sidebar.slider("🔄 YOLO Refresh Rate (frames)", 0, 100, 0)
        show_detailed = st.sidebar.checkbox("🔍 Show Detailed View", True)
        save_stream = st.sidebar.checkbox("💾 Save Stream", False)

# Processing settings
st.sidebar.subheader("⚙️ Processing Settings")
conf_threshold = st.sidebar.slider("🎯 YOLO Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
use_yolo_detection = st.sidebar.checkbox("🔍 Use YOLO Detection", True)

# ------------------------------- Main Content -------------------------------

# Load models
segmentation_model = load_segmentation_model()
yolo_model = load_yolo_model() if use_yolo_detection else None

# ------------------------------- Live Camera Mode -------------------------------
if mode == "📷 Live Camera":
    st.header("📷 Live Camera Stream")
    
    col1, col2, col3 = st.columns(3)
    
    # Control buttons
    start_stream = col1.button("▶️ Start Stream", type="primary")
    stop_stream = col2.button("⏹️ Stop Stream")
    
    # Placeholders for display
    main_placeholder = st.empty()
    info_placeholder = st.empty()
    
    # Session state for camera control
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'camera_instance' not in st.session_state:
        st.session_state.camera_instance = None
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []
    
    if start_stream and not st.session_state.camera_running:
        if segmentation_model is None:
            st.error("Segmentation model not loaded!")
        else:
            st.session_state.camera_running = True
            st.session_state.camera_instance = CameraReceiver.get_instance(config_path, camera_name)
            
            async def run_camera_stream():
                camera = st.session_state.camera_instance
                connected = await camera.connect()
                
                if not connected:
                    st.error("❌ Failed to connect to camera")
                    st.session_state.camera_running = False
                    return
                
                camera.running = True
                bbox = None
                frame_count = 0
                
                try:
                    async for color_frame, depth_frame in camera.frames():
                        if not st.session_state.camera_running:
                            break
                        
                        if color_frame is None:
                            continue
                        
                        # YOLO detection for bbox
                        if use_yolo_detection and (bbox is None or (refresh_rate > 0 and frame_count % refresh_rate == 0)):
                            new_bbox = yolo_best_bbox(yolo_model, color_frame, conf_threshold)
                            if new_bbox is not None:
                                bbox = new_bbox
                        
                        # Process frame
                        if bbox is not None:
                            x1, y1, x2, y2 = bbox
                            crop = color_frame[y1:y2, x1:x2]
                            crop_resized = resize_with_aspect_ratio(crop, target_size=(224, 224))
                            
                            # Segmentation
                            predictions = run_inference_on_image(segmentation_model, crop_resized)
                            _, nearest, cont_percent, ratio = determine_liquid_level(predictions)
                            
                            # Create visualization
                            if show_detailed:
                                main_frame, crop_display, overlay = create_visualization(
                                    color_frame, crop_resized, predictions, bbox, cont_percent, nearest
                                )
                                
                                # Display in columns
                                with main_placeholder.container():
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.image(cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB), 
                                                caption="Live Stream", use_column_width=True)
                                    with col2:
                                        st.image(cv2.cvtColor(crop_display, cv2.COLOR_BGR2RGB), 
                                                caption="Cropped Vessel", use_column_width=True)
                                    with col3:
                                        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                                                caption="Liquid Detection", use_column_width=True)
                            else:
                                main_frame, _, _ = create_visualization(
                                    color_frame, crop_resized, predictions, bbox, cont_percent, nearest
                                )
                                main_placeholder.image(cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB), 
                                                     caption="Live Stream", use_column_width=True)
                            
                            # Update info
                            with info_placeholder.container():
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("🧪 Liquid Level", f"{cont_percent:.1f}%")
                                col2.metric("📊 Nearest Level", f"{nearest}%")
                                col3.metric("📏 Ratio", f"{ratio:.3f}")
                                col4.metric("🖼️ Frame", frame_count)
                            
                            # Store result
                            result_data = {
                                "timestamp": datetime.now().isoformat(),
                                "frame_number": frame_count,
                                "continuous_percent": round(cont_percent, 2),
                                "nearest_percent": nearest,
                                "liquid_ratio": round(ratio, 4)
                            }
                            st.session_state.results_history.append(result_data)
                            
                        else:
                            # No detection
                            main_placeholder.image(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB), 
                                                 caption="Searching for vessel...", use_column_width=True)
                        
                        frame_count += 1
                        await asyncio.sleep(0.033)  # ~30 FPS
                
                except Exception as e:
                    st.error(f"Stream error: {e}")
                finally:
                    await camera.cleanup()
                    st.session_state.camera_running = False
            
            # Run the async function
            try:
                asyncio.run(run_camera_stream())
            except Exception as e:
                st.error(f"Camera stream error: {e}")
    
    if stop_stream:
        st.session_state.camera_running = False
        if st.session_state.camera_instance:
            try:
                asyncio.run(st.session_state.camera_instance.cleanup())
            except:
                pass
        st.info("Stream stopped")
    
    # Results history
    if st.session_state.results_history:
        st.subheader("📊 Recent Results")
        df = pd.DataFrame(st.session_state.results_history[-10:])  # Last 10 results
        st.dataframe(df, use_container_width=True)
        
        if st.button("💾 Save Results History"):
            json_path = save_results(st.session_state.results_history, "live_stream_results.json")
            st.success(f"Results saved to {json_path}")

# ------------------------------- Single Capture Mode -------------------------------
elif mode == "📸 Single Capture":
    st.header("📸 Single Frame Capture")
    
    if st.button("📷 Capture Frame", type="primary"):
        if segmentation_model is None:
            st.error("Segmentation model not loaded!")
        else:
            with st.spinner("Capturing frame..."):
                camera = CameraReceiver.get_instance(config_path, camera_name)
                
                try:
                    result = asyncio.run(camera.capture_frame(retries=3, timeout=10.0))
                    
                    if result["status"] == "success":
                        # Load captured frame
                        img = cv2.imread(result["color_frame_path"])
                        
                        if img is not None:
                            # Process with segmentation first, then crop
                            first_pass_preds = run_inference_on_image(segmentation_model, img)
                            bbox = get_vessel_bbox_from_segmentation(first_pass_preds)
                            
                            if bbox is None and use_yolo_detection and yolo_model:
                                # Fallback to YOLO
                                bbox = yolo_best_bbox(yolo_model, img, conf_threshold)
                            
                            if bbox is not None:
                                x1, y1, x2, y2 = bbox
                                crop = img[y1:y2, x1:x2]
                                crop_resized = resize_with_aspect_ratio(crop, target_size=(224, 224))
                                
                                # Segmentation on crop
                                predictions = run_inference_on_image(segmentation_model, crop_resized)
                                detected_level, nearest, cont_percent, ratio = determine_liquid_level(predictions)
                                
                                # Create visualization
                                main_frame, crop_display, overlay = create_visualization(
                                    img, crop_resized, predictions, bbox, cont_percent, nearest
                                )
                                
                                # Display results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.image(cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB), 
                                            caption="Original with Detection", use_column_width=True)
                                with col2:
                                    st.image(cv2.cvtColor(crop_display, cv2.COLOR_BGR2RGB), 
                                            caption="Cropped Vessel", use_column_width=True)
                                with col3:
                                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                                            caption="Liquid Level Overlay", use_column_width=True)
                                
                                # Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("🧪 Detected Level", detected_level)
                                col2.metric("📊 Continuous %", f"{cont_percent:.1f}%")
                                col3.metric("🎯 Nearest %", f"{nearest}%")
                                col4.metric("📏 Liquid Ratio", f"{ratio:.3f}")
                                
                                # Save option
                                if st.button("💾 Save Result"):
                                    result_data = {
                                        "timestamp": datetime.now().isoformat(),
                                        "image_path": result["color_frame_path"],
                                        "detected_level": detected_level,
                                        "continuous_percent": round(cont_percent, 2),
                                        "nearest_percent": nearest,
                                        "liquid_ratio": round(ratio, 4)
                                    }
                                    json_path = save_results([result_data], "single_capture_results.json")
                                    st.success(f"Result saved to {json_path}")
                            else:
                                st.warning("No vessel detected in captured frame")
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                                        caption="Captured Frame", use_column_width=True)
                        else:
                            st.error("Failed to load captured image")
                    else:
                        st.error(f"Failed to capture frame: {result}")
                        
                except Exception as e:
                    st.error(f"Capture error: {e}")

# ------------------------------- Image Upload Mode -------------------------------
elif mode == "🖼️ Image Upload":
    st.header("🖼️ Image Upload Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing a transparent vessel with liquid"
    )
    
    if uploaded_file is not None:
        # Convert uploaded file to cv2 format
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔬 Analyze Image", type="primary"):
            if segmentation_model is None:
                st.error("Segmentation model not loaded!")
            else:
                with st.spinner("Analyzing image..."):
                    # First pass segmentation
                    first_pass_preds = run_inference_on_image(segmentation_model, img_bgr)
                    bbox = get_vessel_bbox_from_segmentation(first_pass_preds)
                    
                    if bbox is None and use_yolo_detection and yolo_model:
                        # Fallback to YOLO
                        bbox = yolo_best_bbox(yolo_model, img_bgr, conf_threshold)
                    
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        crop = img_bgr[y1:y2, x1:x2]
                        crop_resized = resize_with_aspect_ratio(crop, target_size=(224, 224))
                        
                        # Second pass segmentation
                        predictions = run_inference_on_image(segmentation_model, crop_resized)
                        detected_level, nearest, cont_percent, ratio = determine_liquid_level(predictions)
                        
                        # Create visualization
                        main_frame, crop_display, overlay = create_visualization(
                            img_bgr, crop_resized, predictions, bbox, cont_percent, nearest
                        )
                        
                        # Display results
                        st.subheader("Analysis Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB), 
                                    caption="Detection Result", use_column_width=True)
                        with col2:
                            st.image(cv2.cvtColor(crop_display, cv2.COLOR_BGR2RGB), 
                                    caption="Cropped Vessel", use_column_width=True)
                        with col3:
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                                    caption="Liquid Level", use_column_width=True)
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("🧪 Detected Level", detected_level)
                        col2.metric("📊 Continuous %", f"{cont_percent:.1f}%")
                        col3.metric("🎯 Nearest %", f"{nearest}%")
                        col4.metric("📏 Liquid Ratio", f"{ratio:.3f}")
                        
                    else:
                        st.warning("⚠️ No vessel detected in the image")

# ------------------------------- Video File Mode -------------------------------
elif mode == "🎬 Video File":
    st.header("🎬 Video File Analysis")
    
    uploaded_video = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file containing transparent vessels"
    )
    
    if uploaded_video is not None:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        # Video processing options
        col1, col2 = st.columns(2)
        with col1:
            process_every_n = st.slider("Process every N frames", 1, 30, 5)
        with col2:
            max_frames = st.slider("Max frames to process", 10, 1000, 100)
        
        if st.button("🎬 Process Video", type="primary"):
            if segmentation_model is None:
                st.error("Segmentation model not loaded!")
            else:
                # Process video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Cannot open video file")
                else:
                    progress_bar = st.progress(0)
                    results_container = st.empty()
                    video_results = []
                    
                    frame_idx = 0
                    processed_count = 0
                    bbox = None
                    
                    while processed_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_idx % process_every_n == 0:
                            # YOLO detection
                            if use_yolo_detection and (bbox is None or processed_count % 10 == 0):
                                bbox = yolo_best_bbox(yolo_model, frame, conf_threshold)
                            
                            if bbox is not None:
                                x1, y1, x2, y2 = bbox
                                crop = frame[y1:y2, x1:x2]
                                crop_resized = resize_with_aspect_ratio(crop, target_size=(224, 224))
                                
                                # Segmentation
                                predictions = run_inference_on_image(segmentation_model, crop_resized)
                                detected_level, nearest, cont_percent, ratio = determine_liquid_level(predictions)
                                
                                video_results.append({
                                    "frame": frame_idx,
                                    "timestamp": frame_idx / cap.get(cv2.CAP_PROP_FPS),
                                    "detected_level": detected_level,
                                    "continuous_percent": round(cont_percent, 2),
                                    "nearest_percent": nearest,
                                    "liquid_ratio": round(ratio, 4)
                                })
                            
                            processed_count += 1
                            progress_bar.progress(processed_count / max_frames)
                            
                            # Update results display
                            if video_results:
                                df = pd.DataFrame(video_results)
                                with results_container.container():
                                    st.subheader(f"Processed {len(video_results)} frames")
                                    st.line_chart(df.set_index('frame')[['continuous_percent']])
                                    st.dataframe(df.tail(10), use_container_width=True)