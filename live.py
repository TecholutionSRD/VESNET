# ------------------------------- Imports --------------------------------
import os
import torch
import numpy as np
import cv2
from rich.console import Console
import time
import asyncio
import threading
from src.model import SegModel as FCN
import src.CategoryDictionary as CatDic
from camera_receiver import CameraReceiver


# ------------------------------- Setup ----------------------------------
console = Console()

MODEL_PATH = "checkpoints/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"
USE_GPU = True
FREEZE_BN = False
MAX_SIZE = 840

LEVEL_TO_PERCENT = {
    'Empty': 0, 'Level10': 10, 'Level20': 20, 'Level30': 30,
    'Level40': 40, 'Level50': 50, 'Level60': 60, 'Level70': 70,
    'Level80': 80, 'Level90': 90, 'Full': 100
}


# ------------------------------- Functions -------------------------------
def load_model(model_path: str, use_gpu: bool, freeze_bn: bool) -> FCN:
    """Load the segmentation model"""
    net = FCN(CatDic.CatNum)
    if use_gpu and torch.cuda.is_available():
        console.log("[bold green]✓ Using GPU")
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        console.log("[bold yellow]⚠ Using CPU")
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        net.cpu()

    if freeze_bn:
        net.eval()
        console.log("[cyan]BatchNorm statistics frozen")
    return net


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image array"""
    if img is None:
        return None
    h, w, _ = img.shape
    r = max(h, w)
    if r > MAX_SIZE:
        fr = MAX_SIZE / r
        img = cv2.resize(img, (int(w * fr), int(h * fr)))
    return img


def determine_liquid_level(predictions: dict) -> tuple:
    """Determine liquid level from predictions"""
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


def create_display_image(img: np.ndarray, predictions: dict, liquid_percent: float, threshold: float) -> np.ndarray:
    """Create display image with overlay and information"""
    h, w, _ = img.shape
    detected_level, level_percent, cont_percent, ratio = determine_liquid_level(predictions)

    # Extract masks
    vessel_mask = predictions.get("Vessel").data.cpu().numpy()[0].astype(np.uint8) if "Vessel" in predictions else None
    liquid_mask = predictions.get("Liquid GENERAL").data.cpu().numpy()[0].astype(np.uint8) if "Liquid GENERAL" in predictions else None
    liquid_inside_vessel = None
    if vessel_mask is not None and liquid_mask is not None:
        liquid_inside_vessel = liquid_mask * (vessel_mask > 0)

    # Create overlay
    display_img = img.copy()
    if liquid_inside_vessel is not None:
        Lb_resized = cv2.resize(liquid_inside_vessel, (w, h), interpolation=cv2.INTER_NEAREST)
        if Lb_resized.mean() > 0.001:
            # Color based on threshold proximity
            if cont_percent >= threshold:
                color = (0, 255, 0)  # Green when above threshold
            elif cont_percent >= threshold * 0.8:
                color = (0, 255, 255)  # Yellow when close to threshold
            else:
                color = (0, int(255 * (cont_percent / 100)), int(255 * (1 - cont_percent / 100)))  # Blue to red gradient
            
            display_img[Lb_resized == 1] = color

    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(w, h) / 400.0)
    thickness = max(1, int(font_scale * 2))
    
    # Main liquid level text
    text1 = f"Liquid Level: {cont_percent:.1f}%"
    text2 = f"Threshold: {threshold:.1f}%"
    text3 = f"Status: {'COMPLETE' if cont_percent >= threshold else 'IN PROGRESS'}"
    
    # Calculate text positions
    y_start = int(30 * font_scale)
    y_spacing = int(35 * font_scale)
    
    # Add background rectangles for better text visibility
    texts = [text1, text2, text3]
    for i, text in enumerate(texts):
        y_pos = y_start + (i * y_spacing)
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(display_img, (5, y_pos - text_h - 5), (text_w + 15, y_pos + 5), (0, 0, 0), -1)
        
        # Text color based on status
        if i == 2:  # Status text
            text_color = (0, 255, 0) if cont_percent >= threshold else (255, 255, 255)
        else:
            text_color = (255, 255, 255)
        
        cv2.putText(display_img, text, (10, y_pos), font, font_scale, text_color, thickness)
    
    # Add progress bar
    bar_x, bar_y = 10, h - 60
    bar_w, bar_h = w - 20, 30
    
    # Background bar
    cv2.rectangle(display_img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    
    # Progress fill
    progress_w = int((cont_percent / 100.0) * bar_w)
    if cont_percent >= threshold:
        progress_color = (0, 255, 0)  # Green
    elif cont_percent >= threshold * 0.8:
        progress_color = (0, 255, 255)  # Yellow
    else:
        progress_color = (0, 100, 255)  # Orange
    
    cv2.rectangle(display_img, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), progress_color, -1)
    
    # Threshold line
    threshold_x = bar_x + int((threshold / 100.0) * bar_w)
    cv2.line(display_img, (threshold_x, bar_y), (threshold_x, bar_y + bar_h), (255, 255, 255), 2)
    
    # Progress text
    progress_text = f"{cont_percent:.1f}%"
    (prog_w, prog_h), _ = cv2.getTextSize(progress_text, font, font_scale * 0.8, thickness)
    text_x = bar_x + (bar_w - prog_w) // 2
    text_y = bar_y + (bar_h + prog_h) // 2
    cv2.putText(display_img, progress_text, (text_x, text_y), font, font_scale * 0.8, (255, 255, 255), thickness)

    return display_img


def get_vessel_bbox(predictions: dict, min_area: int = 500) -> tuple:
    """Get vessel bounding box from predictions"""
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
    """Run inference on image"""
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
    """Resize while keeping aspect ratio, then pad to target size"""
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


def process_live_frame(net, img: np.ndarray, threshold: float) -> dict:
    """Process a single live frame and return results"""
    try:
        # Preprocess the image
        preprocessed_img = preprocess_image(img)
        if preprocessed_img is None:
            return {"error": "Failed to preprocess image"}

        # First pass inference
        first_pass_preds = run_inference_on_image(net, preprocessed_img, use_gpu=USE_GPU, freeze_bn=FREEZE_BN)
        bbox = get_vessel_bbox(first_pass_preds)
        
        if bbox is None:
            return {"error": "No vessel detected", "liquid_percent": 0.0, "display_image": img}

        # Crop and resize
        x1, y1, x2, y2 = bbox
        cropped = preprocessed_img[y1:y2, x1:x2]
        cropped_resized = resize_with_aspect_ratio(cropped, target_size=(224, 224))

        # Second pass inference
        second_pass_preds = run_inference_on_image(net, cropped_resized, use_gpu=USE_GPU, freeze_bn=FREEZE_BN)
        
        # Get liquid level
        detected_level, level_percent, cont_percent, ratio = determine_liquid_level(second_pass_preds)
        
        # Create display image
        display_image = create_display_image(cropped_resized, second_pass_preds, cont_percent, threshold)
        
        return {
            "status": "success",
            "detected_level": detected_level,
            "level_percent": level_percent,
            "liquid_percent": cont_percent,
            "liquid_ratio": ratio,
            "display_image": display_image,
            "threshold_reached": cont_percent >= threshold
        }
        
    except Exception as e:
        return {"error": str(e), "liquid_percent": 0.0, "display_image": img}


class LiveLiquidDetector:
    """Thread-safe live liquid level detector with CV2 display and threshold monitoring"""
    
    def __init__(self, config_path: str, camera_name: str = "D435I", 
                 capture_interval: float = 1.0, threshold: float = 80.0,
                 window_name: str = "Live Liquid Detection"):
        self.config_path = config_path
        self.camera_name = camera_name
        self.capture_interval = capture_interval
        self.threshold = threshold
        self.window_name = window_name
        self.running = False
        self.completed = False
        self.thread = None
        self.net = None
        self.camera = None
        self.completion_callback = None
        
    def set_completion_callback(self, callback):
        """Set callback function called when threshold is reached"""
        self.completion_callback = callback
    
    def _load_model(self):
        """Load model in thread"""
        try:
            self.net = load_model(MODEL_PATH, USE_GPU, FREEZE_BN)
            return True
        except Exception as e:
            console.log(f"[red]❌ Failed to load model: {e}")
            return False
    
    async def _detection_loop(self):
        """Main detection loop with CV2 display"""
        try:
            # Initialize camera
            self.camera = CameraReceiver.get_instance(self.config_path, self.camera_name)
            
            # Connect to camera
            connection_success = await self.camera.connect()
            if not connection_success:
                console.log("[red]❌ Failed to connect to camera")
                return
            
            console.log(f"[green]✓ Live liquid detection started (Threshold: {self.threshold}%)")
            console.log("[yellow]Press 'q' to quit or 'r' to reset")
            
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_RESIZABLE)
            
            frame_count = 0
            
            while self.running and not self.completed:
                try:
                    # Get live frame from camera
                    color_frame, depth_frame = await self.camera.decode_frames()
                    
                    if color_frame is None:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Process the frame
                    result = process_live_frame(self.net, color_frame, self.threshold)
                    
                    # Display the result
                    if "display_image" in result:
                        cv2.imshow(self.window_name, result["display_image"])
                    
                    # Check for completion
                    if result.get("threshold_reached", False) and not self.completed:
                        self.completed = True
                        liquid_percent = result.get("liquid_percent", 0)
                        
                        console.log(f"[bold green]🎉 COMPLETION ACHIEVED! 🎉")
                        console.log(f"[bold green]Liquid level reached {liquid_percent:.1f}% (Threshold: {self.threshold}%)")
                        
                        # Call completion callback if provided
                        if self.completion_callback:
                            try:
                                self.completion_callback(result)
                            except Exception as e:
                                console.log(f"[yellow]⚠ Callback error: {e}")
                        
                        # Show completion message on image for a few seconds
                        await asyncio.sleep(3.0)
                        break
                    
                    # Handle key presses (non-blocking)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        console.log("[yellow]User requested quit")
                        break
                    elif key == ord('r'):
                        console.log("[cyan]Resetting completion status")
                        self.completed = False
                    
                    frame_count += 1
                    
                    # Wait for next capture
                    await asyncio.sleep(self.capture_interval)
                    
                except Exception as e:
                    console.log(f"[red]❌ Error in detection loop: {e}")
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            console.log(f"[red]❌ Fatal error in detection loop: {e}")
        finally:
            # Cleanup
            cv2.destroyWindow(self.window_name)
            if self.camera:
                await self.camera.cleanup()
            
            if self.completed:
                console.log("[bold green]✅ Detection completed successfully!")
            else:
                console.log("[yellow]Live liquid detection stopped")
    
    def _run_async_loop(self):
        """Run async loop in thread"""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._detection_loop())
        finally:
            loop.close()
    
    def start(self):
        """Start detection in background thread"""
        if self.running:
            console.log("[yellow]⚠ Detection already running")
            return False
        
        # Load model
        if not self._load_model():
            return False
        
        self.running = True
        self.completed = False
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        
        console.log("[green]✓ Live liquid detection started in background")
        return True
    
    def stop(self):
        """Stop detection"""
        if not self.running:
            console.log("[yellow]⚠ Detection not running")
            return
        
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        console.log("[green]✓ Live liquid detection stopped")
    
    def is_running(self):
        """Check if detection is running"""
        return self.running and self.thread and self.thread.is_alive()
    
    def is_completed(self):
        """Check if threshold has been reached"""
        return self.completed
    
    def reset(self):
        """Reset completion status"""
        self.completed = False
        console.log("[cyan]Completion status reset")


# ------------------------------- Main Function -------------------------------
def start_live_detection(config_path: str, camera_name: str = "D435I", 
                        capture_interval: float = 1.0, threshold: float = 80.0,
                        window_name: str = "Live Liquid Detection",
                        completion_callback=None):
    """
    Start live liquid level detection with CV2 display and automatic completion
    
    Args:
        config_path: Path to camera configuration file
        camera_name: Camera name in configuration
        capture_interval: Seconds between captures
        threshold: Liquid level threshold for completion (percentage)
        window_name: CV2 window name
        completion_callback: Optional callback function called when threshold reached
        
    Returns:
        LiveLiquidDetector: Detector instance for control
    """
    detector = LiveLiquidDetector(config_path, camera_name, capture_interval, 
                                threshold, window_name)
    
    if completion_callback:
        detector.set_completion_callback(completion_callback)
    
    if detector.start():
        return detector
    else:
        return None


# ------------------------------- Example Usage -------------------------------
if __name__ == "__main__":
    # Example completion callback
    def on_completion(result):
        print(f"🎉 TASK COMPLETED!")
        print(f"Final liquid level: {result['liquid_percent']:.1f}%")
        print(f"Threshold was: {80.0}%")
    
    # Start detection
    detector = start_live_detection(
        config_path="config/camera_config.yaml",
        camera_name="D435I",
        capture_interval=1.0,
        threshold=80.0,
        completion_callback=on_completion
    )
    
    if detector:
        try:
            # Wait for completion or user interruption
            while detector.is_running() and not detector.is_completed():
                time.sleep(1)
            
            if detector.is_completed():
                print("Detection completed automatically!")
            
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            detector.stop()
    else:
        print("Failed to start detector")