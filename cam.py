import uuid
import asyncio
import websockets
import cv2
import ssl
import numpy as np
import base64
import json
import io
import os
import sys
import pyrealsense2 as rs
import threading
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config
from Config.logger import setup_logger

logger = setup_logger("Camera","logs","info")

class CameraReceiver:
    """
    Class to receive and process image frames from an MQTT WebSocket server.
    """
    _instances = {}
    _locks = {}

    @classmethod
    def get_instance(cls, config_path: str, camera_name: str = "D435I") -> 'CameraReceiver':
        """
        Get or create a singleton instance of CameraReceiver for a specific camera.
        
        Args:
            config_path (str): Path to the configuration YAML file.
            camera_name (str): Name of the camera. Defaults to "D435I".
            
        Returns:
            CameraReceiver: Singleton instance for the specified camera.
        """
        key = f"{camera_name}_{config_path}"
        
        if key not in cls._locks:
            cls._locks[key] = threading.Lock()
            
        with cls._locks[key]:
            if key not in cls._instances or cls._instances[key].websocket is None:
                cls._instances[key] = cls(config_path, camera_name)
                logger.info(f"Created new CameraReceiver instance for {camera_name}")
            else:
                logger.debug(f"Reusing existing CameraReceiver instance for {camera_name}")
                
            return cls._instances[key]

    def __init__(self, config_path: str, camera_name: str = "D435I"):
        """
        Initializes the CameraReceiver with the provided configuration file.
        This should only be called through get_instance().
        """
        if hasattr(self, 'config'):  # Already initialized
            return
            
        try:
            self.camera_name = camera_name
            config = load_config(config_path)
            
            self.camera_config = config.get("Cameras", {})
            
            self.config = self.camera_config.get(camera_name, {})

            websocket_config = self.config.get("websocket", {})
            
            self.websocket_server = websocket_config.get("server", "")
            logger.info(f"Server : {self.websocket_server}")
            self.websocket_topic = websocket_config.get("topic", "")
            
            # Log the loaded configuration values
            logger.debug(f"Loaded websocket_server: {self.websocket_server}")
            logger.debug(f"Loaded websocket_topic: {self.websocket_topic}")
            
            # Corrected save_path access
            self.save_path = self.config.get("save_path", "data/captured_frames")

            self.websocket = None
            self.running = False
            self.display_task = None

            logger.debug(f"CameraReceiver Initialized for {camera_name}")

        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error during initialization: {e}", exc_info=True)
            raise

    async def connect(self) -> bool:
        """
        Async connection with better error handling and non-blocking operations.
        """
        try:
            # Check if already connected
            if hasattr(self, '_connected') and self._connected and self.websocket and not self.websocket.close():
                logger.debug("Camera already connected")
                return True

                
            # Make connection non-blocking
            websocket_url = f"{self.websocket_server}{self.websocket_topic}"
            logger.debug(f"Connecting to WebSocket URL: {websocket_url}")
            
            # Use asyncio-friendly connection with timeout
            try:
                connection_future = asyncio.wait_for(
                    self._connect_websocket(websocket_url),
                    timeout=10.0
                )
                self.websocket = await connection_future
                self._connected = True
                logger.info(f"Successfully connected to WebSocket Server at {websocket_url}")
                return True
            except asyncio.TimeoutError:
                logger.error("WebSocket connection timed out")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return False

    async def _connect_websocket(self, url: str):
        """Helper method for websocket connection."""
        import websockets
        # Create SSL context for WSS connections
        # if url.startswith('wss://'):
        #     ssl_context = ssl.create_default_context()
        #     ssl_context.check_hostname = False
        #     ssl_context.verify_mode = ssl.CERT_NONE
            
            # Alternative: Load your CA certificate if you want proper validation
            # ssl_context = ssl.create_default_context()
            # ssl_context.load_verify_locations('/home/kailas/Workspace/AIHand repos/AIHand-backend/certs/techolution-ca.crt')
            
            # return await websockets.connect(url, ssl=ssl_context)
        # else:
            # Regular WS connection
        return await websockets.connect(url)

    async def _get_intrinsics(self, location: str = "India", camera_name: str = "D435I"):
        """
        Asynchronously get the camera intrinsics from the configuration file.
        """
        try:
            logger.debug(f"Fetching intrinsics for location: {location}, camera: {camera_name}")
            intrinsics = rs.intrinsics()
            
            # Simulate async operation in case this becomes I/O bound in future
            await asyncio.sleep(0)  # No-op for context consistency
            
            color_intrinsics = self.config[location]["intrinsics"]["color"]
            intrinsics.width = 640
            intrinsics.height = 480
            intrinsics.ppx = color_intrinsics.get("ppx", 0)
            intrinsics.ppy = color_intrinsics.get("ppy", 0)
            intrinsics.fx = color_intrinsics.get("fx", 0)
            intrinsics.fy = color_intrinsics.get("fy", 0)
            intrinsics.model = rs.distortion.inverse_brown_conrady
            intrinsics.coeffs = [0, 0, 0, 0, 0]

            logger.debug("Intrinsics extracted successfully.")
            return intrinsics

        except KeyError as e:
            logger.error(f"Missing key in camera configuration: {e}")
            raise

        except Exception as e:
            logger.critical(f"Unexpected error while extracting intrinsics: {e}", exc_info=True)
            raise


    async def decode_frames(self):
        """
        Receives and decodes both color and depth frames from JSON data.
        Supports both old and new frame data formats.

        Returns:
            tuple: (color_frame, depth_frame) if successful, else (None, None)
        """
        if self.websocket is None:
            logger.error("WebSocket connection is not established.")
            return None, None

        try:
            # Add timeout to prevent indefinite waiting
            json_data = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            frame_data = json.loads(json_data)
            # logger.debug("Received frame data.")

            # Check what format we're dealing with
            is_new_format = "rgb" in frame_data and "raw_depth" in frame_data
            is_old_format = "color" in frame_data and "depth" in frame_data
            is_new_format = True
            if not (is_new_format or is_old_format):
                logger.warning(f"Incomplete frame data received. Keys: {frame_data.keys()}")
                return None, None

            # Decode color frame (handle both formats)
            if is_new_format:
                color_data = base64.b64decode(frame_data.get("rgb", ""))
            else:
                color_data = base64.b64decode(frame_data.get("color", ""))
                
            color_arr = np.frombuffer(color_data, np.uint8)
            color_frame = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
            # logger.debug("Color frame decoded.")

            # Decode depth frame
            logger.debug("Starting depth frame decoding")
            logger.debug(f"Frame data keys: {list(frame_data.keys())}")

            if is_new_format:
                # New format - decode raw depth data
                logger.debug("Using new format with raw_depth")
                raw_depth_data = base64.b64decode(frame_data.get("raw_depth", ""))
                
                # Get dimensions from metadata
                width = frame_data.get("width", 640)
                height = frame_data.get("height", 480)
                
                # Convert raw bytes to numpy array (uint16 for depth data)
                try:
                    depth_frame = np.frombuffer(raw_depth_data, dtype=np.uint16).reshape((height, width))
                    logger.debug(f"Depth frame decoded successfully: shape={depth_frame.shape}, dtype={depth_frame.dtype}")
                except Exception as e:
                    logger.error(f"Error reshaping depth data: {e}")
                    return None, None
            else:
                # Old format - load pickled numpy array
                logger.debug("Using old format with pickled depth data")
                try:
                    depth_data = base64.b64decode(frame_data.get("depth", ""))
                    depth_bytes = io.BytesIO(depth_data)
                    depth_frame = np.load(depth_bytes, allow_pickle=True)
                    logger.debug(f"Depth frame decoded successfully: shape={depth_frame.shape}, dtype={depth_frame.dtype}")
                except Exception as e:
                    logger.error(f"Error unpickling depth data: {e}")
                    return None, None

            logger.debug("Depth frame decoding completed")
            return color_frame, depth_frame
            
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for frame data")
            return None, None
        except (json.JSONDecodeError, KeyError, ValueError, cv2.error) as e:
            logger.error(f"Decoding error: {e}")
            return None, None
        except Exception as e:
            logger.critical(f"Unexpected error in decode_frames: {e}", exc_info=True)
            return None, None

    async def frames(self):
        """
        Generator that yields decoded color and depth frames from the WebSocket.
        Supports both old and new frame data formats.
        
        Yields:
            tuple: (color_frame, depth_frame) for each received frame
        """
        if self.websocket is None:
            logger.error("WebSocket connection is not established.")
            return

        try:
            while self.running:
                try:
                    # Add timeout to prevent indefinite waiting
                    frame_data = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    frame_data = json.loads(frame_data)

                    # Check what format we're dealing with
                    is_new_format = "rgb" in frame_data and "raw_depth" in frame_data
                    is_old_format = "color" in frame_data and "depth" in frame_data
                    
                    if not (is_new_format or is_old_format):
                        logger.warning(f"Incomplete frame data received. Keys: {frame_data.keys()}")
                        continue

                    # Decode color frame (handle both formats)
                    if is_new_format:
                        color_data = base64.b64decode(frame_data.get("rgb", ""))
                    else:
                        color_data = base64.b64decode(frame_data.get("color", ""))
                        
                    color_arr = np.frombuffer(color_data, np.uint8)
                    color_frame = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)

                    # Decode depth frame based on format
                    if is_new_format:
                        # New format - decode raw depth data
                        raw_depth_data = base64.b64decode(frame_data.get("raw_depth", ""))
                        
                        # Get dimensions from metadata
                        width = frame_data.get("width", 640)
                        height = frame_data.get("height", 480)
                        
                        # Convert raw bytes to numpy array (uint16 for depth data)
                        depth_frame = np.frombuffer(raw_depth_data, dtype=np.uint16).reshape((height, width))
                    else:
                        # Old format - load pickled numpy array
                        depth_data = base64.b64decode(frame_data.get("depth", ""))
                        depth_frame = np.load(io.BytesIO(depth_data), allow_pickle=True)

                    yield color_frame, depth_frame
                    
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for frame data in frames generator")
                    continue
                except (json.JSONDecodeError, KeyError, ValueError, cv2.error) as e:
                    logger.error(f"Error decoding frame in generator: {e}")
                    continue
                    
        except (websockets.exceptions.ConnectionClosed, Exception) as e:
            logger.warning(f"WebSocket connection closed: {e}")
            await self.cleanup()

    async def cleanup(self):
        """
        Closes the WebSocket connection and releases resources.
        """
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket connection closed.")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
        self.running = False

    async def display_async(self):
        """
        Connects to the WebSocket server and displays the received frames.
        """
        connection_success = await self.connect()

        if not connection_success:
            logger.error("Failed to connect to WebSocket server.")
            return {"error": "Failed to connect to WebSocket server."}

        if self.websocket:
            async for color_frame, depth_frame in self.frames():
                if color_frame is not None:
                    cv2.imshow("Color Frame", color_frame)
                if depth_frame is not None:
                    normalized_depth = cv2.normalize(
                        depth_frame, None, 0, 255, cv2.NORM_MINMAX
                    )
                    normalized_depth = normalized_depth.astype(np.uint8)
                    cv2.imshow("Depth Frame", normalized_depth)
                if (cv2.waitKey(1) & 0xFF == ord("q")) or not self.running:
                    break

        cv2.destroyAllWindows()
        await self.cleanup()
        logger.info("Display stopped.")
        return {"message": "Display stopped."}

    async def stop_display_async(self):
        """
        Stops the display by setting running flag to False and cleaning up.
        """
        self.running = False
        cv2.destroyAllWindows()
        await self.cleanup()
        logger.info("Display stopped asynchronously.")
        return {"message": "Display stopped asynchronously."}

    async def capture_frame(self, retries: int = 3, timeout: float = 10.0) -> dict:
        """
        Captures a single frame from the WebSocket stream with retries and timeout.

        Args:
            retries (int): Number of retry attempts. Default is 3.
            timeout (float): Timeout in seconds for connection and frame capture. Default is 10.0.

        Returns:
            dict: A dictionary containing paths to the 'rgb' and 'depth' frame files.
        """
        attempt = 0
        id = uuid.uuid4()
        rgb_dir = f"{self.save_path}/{id}/rgb"
        depth_dir = f"{self.save_path}/{id}/depth"

        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        logger.debug(
            f"Directories created:\n"
            f"RGB Directory: {rgb_dir}\n"
            f"Depth Directory: {depth_dir}"
        )

        while attempt < retries:
            attempt += 1
            logger.info(f"Attempt {attempt}/{retries} to capture frame...")
            self.running = True

            try:
                # Use timeout correctly with asyncio.wait_for
                connection_task = self.connect()
                connection_success = await asyncio.wait_for(connection_task, timeout)

                if not connection_success:
                    logger.error("Failed to connect to WebSocket server.")
                    raise ConnectionError("Failed to connect to WebSocket server.")

                if not self.websocket:
                    logger.error("WebSocket connection is not established after successful connect().")
                    raise RuntimeError("WebSocket connection is not established.")

                decode_task = self.decode_frames()
                color_frame, depth_frame = await asyncio.wait_for(decode_task, timeout)
                if color_frame is None or depth_frame is None:
                    logger.error("Received None for color or depth frame.")
                    raise ValueError("Captured frame is None.")

                color_frame_path = f"{rgb_dir}/image_0.jpg"
                depth_frame_path = f"{depth_dir}/image_0.npy"

                # Run file I/O operations asynchronously
                loop = asyncio.get_event_loop()
                await asyncio.gather(
                    loop.run_in_executor(None, cv2.imwrite, color_frame_path, color_frame),
                    loop.run_in_executor(None, np.save, depth_frame_path, depth_frame)
                )
                logger.info(f"Saved: {color_frame_path}, {depth_frame_path}")

                await self.cleanup()
                return {
                    "status": "success",
                    "color_frame_path": color_frame_path,
                    "depth_frame_path": depth_frame_path,
                }

            except (asyncio.TimeoutError, ConnectionError) as e:
                logger.warning(f"Connection/Timeout error: {e}")
                await self.cleanup()
            except ValueError as ve:
                logger.error(f"Frame error: {ve}")
                await self.cleanup()
            except Exception as e:
                logger.critical(f"Unexpected error in capture_frame: {e}", exc_info=True)
                await self.cleanup()

            # Exponential backoff for retries
            await asyncio.sleep(min(2 ** attempt, 10))

        logger.warning("Failed to capture frame after multiple retries.")
        return {"status": "failed", "color_frame_path": None, "depth_frame_path": None}

    async def capture_temporal_frames(self, retries: int = 3, timeout: float = 10.0, frames: int = 5) -> dict:
        """
        Captures multiple frames from the WebSocket stream, computes median depth frame, and returns a single color + depth frame.

        Args:
            retries (int): Number of retry attempts. Default is 3.
            timeout (float): Timeout in seconds for connection and frame capture. Default is 10.0.
            frames (int): Number of frames to capture. Default is 5.

        Returns:
            dict: A dictionary with 'color_frame_path' and 'depth_frame_path' for the median result.
        """
        attempt = 0
        id = uuid.uuid4()
        rgb_dir = f"{self.save_path}/{id}/rgb"
        depth_dir = f"{self.save_path}/{id}/depth"

        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        logger.debug(f"Directories created:\nRGB Directory: {rgb_dir}\nDepth Directory: {depth_dir}")

        while attempt < retries:
            attempt += 1
            logger.info(f"Attempt {attempt}/{retries} to capture {frames} frames...")
            self.running = True

            try:
                # Increase timeout for connection
                connection_task = self.connect()
                connection_success = await asyncio.wait_for(connection_task, timeout)

                if not connection_success:
                    logger.error("Failed to connect to WebSocket server.")
                    raise ConnectionError("Failed to connect to WebSocket server.")
                    
                if not self.websocket:
                    logger.error("WebSocket connection is not established after successful connect().")
                    raise RuntimeError("WebSocket connection is not established.")

                color_frames = []
                depth_frames = []

                for i in range(frames):
                    logger.debug(f"Trying to capture frame {i+1}/{frames}")
                    decode_task = self.decode_frames()
                    color_frame, depth_frame = await asyncio.wait_for(decode_task, timeout)

                    if color_frame is None or depth_frame is None:
                        logger.error(f"Captured frame {i+1} is None.")
                        raise ValueError(f"Captured frame {i+1} is None.")

                    color_frames.append(color_frame)
                    depth_frames.append(depth_frame)
                    logger.debug(f"Successfully captured frame {i+1}")
                    # Add a small delay between frame captures
                    await asyncio.sleep(0.2)

                await self.cleanup()

                # Compute median depth
                try:
                    median_depth = np.median(np.stack(depth_frames, axis=0), axis=0).astype(np.uint16)
                except Exception as e:
                    logger.error(f"Error calculating median depth: {e}")
                    raise

                # Use the first color frame as representative
                final_color_frame = color_frames[0]

                # Save median results
                color_frame_path = f"{rgb_dir}/image_final.jpg"
                depth_frame_path = f"{depth_dir}/image_final.npy"

                loop = asyncio.get_event_loop()
                await asyncio.gather(
                    loop.run_in_executor(None, cv2.imwrite, color_frame_path, final_color_frame),
                    loop.run_in_executor(None, np.save, depth_frame_path, median_depth)
                )
                logger.info(f"Saved: {color_frame_path}, {depth_frame_path}")


                logger.info(f"Saved median color frame: {color_frame_path}")
                logger.info(f"Saved median depth frame: {depth_frame_path}")

                return {
                    "status": "success",
                    "color_frame_path": color_frame_path,
                    "depth_frame_path": depth_frame_path
                }

            except (asyncio.TimeoutError, ConnectionError) as e:
                logger.warning(f"Connection/Timeout error: {e}")
                await self.cleanup()
            except ValueError as ve:
                logger.error(f"Frame error: {ve}")
                await self.cleanup()
            except Exception as e:
                logger.critical(f"Unexpected error in capture_temporal_frames: {e}", exc_info=True)
                await self.cleanup()

            # Exponential backoff for retries
            retry_delay = min(2 ** attempt, 10)
            logger.debug(f"Waiting {retry_delay} seconds before next retry")
            await asyncio.sleep(retry_delay)

        logger.warning(f"Failed to capture {frames} frames after multiple retries.")
        return {
            "status": "failed",
            "color_frame_path": None,
            "depth_frame_path": None
        }

    async def capture_temporal_frames_streaming(self, frames: int = 5, max_wait_time: float = 3.0) -> dict:
        """
        Ultra-fast streaming version that captures frames as they arrive.
        
        Args:
            frames (int): Number of frames to capture
            max_wait_time (float): Maximum time to wait for all frames
            
        Returns:
            dict: Result with frame paths
        """
        import time
        from collections import deque
        
        id = uuid.uuid4()
        rgb_dir = f"{self.save_path}/{id}/rgb"
        depth_dir = f"{self.save_path}/{id}/depth"
        
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        try:
            # Ensure connection
            if not self.websocket or self.websocket.close():
                connection_success = await self.connect()
                if not connection_success:
                    raise ConnectionError("Failed to connect to WebSocket server.")
            
            self.running = True
            color_frames = deque(maxlen=frames)
            depth_frames = deque(maxlen=frames)
            
            start_time = time.time()
            frames_captured = 0
            
            # Use the existing frames generator for streaming
            async for color_frame, depth_frame in self.frames():
                if color_frame is not None and depth_frame is not None:
                    color_frames.append(color_frame)
                    depth_frames.append(depth_frame)
                    frames_captured += 1
                    
                    logger.debug(f"Captured frame {frames_captured}/{frames}")
                    
                    # Check if we have enough frames
                    if frames_captured >= frames:
                        break
                        
                    # Yield control frequently
                    await asyncio.sleep(0.001)
                
                # Timeout check
                if time.time() - start_time > max_wait_time:
                    logger.warning(f"Timeout: Only captured {frames_captured} frames in {max_wait_time}s")
                    break
            
            # Stop the streaming
            self.running = False
            
            if len(depth_frames) < 2:
                raise ValueError(f"Only captured {len(depth_frames)} frames, need at least 2")
            
            # Quick median computation
            median_depth = np.median(np.array(list(depth_frames)), axis=0).astype(np.uint16)
            final_color_frame = color_frames[0]
            
            # Save files
            color_frame_path = f"{rgb_dir}/image_final.jpg"
            depth_frame_path = f"{depth_dir}/image_final.npy"
            
            # Use asyncio for file operations
            loop = asyncio.get_event_loop()
            
            await asyncio.gather(
                loop.run_in_executor(None, cv2.imwrite, color_frame_path, final_color_frame),
                loop.run_in_executor(None, np.save, depth_frame_path, median_depth)
            )
            
            capture_time = time.time() - start_time
            logger.info(f"Captured {frames_captured} frames in {capture_time:.2f}s")
            
            return {
                "status": "success",
                "color_frame_path": color_frame_path,
                "depth_frame_path": depth_frame_path,
                "frames_captured": frames_captured,
                "capture_time": capture_time
            }
            
        except Exception as e:
            logger.error(f"Error in streaming capture: {e}")
            return {
                "status": "failed",
                "color_frame_path": None,
                "depth_frame_path": None,
                "error": str(e)
            }

    
    async def reset_state(self):
        """Reset all camera state between captures"""
        try:
            # Reset internal caches and flags
            self._frame_cache = {}
            self._coordinate_cache = {}
            self._last_capture_time = 0
            self._last_frame = None
            self._last_depth_frame = None
            self._last_coordinates = None
            
            # Reset camera connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            # Wait for cleanup
            await asyncio.sleep(0.1)
            
            # Reinitialize connection
            await self.connect()
            
            logger.info(f"Camera {self.camera_name} state reset complete")
            return True
        except Exception as e:
            logger.error(f"Error resetting camera state: {e}")
            return False

    async def start_display(self):
        """
        Asynchronously starts the display process.
        This method sets the running flag to True and then calls the display_async method
        to handle the display functionality.
        Returns:
            None
        """
        self.running = True
        await self.display_async()

    def start(self, mode="display"):
        """
        Starts the camera receiver in the specified mode.
        Args:
            mode (str): The mode in which to start the camera receiver.
                        Options are "display" to start displaying the camera feed,
                        or "capture" to capture a single frame. Default is "display".
        Returns:
            If mode is "capture", returns the captured frame.
            If mode is "display", runs the display loop until stopped.
        """
        try:
            self.running = True
            if mode == "display":
                asyncio.run(self.start_display())
            elif mode == "capture":
                return asyncio.run(self.capture_frame())
        except Exception as e:
            logger.error(f"Error occurred in start method: {e}", exc_info=True)
            self.running = False
            return {"error": str(e)}

    async def stop_display_async(self):
        """
        Asynchronously stops the display and closes the websocket connection.
        This method sets the running flag to False, closes the websocket connection
        if it exists, and destroys all OpenCV windows.
        Returns:
            None
        """
        self.running = False
        if self.websocket:
            await self.websocket.close()
        cv2.destroyAllWindows()

    def stop(self):
        """
        Stops the camera receiver by setting the running flag to False.

        Returns:
            dict: Status message
        """
        try:
            if not self.running:
                logger.info("Camera receiver is not running.")
                return {"message": "Camera receiver is not running."}

            self.running = False
            if hasattr(self, "display_thread") and self.display_thread.is_alive():
                self.display_thread.join(timeout=5.0)

            cv2.destroyAllWindows()
            logger.info("Camera receiver stopped successfully.")
            return {"message": "Camera receiver stopped successfully."}
        except Exception as e:
            logger.error(f"Error occurred while stopping the camera receiver: {e}", exc_info=False)
            return {"error": f"Failed to stop the camera receiver: {str(e)}"}

    def capture_frame_sync(self, timeout: int = 60) -> Dict[str, Any]:
        """
        Synchronous version of capture_frame for use in thread pools.
        This prevents blocking the async event loop.
        
        Args:
            timeout (int): Timeout in seconds for frame capture
            
        Returns:
            Dict[str, Any]: Frame capture results
        """
        try:
            # Run the async method in a new event loop
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.capture_frame(timeout=timeout))
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Sync frame capture failed: {e}")
            return {"status": "failed", "error": str(e)}
