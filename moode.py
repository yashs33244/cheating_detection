import os
import cv2
import numpy as np
import time
import argparse
from datetime import datetime
from deepface import DeepFace
from ultralytics import YOLO
import torch
import threading
from queue import Queue
import redis
import json
import uuid
from flask import Flask, request, jsonify
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RTSP URL configuration
RTSP_URL = "rtsp://admin123:admin123@192.168.79.98:554/11"
OUTPUT_DIR = 'mood_frames'
FPS = 60  # Increased for maximum performance

# --- Step 1: Setup paths and arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Mood and Focus Analysis for Classroom Surveillance')
    parser.add_argument('--redis-host', type=str, default='redis', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to process')
    parser.add_argument('--processing-interval', type=float, default=0.5, help='Interval in seconds between processing frames')
    parser.add_argument('--phone-confidence', type=float, default=0.5, help='Confidence threshold for phone detection')
    parser.add_argument('--output-dir', type=str, default='mood_frames', help='Directory to save mood frames')
    return parser.parse_args()

# --- Step 2: Setup camera connection ---
def setup_camera(source):
    """
    Setup camera connection for either RTSP stream or video file/URL
    
    Args:
        source: RTSP URL, local file path, or public URL
        
    Returns:
        cv2.VideoCapture object
    """
    # Check if source is a URL (including S3)
    is_url = bool(re.match(r'https?://', source))
    is_rtsp = bool(re.match(r'rtsp://', source))
    
    logger.info(f"Connecting to video source: {source}")
    
    # For RTSP streams, we need to set additional parameters
    if is_rtsp:
        logger.info("Detected RTSP stream, setting up with RTSP parameters")
        # Set RTSP transport to TCP for better reliability
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size for lower latency
        
        # Set additional RTSP parameters for better connection
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
        
        # Try to connect with a timeout
        start_time = time.time()
        timeout = 10  # 10 seconds timeout
        
        while not cap.isOpened() and time.time() - start_time < timeout:
            logger.info("Waiting for RTSP connection...")
            time.sleep(1)
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        # For URLs or local files
        if is_url:
            logger.info(f"Processing video from URL: {source}")
        else:
            # For local files, check if they exist
            if not os.path.exists(source):
                raise Exception(f"File not found: {source}")
            logger.info(f"Processing local video file: {source}")
        
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise Exception(f"Failed to open video source: {source}")
    
    logger.info("Successfully connected to video source")
    return cap

class MoodAnalyzer:
    def __init__(self, rtsp_url=None, video_path=None, output_dir='mood_frames', fps=60):
        self.rtsp_url = rtsp_url or RTSP_URL
        self.video_path = video_path
        self.output_dir = output_dir
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize flags
        self.is_running = False
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps_display = 0
        
        # Initialize statistics
        self.stats = {
            "total_frames": 0,
            "total_faces": 0,
            "low_focus_count": 0,
            "phone_usage_count": 0,
            "emotion_counts": {
                "angry": 0, "disgust": 0, "fear": 0, "happy": 0, 
                "sad": 0, "surprise": 0, "neutral": 0
            }
        }
        
        # Initialize display window
        self.window_name = 'Mood and Focus Analysis'
        self.stats_window = 'Statistics'
        
        # Set OpenCV window properties
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.stats_window, cv2.WINDOW_NORMAL)
        
        # Initialize models
        self.yolo_model = None
        self.face_cascade = None
        self.models_loaded = False
        
        # Initialize processing flags
        self.processing_enabled = True
        self.last_processing_time = 0
        self.processing_interval = 0.5  # Process every 0.5 seconds
        
        # Phone detection settings
        self.phone_classes = [67, 68, 69]  # YOLO classes for cell phone, laptop, keyboard
        self.phone_confidence_threshold = 0.5
        self.phone_detection_history = []
        self.phone_detection_window = 10  # Number of frames to consider for phone usage
        
        # Memory management
        self.last_result = None
        self.gc_counter = 0
        self.gc_interval = 100  # Run garbage collection every 100 frames
    
    def load_models(self):
        """Load all models upfront before showing any video"""
        print("Loading all models before starting video feed...")
        
        # Show loading message
        loading_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(loading_img, "Loading YOLOv8 model...", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(self.stats_window, loading_img)
        cv2.waitKey(1)
        
        # Load YOLO model
        print("Loading YOLOv8 model...")
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Update loading message
        cv2.putText(loading_img, "YOLOv8 model loaded", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(loading_img, "Loading face detection model...", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(self.stats_window, loading_img)
        cv2.waitKey(1)
        
        # Load face detection model
        print("Loading face detection model...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Update loading message
        cv2.putText(loading_img, "YOLOv8 model loaded", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(loading_img, "Face detection model loaded", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(loading_img, "Preparing to connect to camera...", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(self.stats_window, loading_img)
        cv2.waitKey(1)
        
        self.models_loaded = True
        print("All models loaded successfully")
    
    def connect_camera(self):
        """Establish connection to camera or video"""
        if self.rtsp_url:
            print(f"Connecting to camera at {self.rtsp_url}")
            
            # Set RTSP transport protocol to TCP for more reliable streaming
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            # Try different connection options
            connection_options = [
                # Option 1: Standard connection with TCP and minimal buffering
                f"{self.rtsp_url}?rtsp_transport=tcp&buffer_size=0",
                
                # Option 2: Standard connection with TCP
                f"{self.rtsp_url}?rtsp_transport=tcp",
                
                # Option 3: Standard connection
                self.rtsp_url
            ]
            
            for option in connection_options:
                try:
                    print(f"Trying connection option: {option}")
                    self.cap = cv2.VideoCapture(option, cv2.CAP_FFMPEG)
                    
                    # Set buffer size to reduce latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Try to read a frame to verify connection
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"Successfully connected using option: {option}")
                        print(f"Frame size: {frame.shape}")
                        break
                    else:
                        print(f"Failed to read frame with option: {option}")
                        self.cap.release()
                except Exception as e:
                    print(f"Error with connection option {option}: {str(e)}")
                    if hasattr(self, 'cap'):
                        self.cap.release()
            
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                raise Exception("Failed to connect to camera with any option")
                
        elif self.video_path:
            print(f"Opening video file: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            raise ValueError("Either rtsp_url or video_path must be provided")
        
        if not self.cap.isOpened():
            raise Exception("Failed to connect to camera or open video file")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_actual = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video properties: {self.frame_width}x{self.frame_height} @ {self.fps_actual}fps")
        
        # Set window sizes
        cv2.resizeWindow(self.window_name, self.frame_width, self.frame_height)
        cv2.resizeWindow(self.stats_window, 400, 200)
        
        return self.cap

    def detect_faces(self, frame):
        """Detect faces in the frame using YOLO person detection and face cascade"""
        # Get person detections from YOLO
        yolo_results = self.yolo_model(frame)
        
        # Extract person bounding boxes
        person_boxes = []
        for i, box in enumerate(yolo_results[0].boxes.xyxy):
            if int(yolo_results[0].boxes.cls[i]) == 0:  # person class
                person_boxes.append(box.cpu().numpy().astype(int))
        
        # For each person, try to detect face
        face_locations = []
        for box in person_boxes:
            x1, y1, x2, y2 = box
            person_roi = frame[y1:y2, x1:x2]
            
            # Skip if ROI is empty or too small
            if person_roi.size == 0 or person_roi.shape[0] < 10 or person_roi.shape[1] < 10:
                continue
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Convert face coordinates to original frame coordinates
            for (fx, fy, fw, fh) in faces:
                face_x1 = x1 + fx
                face_y1 = y1 + fy
                face_x2 = x1 + fx + fw
                face_y2 = y1 + fy + fh
                face_locations.append([face_x1, face_y1, face_x2, face_y2])
        
        return face_locations, person_boxes
    
    def detect_phones(self, frame):
        """Detect phones and other electronic devices in the frame"""
        # Get all detections from YOLO
        yolo_results = self.yolo_model(frame)
        
        # Extract phone/device bounding boxes
        phone_boxes = []
        phone_confidences = []
        phone_classes = []
        
        for i, box in enumerate(yolo_results[0].boxes.xyxy):
            cls_id = int(yolo_results[0].boxes.cls[i])
            conf = float(yolo_results[0].boxes.conf[i])
            
            # Check if it's a phone or other electronic device
            if cls_id in self.phone_classes and conf > self.phone_confidence_threshold:
                phone_boxes.append(box.cpu().numpy().astype(int))
                phone_confidences.append(conf)
                phone_classes.append(cls_id)
        
        # Update phone detection history
        self.phone_detection_history.append(len(phone_boxes) > 0)
        if len(self.phone_detection_history) > self.phone_detection_window:
            self.phone_detection_history.pop(0)
        
        # Determine if phones are being used based on detection history
        phone_usage = sum(self.phone_detection_history) / len(self.phone_detection_history) > 0.5 if self.phone_detection_history else False
        
        # Update phone usage count if phones are detected
        if phone_usage and len(self.phone_detection_history) == self.phone_detection_window:
            self.stats["phone_usage_count"] += 1
        
        return phone_boxes, phone_confidences, phone_classes, phone_usage
    
    def analyze_emotions(self, frame, face_locations):
        """Analyze emotions for each detected face"""
        emotions = []
        focus_scores = []
        
        for face_loc in face_locations:
            try:
                # Extract face region
                x1, y1, x2, y2 = face_loc
                face_img = frame[y1:y2, x1:x2]
                
                # Skip if face image is empty or too small
                if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                    emotions.append('unknown')
                    focus_scores.append(0.5)
                    continue
                
                # Analyze emotions using DeepFace
                result = DeepFace.analyze(face_img, 
                                        actions=['emotion'],
                                        enforce_detection=False,
                                        silent=True)
                
                # Get dominant emotion
                emotion = result[0]['dominant_emotion']
                emotions.append(emotion)
                
                # Update emotion statistics
                self.stats["emotion_counts"][emotion] += 1
                
                # Calculate focus score based on emotions
                if emotion in ['neutral', 'happy']:
                    focus_score = 0.8  # High focus
                elif emotion in ['sad', 'angry']:
                    focus_score = 0.4  # Low focus
                elif emotion in ['surprise', 'fear']:
                    focus_score = 0.3  # Very low focus (distracted)
                else:
                    focus_score = 0.6  # Medium focus
                
                # Update low focus count if score is below threshold
                if focus_score < 0.5:
                    self.stats["low_focus_count"] += 1
                
                focus_scores.append(focus_score)
                
            except Exception as e:
                logger.error(f"Face analysis error: {str(e)}")
                emotions.append('unknown')
                focus_scores.append(0.5)  # Default score
        
        return focus_scores, emotions
    
    def process_frame(self, frame):
        """Process a single frame for mood and focus analysis"""
        # Update statistics
        self.stats["total_frames"] += 1
        
        # Detect faces
        face_locations, person_boxes = self.detect_faces(frame)
        
        # Detect phones
        phone_boxes, phone_confidences, phone_classes, phone_usage = self.detect_phones(frame)
        
        # Update statistics
        self.stats["total_faces"] += len(face_locations)
        
        # Analyze emotions and focus
        focus_scores, emotions = self.analyze_emotions(frame, face_locations)
        
        # Convert numpy arrays to lists for JSON serialization
        face_locs_list = [loc.tolist() if isinstance(loc, np.ndarray) else loc for loc in face_locations]
        person_boxes_list = [box.tolist() if isinstance(box, np.ndarray) else box for box in person_boxes]
        phone_boxes_list = [box.tolist() if isinstance(box, np.ndarray) else box for box in phone_boxes]
        
        # Create result dictionary
        result = {
            'timestamp': datetime.now().isoformat(),
            'face_locations': face_locs_list,
            'person_boxes': person_boxes_list,
            'phone_boxes': phone_boxes_list,
            'phone_confidences': phone_confidences,
            'phone_classes': phone_classes,
            'phone_usage': phone_usage,
            'focus_scores': focus_scores,
            'emotions': emotions,
            'has_low_focus': any(score < 0.5 for score in focus_scores),
            'stats': {
                'total_frames': self.stats['total_frames'],
                'total_faces': self.stats['total_faces'],
                'low_focus_count': self.stats['low_focus_count'],
                'phone_usage_count': self.stats['phone_usage_count'],
                'emotion_counts': self.stats['emotion_counts']
            }
        }
        
        return result
    
    def draw_overlay(self, frame, result):
        """Draw mood detection results on frame"""
        # Draw person boxes
        for box in result['person_boxes']:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw phone boxes
        for i, (box, conf, cls_id) in enumerate(zip(result['phone_boxes'], 
                                                  result['phone_confidences'], 
                                                  result['phone_classes'])):
            x1, y1, x2, y2 = box
            
            # Determine class name
            if cls_id == 67:
                class_name = "Cell Phone"
            elif cls_id == 68:
                class_name = "Laptop"
            elif cls_id == 69:
                class_name = "Keyboard"
            else:
                class_name = "Device"
            
            # Draw phone box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add confidence text
            conf_text = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, conf_text, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw face boxes and add text
        for i, (face_loc, focus, emotion) in enumerate(zip(result['face_locations'], 
                                                         result['focus_scores'], 
                                                         result['emotions'])):
            x1, y1, x2, y2 = face_loc
            
            # Draw face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add focus and emotion text
            focus_text = f"Focus: {focus:.2f} ({emotion})"
            cv2.putText(frame, focus_text, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add FPS and timestamp
        cv2.putText(frame, f"FPS: {self.fps_display:.1f}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Time: {result['timestamp']}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add phone usage warning if phones are detected
        if result['phone_usage']:
            cv2.putText(frame, "PHONE USAGE DETECTED!", (10, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def create_stats_display(self):
        """Create statistics display image"""
        stats_img = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Add statistics text
        cv2.putText(stats_img, f"Frames: {self.stats['total_frames']}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(stats_img, f"Faces: {self.stats['total_faces']}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(stats_img, f"Low Focus: {self.stats['low_focus_count']}", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(stats_img, f"Phone Usage: {self.stats['phone_usage_count']}", (10, 120), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Add emotion distribution
        y_pos = 150
        for emotion, count in self.stats['emotion_counts'].items():
            if self.stats['total_faces'] > 0:
                percentage = (count / self.stats['total_faces']) * 100
                cv2.putText(stats_img, f"{emotion}: {count} ({percentage:.1f}%)", (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20
        
        return stats_img

    def run(self):
        """Run the mood analysis system in a single thread"""
        try:
            # Load all models upfront
            self.load_models()
            
            # Connect to camera
            self.connect_camera()
            
            # Set running flag
            self.is_running = True
            
            # Initialize FPS calculation
            last_fps_time = time.time()
            fps_counter = 0
            
            # Initialize result for overlay
            result = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'face_locations': [],
                'person_boxes': [],
                'phone_boxes': [],
                'phone_confidences': [],
                'phone_classes': [],
                'phone_usage': False,
                'focus_scores': [],
                'emotions': [],
                'has_low_focus': False
            }
            
            print("Starting mood analysis. Press 'q' to quit.")
            
            # Main loop
            while self.is_running:
                # Read frame directly without any queuing
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Failed to read frame, attempting to reconnect...")
                    self.cap.release()
                    time.sleep(1)
                    self.connect_camera()
                    continue
                
                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.fps_display = fps_counter
                    fps_counter = 0
                    last_fps_time = current_time
                
                # Process frame based on time interval
                if self.processing_enabled and (current_time - self.last_processing_time) >= self.processing_interval:
                    try:
                        # Process frame directly without copying
                        result = self.process_frame(frame)
                        self.last_processing_time = current_time
                        self.last_result = result
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
                        # Use last successful result if available
                        if self.last_result is not None:
                            result = self.last_result
                
                # Draw overlay on frame
                display_frame = self.draw_overlay(frame, result)
                
                # Create stats display
                stats_img = self.create_stats_display()
                
                # Display frames
                cv2.imshow(self.window_name, display_frame)
                cv2.imshow(self.stats_window, stats_img)
                
                # Make sure windows are visible and properly sized
                cv2.resizeWindow(self.window_name, self.frame_width, self.frame_height)
                cv2.resizeWindow(self.stats_window, 400, 200)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break
                
                # Run garbage collection periodically to prevent memory leaks
                self.gc_counter += 1
                if self.gc_counter >= self.gc_interval:
                    self.gc_counter = 0
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                
                # No sleep to maximize FPS
                
        except Exception as e:
            print(f"Error in run method: {str(e)}")
        finally:
            # Clean up
            if hasattr(self, 'cap'):
                self.cap.release()
            cv2.destroyAllWindows()
            print("Mood analysis stopped")

# --- Step 5: Redis Pub/Sub and Stream Processing ---
class StreamProcessor:
    def __init__(self, redis_host, redis_port, output_dir='mood_frames', fps=5, processing_interval=0.5, phone_confidence=0.5):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.active_streams = {}  # Dictionary to track active streams
        self.fps = fps
        self.lock = threading.Lock()
        self.output_dir = output_dir
        self.processing_interval = processing_interval
        self.phone_confidence = phone_confidence
        
        # Create analyzer instance
        self.analyzer = MoodAnalyzer(output_dir=output_dir)
        self.analyzer.processing_interval = processing_interval
        self.analyzer.phone_confidence_threshold = phone_confidence
        
        # Load models
        self.analyzer.load_models()
        
    def start_stream(self, source, client_id):
        """Start processing a stream for a specific client"""
        with self.lock:
            if client_id in self.active_streams:
                return False, "Stream already active for this client"
            
            # Create a unique channel for this client
            channel = f"mood_analysis:{client_id}"
            
            # Add to active streams
            self.active_streams[client_id] = {
                "source": source,
                "channel": channel,
                "active": True
            }
            
            # Start processing thread
            thread = threading.Thread(
                target=self._process_stream,
                args=(source, client_id, channel)
            )
            thread.daemon = True
            thread.start()
            
            return True, f"Stream started for client {client_id}"
    
    def stop_stream(self, client_id):
        """Stop processing a stream for a specific client"""
        with self.lock:
            if client_id not in self.active_streams:
                return False, "No active stream for this client"
            
            # Mark stream as inactive
            self.active_streams[client_id]["active"] = False
            
            # Remove from active streams
            del self.active_streams[client_id]
            
            return True, f"Stream stopped for client {client_id}"
    
    def _process_stream(self, source, client_id, channel):
        """Process a stream and publish results to Redis"""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        reconnect_delay = 5  # seconds
        
        while self.active_streams.get(client_id, {}).get("active", False):
            try:
                # Setup camera
                logger.info(f"Connecting to video source: {source}")
                cap = setup_camera(source)
                
                # Calculate frame delay for desired FPS
                frame_delay = 1.0 / self.fps
                last_frame_time = time.time()
                last_processing_time = 0
                
                logger.info(f"Starting mood analysis at {self.fps} FPS for client {client_id}...")
                
                # Reset reconnect attempts on successful connection
                reconnect_attempts = 0
                
                while self.active_streams.get(client_id, {}).get("active", False):
                    current_time = time.time()
                    if current_time - last_frame_time >= frame_delay:
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning("Failed to read frame, attempting to reconnect...")
                            break  # Break inner loop to attempt reconnection
                        
                        # Only process frame at the specified interval
                        if current_time - last_processing_time >= self.analyzer.processing_interval:
                            # Process the frame
                            result = self.analyzer.process_frame(frame)
                            
                            # Publish result to Redis
                            self.redis_client.publish(channel, json.dumps(result))
                            
                            # Save frame if it shows low focus or phone usage
                            if result['has_low_focus'] or result['phone_usage']:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                frame_path = os.path.join(self.output_dir, f"incident_{client_id}_{timestamp}.jpg")
                                cv2.imwrite(frame_path, frame)
                                
                                # Log incident
                                incident_type = "Phone Usage" if result['phone_usage'] else "Low Focus"
                                logger.info(f"[{timestamp}] Client {client_id}: {incident_type} detected")
                            
                            last_processing_time = current_time
                        
                        last_frame_time = current_time
                    
                    time.sleep(0.01)
    
                # Clean up
                cap.release()
                
            except Exception as e:
                logger.error(f"Error processing stream: {str(e)}")
                self.redis_client.publish(channel, json.dumps({"error": str(e)}))
                
                # Handle reconnection
                reconnect_attempts += 1
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts ({max_reconnect_attempts}) reached. Stopping stream.")
                    self.redis_client.publish(channel, json.dumps({"error": "Max reconnection attempts reached. Stream stopped."}))
                    break
                
                logger.info(f"Attempting to reconnect in {reconnect_delay} seconds (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                time.sleep(reconnect_delay)
        
        logger.info(f"Stream processing stopped for client {client_id}")
    
    def get_active_streams(self):
        """Get list of active streams"""
        with self.lock:
            return list(self.active_streams.keys())

# --- Step 6: Flask API ---
app = Flask(__name__)

@app.route('/api/mood/start', methods=['POST'])
def start_stream():
    data = request.json
    if not data or 'source' not in data:
        return jsonify({"error": "Missing source parameter"}), 400
    
    source = data['source']
    client_id = data.get('client_id', str(uuid.uuid4()))
    
    success, message = stream_processor.start_stream(source, client_id)
    if success:
        return jsonify({
            "success": True,
            "client_id": client_id,
            "message": message
        })
    else:
        return jsonify({"error": message}), 400

@app.route('/api/mood/stop', methods=['POST'])
def stop_stream():
    data = request.json
    if not data or 'client_id' not in data:
        return jsonify({"error": "Missing client_id parameter"}), 400
    
    client_id = data['client_id']
    success, message = stream_processor.stop_stream(client_id)
    
    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"error": message}), 400

@app.route('/api/mood/status', methods=['GET'])
def stream_status():
    active_streams = stream_processor.get_active_streams()
    return jsonify({"active_streams": active_streams})

@app.route('/api/mood/subscribe', methods=['GET'])
def subscribe_stream():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({"error": "Missing client_id parameter"}), 400
    
    # Check if client has an active stream
    if client_id not in stream_processor.active_streams:
        return jsonify({"error": "No active stream for this client"}), 400
    
    # Return the channel name for the client to subscribe to
    channel = f"mood_analysis:{client_id}"
    return jsonify({
        "channel": channel,
        "message": f"Subscribe to Redis channel '{channel}' to receive real-time updates"
    })

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Create stream processor
    stream_processor = StreamProcessor(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        output_dir=args.output_dir,
        fps=args.fps,
        processing_interval=args.processing_interval,
        phone_confidence=args.phone_confidence
    )
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5002)
