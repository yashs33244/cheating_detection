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
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RTSP URL configuration
RTSP_URL = "rtsp://admin123:admin123@192.168.79.98:554/11"
OUTPUT_DIR = 'mood_frames'
FPS = 60  # Increased for maximum performance

# Custom JSON encoder to handle numpy arrays and types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Step 1: Setup paths and arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Mood and Focus Analysis for Classroom Surveillance')
    parser.add_argument('--redis-host', type=str, default='redis', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to process')
    parser.add_argument('--processing-interval', type=float, default=0.5, help='Interval in seconds between processing frames')
    parser.add_argument('--phone-confidence', type=float, default=0.5, help='Confidence threshold for phone detection')
    parser.add_argument('--output-dir', type=str, default='mood_frames', help='Directory to save mood frames')
    parser.add_argument('--headless', type=str, default='false', help='Run in headless mode without UI windows')
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
        
        # Set environment variables for RTSP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024|stimeout;5000000"
        
        # Try different RTSP connection options
        connection_attempts = 0
        max_attempts = 3
        
        while connection_attempts < max_attempts:
            connection_attempts += 1
            logger.info(f"RTSP connection attempt {connection_attempts}/{max_attempts}")
            
            try:
                # Create capture with FFMPEG backend
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                
                # Set buffer size to reduce latency but not too small
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                # Set timeout
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for more reliable streaming
                
                # Try to read the first frame with a timeout
                logger.info("Attempting to read first frame from RTSP...")
                read_success = False
                
                # Set a timeout for the first frame
                start_time = time.time()
                timeout = 10  # 10 seconds timeout
                
                while time.time() - start_time < timeout and not read_success:
                    read_success, frame = cap.read()
                    if read_success and frame is not None:
                        logger.info(f"Successfully read first frame from RTSP stream, shape: {frame.shape}")
                        return cap
                    time.sleep(0.5)
                
                if not read_success:
                    logger.warning(f"Failed to read first frame from RTSP within timeout, retrying...")
                    cap.release()
            except Exception as e:
                logger.error(f"Error connecting to RTSP stream: {str(e)}")
            
            time.sleep(2)  # Wait before retrying
        
        # If we've tried all options and failed, try with default settings as last resort
        logger.warning("All RTSP connection attempts failed, trying with default settings")
        cap = cv2.VideoCapture(source)
                
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
    
    # Try to get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Successfully connected to video source. Resolution: {width}x{height}, FPS: {fps}")
    return cap

class MoodAnalyzer:
    def __init__(self, rtsp_url=None, video_path=None, output_dir='mood_frames', fps=60, headless=False):
        self.rtsp_url = rtsp_url or RTSP_URL
        self.video_path = video_path
        self.output_dir = output_dir
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.headless = headless
        
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
        
        # Initialize display window only if not in headless mode
        if not self.headless:
            self.window_name = 'Mood and Focus Analysis'
            self.stats_window = 'Statistics'
            # Set OpenCV window properties
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.stats_window, cv2.WINDOW_NORMAL)
        else:
            self.window_name = None
            self.stats_window = None
        
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
        
        if not self.headless:
            # Show loading message
            loading_img = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(loading_img, "Loading YOLOv8 model...", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(self.stats_window, loading_img)
            cv2.waitKey(1)
        
        # Load YOLO model
        print("Loading YOLOv8 model...")
        self.yolo_model = YOLO("yolov8n.pt")
        
        if not self.headless:
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
        
        if not self.headless:
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
            
            while self.is_running:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("Error: Failed to capture frame")
                    break
                
                current_time = time.time()
                
                # Calculate and update FPS
                fps_counter += 1
                if current_time - last_fps_time >= 1.0:
                    self.fps_display = fps_counter
                    fps_counter = 0
                    last_fps_time = current_time
                
                # Process frame if enough time has passed since last processing
                if current_time - self.last_processing_time >= self.processing_interval:
                    if self.processing_enabled:
                        result = self.process_frame(frame)
                        self.last_result = result
                    self.last_processing_time = current_time
                
                # Draw overlay on frame
                if self.last_result:
                    display_frame = self.draw_overlay(frame.copy(), self.last_result)
                else:
                    display_frame = frame.copy()
                
                # Save frame periodically
                if self.frame_count % 30 == 0:  # Save every 30 frames
                    frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count}.jpg")
                    cv2.imwrite(frame_path, display_frame)
                
                # Draw FPS counter
                cv2.putText(display_frame, f"FPS: {self.fps_display}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Create stats display
                stats_img = self.create_stats_display()
                
                # Display frames if not in headless mode
                if not self.headless:
                    cv2.imshow(self.window_name, display_frame)
                    cv2.imshow(self.stats_window, stats_img)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.processing_enabled = not self.processing_enabled
                        print(f"Processing {'enabled' if self.processing_enabled else 'disabled'}")
                
                # Increment frame counter
                self.frame_count += 1
                
                # Garbage collection to prevent memory leaks
                self.gc_counter += 1
                if self.gc_counter >= self.gc_interval:
                    gc.collect()
                    self.gc_counter = 0
                
                # Control frame rate
                elapsed = time.time() - current_time
                delay = max(0, self.frame_delay - elapsed)
                if delay > 0:
                    time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error in mood analysis: {str(e)}")
        finally:
            self.is_running = False
            if hasattr(self, 'cap'):
                self.cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            print("Mood analysis stopped")

# --- Step 5: Redis Pub/Sub and Stream Processing ---
class StreamProcessor:
    def __init__(self, redis_host, redis_port, output_dir='mood_frames', fps=5, processing_interval=0.5, phone_confidence=0.5, headless=False):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.output_dir = output_dir
        self.fps = fps
        self.processing_interval = processing_interval
        self.phone_confidence = phone_confidence
        self.headless = headless
        self.active_streams = {}
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.lock = threading.Lock()
        print(f"StreamProcessor initialized with Redis at {redis_host}:{redis_port}")
        
        # Create analyzer instance
        self.analyzer = MoodAnalyzer(output_dir=output_dir, fps=fps, headless=headless)
        self.analyzer.processing_interval = processing_interval
        self.analyzer.phone_confidence_threshold = phone_confidence
        
        # Load models
        self.analyzer.load_models()
        
    def start_stream(self, source, client_id):
        """Start processing a stream for a specific client"""
        with self.lock:
            if client_id in self.active_streams:
                logger.info(f"Stream already active for client {client_id}")
                return False, "Stream already active for this client"
            
            # Create a unique channel for this client
            channel = f"mood_analysis:{client_id}"
            
            # Add to active streams dictionary with all necessary info
            self.active_streams[client_id] = {
                "source": source,
                "channel": channel,
                "start_time": datetime.now().isoformat(),
                "active": True
            }
            
            logger.info(f"Added client {client_id} to active_streams. Current active streams: {list(self.active_streams.keys())}")
            
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
                # Stream might have already been cleaned up after error or completion
                # Return success to prevent client error
                logger.info(f"Stop request for client {client_id} but no active stream found. The stream may have already closed.")
                return True, f"No active stream for client {client_id}, it may have already been closed"
            
            # Mark stream as inactive
            self.active_streams[client_id]["active"] = False
            
            # Remove from active streams
            del self.active_streams[client_id]
            
            logger.info(f"Stream stopped for client {client_id}")
            return True, f"Stream stopped for client {client_id}"
    
    def _process_stream(self, source, client_id, channel):
        """Process a video stream and publish results to a Redis channel"""
        try:
            print(f"Starting stream processing for client {client_id}")
            
            # Create a new analyzer instance for this stream
            analyzer = MoodAnalyzer(output_dir=self.output_dir, fps=self.fps, headless=self.headless)
            analyzer.processing_interval = self.processing_interval
            analyzer.phone_confidence_threshold = self.phone_confidence
            
            # Check if this is an RTSP stream
            is_rtsp = bool(re.match(r'rtsp://', source))
            if is_rtsp:
                logger.info(f"Processing RTSP stream for client {client_id}: {source}")
                # Publish initial status message
                self.redis_client.publish(channel, json.dumps({
                    'status': 'connecting',
                    'message': f"Connecting to RTSP stream: {source}",
                    'timestamp': datetime.now().isoformat()
                }, cls=NumpyEncoder))
            
            # Setup camera
            try:
                cap = setup_camera(source)
                
                # Publish connection success message
                self.redis_client.publish(channel, json.dumps({
                    'status': 'connected',
                    'message': f"Successfully connected to video source",
                    'timestamp': datetime.now().isoformat()
                }, cls=NumpyEncoder))
                
            except Exception as e:
                error_msg = f"Error setting up camera: {str(e)}"
                logger.error(error_msg)
                self.redis_client.publish(channel, json.dumps({
                    'error': f"Failed to connect to camera source: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }, cls=NumpyEncoder))
                
                # Remove from active streams
                with self.lock:
                    if client_id in self.active_streams:
                        del self.active_streams[client_id]
                return
            
            # Load models (this may take some time)
            logger.info(f"Loading models for client {client_id}")
            
            # Publish model loading message
            self.redis_client.publish(channel, json.dumps({
                'status': 'loading_models',
                'message': "Loading AI models for analysis...",
                'timestamp': datetime.now().isoformat()
            }, cls=NumpyEncoder))
            
            analyzer.load_models()
            
            # Publish models loaded message
            self.redis_client.publish(channel, json.dumps({
                'status': 'models_loaded',
                'message': "AI models loaded successfully",
                'timestamp': datetime.now().isoformat()
            }, cls=NumpyEncoder))
            
            # Update active streams with frame count - don't replace the entire entry
            with self.lock:
                if client_id in self.active_streams:
                    self.active_streams[client_id]['frame_count'] = 0
                    self.active_streams[client_id]['status'] = 'processing'
                else:
                    # Stream was stopped during setup
                    cap.release()
                    return
            
            logger.info(f"Processing stream for client {client_id}")
            
            # Publish processing started message
            self.redis_client.publish(channel, json.dumps({
                'status': 'processing',
                'message': "Stream processing started",
                'timestamp': datetime.now().isoformat()
            }, cls=NumpyEncoder))
            
            frame_count = 0
            processing_enabled = True
            last_processing_time = time.time()
            empty_frame_count = 0
            max_empty_frames = 10  # Maximum consecutive empty frames before reconnecting
            
            # Main processing loop - check if the client_id is still in active_streams
            while client_id in self.active_streams and self.active_streams.get(client_id, {}).get('active', False):
                # Read frame
                ret, frame = cap.read()
                
                # Handle empty frames or read errors
                if not ret or frame is None:
                    empty_frame_count += 1
                    logger.warning(f"Failed to read frame for client {client_id}, empty frames: {empty_frame_count}/{max_empty_frames}")
                    
                    # If we've had too many empty frames in a row, try to reconnect for RTSP
                    if empty_frame_count >= max_empty_frames and is_rtsp:
                        logger.warning(f"Too many empty frames, attempting to reconnect RTSP stream for client {client_id}")
                        
                        # Publish reconnecting message
                        self.redis_client.publish(channel, json.dumps({
                            'status': 'reconnecting',
                            'message': "Connection issues detected, attempting to reconnect...",
                            'timestamp': datetime.now().isoformat()
                        }, cls=NumpyEncoder))
                        
                        # Release current capture and try to reconnect
                        cap.release()
                        try:
                            cap = setup_camera(source)
                            empty_frame_count = 0
                            
                            # Publish reconnected message
                            self.redis_client.publish(channel, json.dumps({
                                'status': 'reconnected',
                                'message': "Successfully reconnected to video source",
                                'timestamp': datetime.now().isoformat()
                            }, cls=NumpyEncoder))
                            
                            continue
                        except Exception as e:
                            logger.error(f"Failed to reconnect to RTSP stream: {str(e)}")
                            break
                    
                    # For non-RTSP or if we've exceeded retries, end the stream
                    if empty_frame_count >= 30 or not is_rtsp:
                        logger.error(f"Failed to read frames repeatedly, ending stream for client {client_id}")
                        break
                    
                    # Skip this iteration and try again
                    time.sleep(0.1)
                    continue
                
                # Reset empty frame counter if we successfully read a frame
                empty_frame_count = 0
                current_time = time.time()
                
                # Process frame if enough time has passed
                if processing_enabled and (current_time - last_processing_time) >= analyzer.processing_interval:
                    try:
                        # Process the frame
                        result = analyzer.process_frame(frame)
                        
                        # Add status field to result
                        result['status'] = 'frame_processed'
                        
                        # Publish to Redis using the NumpyEncoder for proper serialization
                        self.redis_client.publish(channel, json.dumps(result, cls=NumpyEncoder))
                        
                        # Update frame count in active streams
                        with self.lock:
                            if client_id in self.active_streams:
                                self.active_streams[client_id]['frame_count'] = frame_count
                                
                        # Update last processing time
                        last_processing_time = current_time
                        
                    except Exception as e:
                        error_msg = f"Error processing frame for client {client_id}: {str(e)}"
                        logger.error(error_msg)
                        self.redis_client.publish(channel, json.dumps({
                            'error': error_msg,
                            'timestamp': datetime.now().isoformat()
                        }, cls=NumpyEncoder))
                
                # Increment frame count
                frame_count += 1
                
                # Sleep to control FPS
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
            # Publish stream ended message
            self.redis_client.publish(channel, json.dumps({
                'status': 'stream_ended',
                'message': f"Stream processing ended for client {client_id} after processing {frame_count} frames",
                'timestamp': datetime.now().isoformat(),
                'frame_count': frame_count
            }, cls=NumpyEncoder))
            
            logger.info(f"Stream processing ended for client {client_id}")
            
            # Cleanup
            cap.release()
            
        except Exception as e:
            error_msg = f"Error in stream processing for client {client_id}: {str(e)}"
            logger.error(error_msg)
            try:
                self.redis_client.publish(channel, json.dumps({
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }, cls=NumpyEncoder))
            except Exception as redis_err:
                logger.error(f"Failed to publish error message to Redis: {str(redis_err)}")
            
        finally:
            # Remove from active streams if still there
            with self.lock:
                if client_id in self.active_streams:
                    del self.active_streams[client_id]
            logger.info(f"Cleaned up resources for client {client_id}")
    
    def get_active_streams(self):
        """Get list of active streams"""
        with self.lock:
            # Just return the list of client IDs for backward compatibility
            return list(self.active_streams.keys())
            
    def get_detailed_streams(self):
        """Get detailed information about active streams"""
        with self.lock:
            # Return a copy of the active streams dictionary to avoid threading issues
            return self.active_streams.copy()

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
    
    # Always return success (200 OK) with appropriate message
    return jsonify({"success": True, "message": message})

@app.route('/api/mood/status', methods=['GET'])
def stream_status():
    with stream_processor.lock:
        active_streams = list(stream_processor.active_streams.keys())
    
    detailed = request.args.get('detailed', 'false').lower() == 'true'
    
    if detailed:
        # Return detailed information about streams
        with stream_processor.lock:
            detailed_streams = stream_processor.active_streams.copy()
        return jsonify({"active_streams": detailed_streams})
    else:
        # Return just the list of client IDs
        return jsonify({"active_streams": active_streams})

@app.route('/api/mood/subscribe', methods=['GET'])
def subscribe_stream():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({"error": "Missing client_id parameter"}), 400
    
    # Check if client has an active stream
    with stream_processor.lock:
        is_active = client_id in stream_processor.active_streams
    
    # Log current active streams
    logger.info(f"Subscribe request for client {client_id}. Active streams: {list(stream_processor.active_streams.keys())}")
    
    # Create channel name regardless of whether the stream is active
    channel = f"mood_analysis:{client_id}"
    
    # Check if the stream is active
    if not is_active:
        # Return channel but with warning
        logger.warning(f"No active stream found for client {client_id} when subscribing")
        return jsonify({
            "channel": channel,
            "warning": f"No active stream found for client {client_id}. The channel may not receive any messages.",
            "message": f"Subscribe to Redis channel '{channel}' to receive real-time updates"
        }), 200  # Still return 200 OK to allow client to continue
    
    # Return the channel name for the client to subscribe to
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
        phone_confidence=args.phone_confidence,
        headless=args.headless.lower() == 'true'
    )
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5002)
