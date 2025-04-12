import os
import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import shutil
import time
import argparse
from datetime import datetime
from deepface import DeepFace
import numpy as np
import redis
import json
import threading
import uuid
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Step 1: Setup paths and arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Cheating Detection from RTSP Stream')
    parser.add_argument('--redis-host', type=str, default='redis', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to process')
    parser.add_argument('--output-dir', type=str, default='flagged_frames', help='Directory to save flagged frames')
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
    logger.info(f"Connecting to camera at {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception(f"Failed to connect to camera at {source}")
    return cap

# --- Step 3: Load models ---
print("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' or a custom-trained model

print("Loading CLIP model...")

# Load the processor and model
vl_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
vl_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vl_model.to(device)

# --- Step 4: Analyze frames ---
def analyze_frame_with_clip(frame, yolo_results, vl_processor, vl_model, device):
    # Convert CV2 frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    texts = [
        "students looking at each other's papers during exam",
        "students using phones or devices during exam",
        "students passing notes or whispering during exam",
        "students sitting normally and taking exam",
        "students working independently on their exam"
    ]
    
    # Process the image and text with the CLIP processor
    inputs = vl_processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get the image and text features
    image_features = vl_model.get_image_features(pixel_values=inputs['pixel_values'])
    text_features = vl_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Calculate the similarity between image and text features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Get the highest similarity score for cheating behaviors (first 3 prompts)
    cheating_score = similarity[0][:3].max().item()
    normal_score = similarity[0][3:].max().item()
    
    # Check YOLO results for phones and suspicious objects
    num_phones = len([obj for obj in yolo_results[0].boxes.cls.tolist() if int(obj) == 67])  # class 67 is 'cell phone'
    num_people = len([obj for obj in yolo_results[0].boxes.cls.tolist() if int(obj) == 0])   # class 0 is 'person'
    
    # Increase confidence if phones are detected
    if num_phones > 0:
        cheating_score = max(cheating_score, 0.8)
    
    # Use a threshold to determine if cheating is detected
    threshold = 0.4
    if cheating_score > threshold and cheating_score > normal_score:
        behavior_type = texts[similarity[0][:3].argmax().item()]
        confidence = round(cheating_score * 100, 2)
        return True, f"Suspicious activity detected: {behavior_type} (Confidence: {confidence}%, People: {num_people}, Phones: {num_phones})"
    else:
        return False, f"No suspicious activity detected (People: {num_people}, Phones: {num_phones})"

# Add new function for mood/focus analysis
def analyze_student_focus(frame, face_locations):
    focus_scores = []
    emotions = []
    
    for face_loc in face_locations:
        try:
            # Extract face region
            x1, y1, x2, y2 = face_loc
            face_img = frame[y1:y2, x1:x2]
            
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(face_img, 
                                    actions=['emotion'],
                                    enforce_detection=False,
                                    silent=True)
            
            # Calculate focus score based on emotions
            emotion = result[0]['dominant_emotion']
            emotions.append(emotion)
            
            # Focus scoring logic
            if emotion in ['neutral', 'happy']:
                focus_score = 0.8  # High focus
            elif emotion in ['sad', 'angry']:
                focus_score = 0.4  # Low focus
            else:
                focus_score = 0.6  # Medium focus
                
            focus_scores.append(focus_score)
            
        except Exception as e:
            logger.error(f"Face analysis error: {str(e)}")
            focus_scores.append(0.5)  # Default score
            emotions.append('unknown')
    
    return focus_scores, emotions

# --- Step 5: Redis Pub/Sub and Stream Processing ---
class CheatDetectionProcessor:
    def __init__(self, redis_host, redis_port, output_dir='flagged_frames', fps=5):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.active_streams = {}  # Dictionary to track active streams
        self.fps = fps
        self.lock = threading.Lock()
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def start_stream(self, source, client_id):
        """Start processing a stream for a specific client"""
        with self.lock:
            if client_id in self.active_streams:
                return False, "Stream already active for this client"
            
            # Create a unique channel for this client
            channel = f"cheating_detection:{client_id}"
            
            # Add to active streams
            self.active_streams[client_id] = {
                "source": source,
                "channel": channel,
                "active": True,
                "incidents": {
                    "phones": [],
                    "looking": [],
                    "passing": []
                }
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
            
            # Generate summary report
            summary = self.generate_summary_report(client_id)
            
            # Publish summary report
            channel = self.active_streams[client_id]["channel"]
            self.redis_client.publish(channel, json.dumps({
                "report": summary,
                "final": True
            }))
            
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
                
                logger.info(f"Starting detection at {self.fps} FPS for client {client_id}...")
                
                # Reset reconnect attempts on successful connection
                reconnect_attempts = 0
                
                while self.active_streams.get(client_id, {}).get("active", False):
                    current_time = time.time()
                    if current_time - last_frame_time >= frame_delay:
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning("Failed to read frame, attempting to reconnect...")
                            break  # Break inner loop to attempt reconnection
                        
                        # Run YOLO detection for people and phones
                        yolo_results = yolo_model(frame)
                        
                        # Get face locations using YOLO person detections
                        face_locations = []
                        for box in yolo_results[0].boxes.xyxy:
                            if int(yolo_results[0].boxes.cls[0]) == 0:  # person class
                                face_locations.append(box.cpu().numpy().astype(int))
                        
                        # Analyze focus and emotions
                        focus_scores, emotions = analyze_student_focus(frame, face_locations)
                        
                        # Run cheating detection
                        is_suspicious, description = analyze_frame_with_clip(
                            frame, 
                            yolo_results,
                            vl_processor,
                            vl_model,
                            device
                        )
                        
                        # Update incidents if suspicious activity is detected
                        if is_suspicious:
                            timestamp = current_time
                            
                            # Determine incident type
                            if "using phones" in description:
                                incident_type = "phones"
                            elif "looking at each other's papers" in description:
                                incident_type = "looking"
                            elif "passing notes" in description:
                                incident_type = "passing"
                            else:
                                incident_type = "other"
                            
                            # Extract confidence from description
                            confidence = float(description.split('Confidence: ')[1].split('%')[0])
                            
                            # Add to incidents list
                            with self.lock:
                                if client_id in self.active_streams:
                                    self.active_streams[client_id]["incidents"][incident_type].append((timestamp, confidence, description))
                            
                            # Save frame with suspicious activity
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            frame_path = os.path.join(self.output_dir, f"incident_{client_id}_{timestamp_str}.jpg")
                            cv2.imwrite(frame_path, frame)
                        
                        # Create result object
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "is_suspicious": is_suspicious,
                            "description": description,
                            "face_locations": [loc.tolist() for loc in face_locations],
                            "focus_scores": focus_scores,
                            "emotions": emotions,
                            "has_low_focus": any(score < 0.5 for score in focus_scores),
                            "num_phones": len([obj for obj in yolo_results[0].boxes.cls.tolist() if int(obj) == 67])
                        }
                        
                        # Publish result to Redis
                        self.redis_client.publish(channel, json.dumps(result))
                        
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
    
    def generate_summary_report(self, client_id):
        """Generate a summary report for the client"""
        with self.lock:
            if client_id not in self.active_streams:
                return {"error": "No active stream for this client"}
            
            incidents = self.active_streams[client_id]["incidents"]
            
            # Calculate total incidents
            total_incidents = len(incidents["phones"]) + len(incidents["looking"]) + len(incidents["passing"])
            
            summary = {
                "total_suspicious_incidents": total_incidents,
            }
            
            # Add phone usage incidents
            if incidents["phones"]:
                phone_timestamps = [f"{t:.2f}s" for t, _, _ in incidents["phones"]]
                max_conf = max([conf for _, conf, _ in incidents["phones"]])
                summary["phone_usage"] = {
                    "detections": len(incidents["phones"]),
                    "timestamps": phone_timestamps,
                    "highest_confidence": max_conf
                }
            
            # Add looking at others' papers incidents
            if incidents["looking"]:
                looking_timestamps = [f"{t:.2f}s" for t, _, _ in incidents["looking"]]
                max_conf = max([conf for _, conf, _ in incidents["looking"]])
                summary["looking_at_papers"] = {
                    "detections": len(incidents["looking"]),
                    "timestamps": looking_timestamps,
                    "highest_confidence": max_conf
                }
            
            # Add passing notes incidents
            if incidents["passing"]:
                passing_timestamps = [f"{t:.2f}s" for t, _, _ in incidents["passing"]]
                max_conf = max([conf for _, conf, _ in incidents["passing"]])
                summary["passing_notes"] = {
                    "detections": len(incidents["passing"]),
                    "timestamps": passing_timestamps,
                    "highest_confidence": max_conf
                }
            
            return summary

# --- Step 6: Flask API ---
app = Flask(__name__)

@app.route('/api/cheat/start', methods=['POST'])
def start_stream():
    data = request.json
    if not data or 'source' not in data:
        return jsonify({"error": "Missing source parameter"}), 400
    
    source = data['source']
    client_id = data.get('client_id', str(uuid.uuid4()))
    
    success, message = processor.start_stream(source, client_id)
    if success:
        return jsonify({
            "success": True,
            "client_id": client_id,
            "message": message
        })
    else:
        return jsonify({"error": message}), 400

@app.route('/api/cheat/stop', methods=['POST'])
def stop_stream():
    data = request.json
    if not data or 'client_id' not in data:
        return jsonify({"error": "Missing client_id parameter"}), 400
    
    client_id = data['client_id']
    success, message = processor.stop_stream(client_id)
    
    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"error": message}), 400

@app.route('/api/cheat/status', methods=['GET'])
def stream_status():
    active_streams = processor.get_active_streams()
    return jsonify({"active_streams": active_streams})

@app.route('/api/cheat/subscribe', methods=['GET'])
def subscribe_stream():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({"error": "Missing client_id parameter"}), 400
    
    # Check if client has an active stream
    if client_id not in processor.active_streams:
        return jsonify({"error": "No active stream for this client"}), 400
    
    # Return the channel name for the client to subscribe to
    channel = f"cheating_detection:{client_id}"
    return jsonify({
        "channel": channel,
        "message": f"Subscribe to Redis channel '{channel}' to receive real-time updates"
    })

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Create stream processor
    processor = CheatDetectionProcessor(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        output_dir=args.output_dir,
        fps=args.fps
    )
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5003)
