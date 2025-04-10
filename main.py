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
import redis
import json
import threading
import uuid
from flask import Flask, request, jsonify
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Step 1: Setup paths and arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Cheating Detection from RTSP Stream or Video URL')
    parser.add_argument('--redis-host', type=str, default='redis', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to process')
    return parser.parse_args()

# --- Step 2: Setup camera connection ---
def setup_camera(source):
    """
    Setup camera connection for either RTSP stream or video file/URL
    
    Args:
        source: RTSP URL, local file path, or public URL (including S3)
        
    Returns:
        cv2.VideoCapture object
    """
    # Check if source is a URL (including S3)
    is_url = bool(re.match(r'https?://', source))
    
    # For URLs, we need to handle them differently
    if is_url:
        logger.info(f"Processing video from URL: {source}")
        # For URLs, we need to use the URL directly
        cap = cv2.VideoCapture(source)
    else:
        # For local files, check if they exist
        if not os.path.exists(source):
            raise Exception(f"File not found: {source}")
        logger.info(f"Processing local video file: {source}")
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise Exception(f"Failed to open video source: {source}")
    
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

# --- Step 5: Redis Pub/Sub and Stream Processing ---
class StreamProcessor:
    def __init__(self, redis_host, redis_port, fps=5):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.pubsub = self.redis_client.pubsub()
        self.active_streams = {}  # Dictionary to track active streams
        self.fps = fps
        self.lock = threading.Lock()
        
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
        try:
            # Setup camera
            logger.info(f"Connecting to video source: {source}")
            cap = setup_camera(source)
            
            # Calculate frame delay for desired FPS
            frame_delay = 1.0 / self.fps
            last_frame_time = time.time()
            
            logger.info(f"Starting detection at {self.fps} FPS for client {client_id}...")
            
            while self.active_streams.get(client_id, {}).get("active", False):
                current_time = time.time()
                if current_time - last_frame_time >= frame_delay:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame, attempting to reconnect...")
                        cap.release()
                        cap = setup_camera(source)
                        continue
                    
                    # Run YOLO detection
                    yolo_results = yolo_model(frame)
                    
                    # Run CLIP analysis
                    is_suspicious, description = analyze_frame_with_clip(
                        frame, 
                        yolo_results,
                        vl_processor,
                        vl_model,
                        device
                    )
                    
                    # Create result object
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "is_suspicious": is_suspicious,
                        "description": description
                    }
                    
                    # Publish result to Redis
                    self.redis_client.publish(channel, json.dumps(result))
                    
                    last_frame_time = current_time
                
                time.sleep(0.01)
            
            # Clean up
            cap.release()
            logger.info(f"Stream processing stopped for client {client_id}")
            
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            self.redis_client.publish(channel, json.dumps({"error": str(e)}))
    
    def get_active_streams(self):
        """Get list of active streams"""
        with self.lock:
            return list(self.active_streams.keys())

# --- Step 6: Flask API ---
app = Flask(__name__)
args = parse_arguments()
stream_processor = StreamProcessor(args.redis_host, args.redis_port, args.fps)

@app.route('/api/stream/start', methods=['POST'])
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

@app.route('/api/stream/stop', methods=['POST'])
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

@app.route('/api/stream/status', methods=['GET'])
def stream_status():
    active_streams = stream_processor.get_active_streams()
    return jsonify({"active_streams": active_streams})

@app.route('/api/stream/subscribe', methods=['GET'])
def subscribe_stream():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({"error": "Missing client_id parameter"}), 400
    
    # Create a Redis pubsub connection for this client
    pubsub = redis.Redis(host=args.redis_host, port=args.redis_port).pubsub()
    channel = f"cheating_detection:{client_id}"
    pubsub.subscribe(channel)
    
    # Return the channel name for the client to subscribe to
    return jsonify({
        "channel": channel,
        "message": f"Subscribe to Redis channel '{channel}' to receive real-time updates"
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
