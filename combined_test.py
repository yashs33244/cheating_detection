import requests
import json
import time
import redis
import argparse
import threading
import os
import sys
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test both Cheating Detection and Mood Analysis APIs')
    parser.add_argument('--source', type=str, required=True, 
                       help='Video source (RTSP URL or file path)')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    
    # Cheating detection service settings
    parser.add_argument('--cheat-host', type=str, default='localhost', help='Cheating detection host')
    parser.add_argument('--cheat-port', type=int, default=5003, help='Cheating detection port')
    
    # Mood analysis service settings
    parser.add_argument('--mood-host', type=str, default='localhost', help='Mood analysis host')
    parser.add_argument('--mood-port', type=int, default=5002, help='Mood analysis port')
    
    # Redis settings
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    return parser.parse_args()

class ServiceClient:
    def __init__(self, service_name, host, port, redis_host, redis_port):
        self.service_name = service_name
        self.host = host
        self.port = port
        self.api_url = f"http://{host}:{port}/api"
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.client_id = None
        self.redis_client = None
        self.pubsub = None
        self.channel = None
        self.results = []
        self.output_dir = None
        
    def start_stream(self, source):
        """Start a stream"""
        endpoint = f"/{self.service_name}/start"
        try:
            response = requests.post(f"{self.api_url}{endpoint}", json={
                "source": source
            })
            
            if response.status_code != 200:
                print(f"Error starting {self.service_name} stream: {response.text}")
                return False
            
            result = response.json()
            self.client_id = result["client_id"]
            print(f"[{self.service_name}] Stream started for client {self.client_id}")
            return True
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {self.service_name} API at {self.api_url}")
            return False
    
    def stop_stream(self):
        """Stop the stream"""
        if not self.client_id:
            return False
        
        endpoint = f"/{self.service_name}/stop"
        try:
            response = requests.post(f"{self.api_url}{endpoint}", json={
                "client_id": self.client_id
            })
            
            if response.status_code != 200:
                print(f"Error stopping {self.service_name} stream: {response.text}")
                return False
            
            result = response.json()
            print(f"[{self.service_name}] Stream stopped: {result['message']}")
            return True
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {self.service_name} API at {self.api_url}")
            return False
    
    def subscribe_to_events(self):
        """Subscribe to events"""
        if not self.client_id:
            return False
        
        # Get subscription details
        endpoint = f"/{self.service_name}/subscribe"
        try:
            response = requests.get(f"{self.api_url}{endpoint}?client_id={self.client_id}")
            
            if response.status_code != 200:
                print(f"Error subscribing to {self.service_name} events: {response.text}")
                return False
            
            result = response.json()
            self.channel = result["channel"]
            print(f"[{self.service_name}] Subscribing to channel: {self.channel}")
            
            # Connect to Redis
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port)
            self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe(self.channel)
            print(f"[{self.service_name}] Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
            
            # Create output directory if not exists
            if not self.output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = f"test_results_{timestamp}"
                os.makedirs(self.output_dir, exist_ok=True)
            
            return True
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {self.service_name} API at {self.api_url}")
            return False
        except redis.exceptions.ConnectionError:
            print(f"Error: Could not connect to Redis at {self.redis_host}:{self.redis_port}")
            return False
    
    def process_events(self, duration, process_callback=None):
        """Process events for the specified duration"""
        if not self.pubsub:
            return False
        
        # Skip the first message (subscription confirmation)
        next(self.pubsub.listen())
        
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < duration:
            message = self.pubsub.get_message(timeout=1.0)
            if message and message['type'] == 'message':
                message_count += 1
                try:
                    data = json.loads(message['data'])
                    
                    # Save result
                    self.results.append(data)
                    
                    # Call custom processing callback if provided
                    if process_callback:
                        process_callback(self, data, message_count)
                    
                    # Save results to file periodically
                    if message_count % 10 == 0:
                        self.save_results()
                        
                except Exception as e:
                    print(f"[{self.service_name}] Error processing message: {str(e)}")
        
        # Save final results
        self.save_results()
        print(f"[{self.service_name}] Received {message_count} messages")
        return True
    
    def save_results(self):
        """Save results to file"""
        if not self.output_dir:
            return
        
        results_file = os.path.join(self.output_dir, f"{self.service_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

def process_cheat_event(client, data, message_count):
    """Process a cheating detection event"""
    if 'error' in data:
        print(f"[cheat] Error: {data['error']}")
        return
    
    if 'final' in data and data['final'] and 'report' in data:
        print(f"\n[cheat] Received final report:")
        print(json.dumps(data['report'], indent=2))
        return
    
    is_suspicious = data.get('is_suspicious', False)
    description = data.get('description', '')
    num_faces = len(data.get('face_locations', []))
    
    print(f"[cheat] Message {message_count}: Suspicious={is_suspicious}, Faces={num_faces}")
    if is_suspicious:
        print(f"[cheat] Description: {description}")

def process_mood_event(client, data, message_count):
    """Process a mood analysis event"""
    if 'error' in data:
        print(f"[mood] Error: {data['error']}")
        return
    
    num_faces = len(data.get('face_locations', [])) if 'face_locations' in data else 0
    has_low_focus = data.get('has_low_focus', False)
    phone_usage = data.get('phone_usage', False)
    emotions = data.get('emotions', [])
    
    # Format emotions for display
    emotion_str = ", ".join(emotions) if emotions else "none"
    
    print(f"[mood] Message {message_count}: Faces={num_faces}, Low Focus={has_low_focus}, Phone={phone_usage}, Emotions={emotion_str}")

def run_service(service_client, source, duration, process_callback):
    """Run a service client from start to finish"""
    # Start stream
    if not service_client.start_stream(source):
        return False
    
    # Subscribe to events
    if not service_client.subscribe_to_events():
        service_client.stop_stream()
        return False
    
    # Process events
    service_client.process_events(duration, process_callback)
    
    # Stop stream
    service_client.stop_stream()
    return True

def main():
    args = parse_arguments()
    
    # Check if source file exists (if not RTSP)
    if not args.source.startswith("rtsp://") and not os.path.exists(args.source):
        print(f"Warning: Source file '{args.source}' not found or inaccessible")
    
    print(f"Testing with source: {args.source}")
    print(f"Test duration: {args.duration} seconds")
    print()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"combined_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create service clients
    cheat_client = ServiceClient(
        "cheat", args.cheat_host, args.cheat_port, 
        args.redis_host, args.redis_port
    )
    cheat_client.output_dir = output_dir
    
    mood_client = ServiceClient(
        "mood", args.mood_host, args.mood_port, 
        args.redis_host, args.redis_port
    )
    mood_client.output_dir = output_dir
    
    # Run services in separate threads
    cheat_thread = threading.Thread(
        target=run_service,
        args=(cheat_client, args.source, args.duration, process_cheat_event)
    )
    
    mood_thread = threading.Thread(
        target=run_service,
        args=(mood_client, args.source, args.duration, process_mood_event)
    )
    
    try:
        print("Starting cheating detection service...")
        cheat_thread.start()
        
        # Wait a bit before starting mood service
        time.sleep(2)
        
        print("Starting mood analysis service...")
        mood_thread.start()
        
        # Wait for threads to complete
        cheat_thread.join()
        mood_thread.join()
        
        print("\nTest completed successfully!")
        print(f"Results saved to {os.path.abspath(output_dir)}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during test: {str(e)}")
    
if __name__ == "__main__":
    main() 