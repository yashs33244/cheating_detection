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
    parser = argparse.ArgumentParser(description='Test Cheating Detection API')
    parser.add_argument('--host', type=str, default='localhost', help='Host where the API is running')
    parser.add_argument('--port', type=int, default=5003, help='Port where the API is running')
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--source', type=str, required=True, 
                       help='Video source (RTSP URL or file path)')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    return parser.parse_args()

def start_stream(api_url, source):
    """Start a cheating detection stream"""
    # Check if source is RTSP or file path
    if source.startswith('rtsp://'):
        print(f"Using RTSP stream: {source}")
    else:
        if not os.path.exists(source):
            print(f"Warning: File '{source}' does not exist or is not accessible")
        print(f"Using video file: {source}")
    
    try:
        response = requests.post(f"{api_url}/api/cheat/start", json={
            "source": source
        })
        
        if response.status_code != 200:
            print(f"Error starting stream: {response.text}")
            return None
        
        result = response.json()
        client_id = result["client_id"]
        print(f"Stream started for client {client_id}")
        return client_id
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
        print("Please check if the cheating detection service is running and accessible.")
        return None

def stop_stream(api_url, client_id):
    """Stop a cheating detection stream"""
    try:
        response = requests.post(f"{api_url}/api/cheat/stop", json={
            "client_id": client_id
        })
        
        if response.status_code != 200:
            print(f"Error stopping stream: {response.text}")
            return False
        
        result = response.json()
        print(f"Stream stopped: {result['message']}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
        return False

def get_stream_status(api_url):
    """Get active streams"""
    try:
        response = requests.get(f"{api_url}/api/cheat/status")
        
        if response.status_code != 200:
            print(f"Error getting stream status: {response.text}")
            return []
        
        result = response.json()
        return result["active_streams"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
        return []

def subscribe_to_events(api_url, client_id, redis_host, redis_port, duration):
    """Subscribe to cheating detection events"""
    # Get subscription details
    try:
        response = requests.get(f"{api_url}/api/cheat/subscribe?client_id={client_id}")
        
        if response.status_code != 200:
            print(f"Error subscribing to events: {response.text}")
            return
        
        result = response.json()
        channel = result["channel"]
        print(f"Subscribing to channel: {channel}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
        return
    
    # Connect to Redis
    try:
        r = redis.Redis(host=redis_host, port=redis_port)
        # Test connection
        r.ping()
        p = r.pubsub()
        p.subscribe(channel)
        print(f"Successfully connected to Redis at {redis_host}:{redis_port}")
    except redis.exceptions.ConnectionError:
        print(f"Error: Could not connect to Redis at {redis_host}:{redis_port}")
        print("Please check if Redis is running and accessible.")
        return
    
    # Create a directory for saving output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process messages for the specified duration
    start_time = time.time()
    results_file = os.path.join(output_dir, "cheat_results.json")
    incidents_file = os.path.join(output_dir, "cheat_incidents.json")
    results = []
    incidents = {
        "phones": [],
        "looking": [],
        "passing": []
    }
    
    print(f"Listening for events (duration: {duration}s)...")
    print(f"Press Ctrl+C to stop early.")
    message_count = 0
    received_final_report = False
    
    # Skip the first message (subscription confirmation)
    next(p.listen())
    
    try:
        while time.time() - start_time < duration and not received_final_report:
            message = p.get_message(timeout=1.0)
            if message and message['type'] == 'message':
                message_count += 1
                try:
                    data = json.loads(message['data'])
                    
                    # Check if this is a final report
                    if 'final' in data and data['final'] and 'report' in data:
                        print("\nReceived final report:")
                        print(json.dumps(data['report'], indent=2))
                        
                        # Save report
                        with open(incidents_file, 'w') as f:
                            json.dump(data['report'], f, indent=2)
                        
                        received_final_report = True
                        continue
                    
                    # Print a summary of the message
                    if 'error' in data:
                        print(f"Error: {data['error']}")
                    else:
                        is_suspicious = data.get('is_suspicious', False)
                        description = data.get('description', '')
                        num_faces = len(data.get('face_locations', []))
                        has_low_focus = data.get('has_low_focus', False)
                        
                        # Categorize incidents
                        if is_suspicious:
                            if "using phones" in description:
                                incident_type = "phones"
                            elif "looking at each other's papers" in description:
                                incident_type = "looking"
                            elif "passing notes" in description:
                                incident_type = "passing"
                            else:
                                incident_type = "other"
                            
                            # Add to incidents
                            incidents[incident_type].append({
                                "timestamp": data.get('timestamp'),
                                "description": description
                            })
                        
                        print(f"Message {message_count}: Suspicious={is_suspicious}, Faces={num_faces}, Low Focus={has_low_focus}")
                        if is_suspicious:
                            print(f"  Description: {description}")
                        
                        # Save result
                        results.append({
                            "timestamp": data.get('timestamp', datetime.now().isoformat()),
                            "is_suspicious": is_suspicious,
                            "description": description if is_suspicious else "",
                            "num_faces": num_faces,
                            "has_low_focus": has_low_focus,
                            "emotions": data.get('emotions', []),
                            "focus_scores": data.get('focus_scores', [])
                        })
                        
                        # Save results to file periodically
                        if message_count % 10 == 0:
                            with open(results_file, 'w') as f:
                                json.dump(results, f, indent=2)
                            with open(incidents_file, 'w') as f:
                                json.dump(incidents, f, indent=2)
                except json.JSONDecodeError:
                    print(f"Warning: Received invalid JSON data: {message['data']}")
                except Exception as e:
                    print(f"Error processing message: {str(e)}")
            
            # Display a progress indicator
            elapsed = time.time() - start_time
            progress = int((elapsed / duration) * 100)
            sys.stdout.write(f"\rProgress: {progress}% complete ({int(elapsed)}s/{duration}s)")
            sys.stdout.flush()
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nReceiving events interrupted by user.")
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(incidents_file, 'w') as f:
        json.dump(incidents, f, indent=2)
    
    print(f"\nReceived {message_count} messages in {int(time.time() - start_time)} seconds")
    print(f"Results saved to:")
    print(f"  - {os.path.abspath(results_file)}")
    print(f"  - {os.path.abspath(incidents_file)}")
    
    # Print incident summary
    phones_count = len(incidents["phones"])
    looking_count = len(incidents["looking"])
    passing_count = len(incidents["passing"])
    total_incidents = phones_count + looking_count + passing_count
    
    print(f"\nDetected {total_incidents} suspicious incidents:")
    print(f"  - Phone usage: {phones_count}")
    print(f"  - Looking at others' papers: {looking_count}")
    print(f"  - Passing notes/whispering: {passing_count}")

def main():
    args = parse_arguments()
    
    if not args.source:
        print("Error: No source provided. Please specify a video source with --source.")
        return
    
    api_url = f"http://{args.host}:{args.port}"
    
    print(f"Testing Cheating Detection API at {api_url}")
    print(f"Using Redis at {args.redis_host}:{args.redis_port}")
    print(f"Video source: {args.source}")
    print(f"Test duration: {args.duration} seconds")
    
    # Start a stream
    client_id = start_stream(api_url, args.source)
    if not client_id:
        return
    
    try:
        # Subscribe to events in the main thread for better user interaction
        subscribe_to_events(api_url, client_id, args.redis_host, args.redis_port, args.duration)
        
        # Stop the stream
        stop_stream(api_url, client_id)
    
    except KeyboardInterrupt:
        print("Test interrupted. Stopping stream...")
        stop_stream(api_url, client_id)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Stopping stream...")
        stop_stream(api_url, client_id)
    
    print("Test completed.")

if __name__ == "__main__":
    main() 