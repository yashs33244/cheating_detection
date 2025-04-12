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
    parser = argparse.ArgumentParser(description='Test Mood Analyzer API')
    parser.add_argument('--host', type=str, default='13.48.5.29', help='Host where the API is running')
    parser.add_argument('--port', type=int, default=5002, help='Port where the API is running')
    parser.add_argument('--redis-host', type=str, default='13.48.5.29', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6371, help='Redis port')
    parser.add_argument('--source', type=str, required=True, 
                       help='Video source (RTSP URL or file path)')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    return parser.parse_args()

def start_stream(api_url, source):
    """Start a mood analysis stream"""
    # Check if source is RTSP or file path
    if source.startswith('rtsp://'):
        print(f"Using RTSP stream: {source}")
    else:
        if not os.path.exists(source):
            print(f"Warning: File '{source}' does not exist or is not accessible")
        print(f"Using video file: {source}")
    
    try:
        response = requests.post(f"{api_url}/api/mood/start", json={
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
        print("Please check if the mood analyzer service is running and accessible.")
        return None

def stop_stream(api_url, client_id):
    """Stop a mood analysis stream"""
    try:
        response = requests.post(f"{api_url}/api/mood/stop", json={
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
        response = requests.get(f"{api_url}/api/mood/status")
        
        if response.status_code != 200:
            print(f"Error getting stream status: {response.text}")
            return []
        
        result = response.json()
        return result["active_streams"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
        return []

def subscribe_to_events(api_url, client_id, redis_host, redis_port, duration):
    """Subscribe to mood analysis events"""
    # Get subscription details
    try:
        response = requests.get(f"{api_url}/api/mood/subscribe?client_id={client_id}")
        
        if response.status_code != 200:
            print(f"Error subscribing to events: {response.text}")
            return
        
        result = response.json()
        channel = result["channel"]
        print(f"Subscribing to channel: {channel}")
        
        # Check if there was a warning
        if "warning" in result:
            print(f"Warning: {result['warning']}")
            print("Will continue listening to the channel anyway, as it may start receiving data shortly.")
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
        return
    
    # Connect to Redis with retry logic
    max_retries = 3
    retry_count = 0
    redis_connected = False
    r = None
    
    while retry_count < max_retries and not redis_connected:
        try:
            print(f"Connecting to Redis at {redis_host}:{redis_port} (attempt {retry_count + 1}/{max_retries})...")
            r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=5.0, socket_connect_timeout=5.0)
            # Test connection
            r.ping()
            redis_connected = True
            print(f"Successfully connected to Redis at {redis_host}:{redis_port}")
        except redis.exceptions.ConnectionError as e:
            retry_count += 1
            print(f"Connection attempt {retry_count} failed: {str(e)}")
            if retry_count < max_retries:
                print(f"Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"Error: Could not connect to Redis at {redis_host}:{redis_port} after {max_retries} attempts")
                print("Please check if Redis is running and accessible.")
                return
    
    p = r.pubsub()
    p.subscribe(channel)
    
    # Create a directory for saving output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process messages for the specified duration
    start_time = time.time()
    results_file = os.path.join(output_dir, "mood_results.json")
    results = []
    
    print(f"Listening for events (duration: {duration}s)...")
    print(f"Press Ctrl+C to stop early.")
    message_count = 0
    
    # Skip the first message (subscription confirmation)
    try:
        first_message = p.get_message(timeout=2.0)
        if first_message:
            print("Received subscription confirmation")
        else:
            print("Warning: No subscription confirmation received, but proceeding anyway")
    except Exception as e:
        print(f"Warning: Error reading subscription confirmation: {str(e)}")
        print("Proceeding anyway...")
    
    rtsp_connected = False
    
    try:
        while time.time() - start_time < duration:
            try:
                message = p.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    message_count += 1
                    try:
                        data = json.loads(message['data'])
                        
                        # Check for status updates
                        if 'status' in data:
                            status = data.get('status')
                            msg = data.get('message', '')
                            
                            if status == 'connecting':
                                print(f"Status: Connecting to stream - {msg}")
                            elif status == 'connected':
                                rtsp_connected = True
                                print(f"Status: Successfully connected to stream")
                            elif status == 'loading_models':
                                print(f"Status: Loading AI models...")
                            elif status == 'models_loaded':
                                print(f"Status: AI models loaded successfully")
                            elif status == 'processing':
                                print(f"Status: Stream processing started")
                            elif status == 'reconnecting':
                                print(f"Status: Connection issues detected, trying to reconnect...")
                            elif status == 'reconnected':
                                print(f"Status: Successfully reconnected to stream")
                            elif status == 'stream_ended':
                                print(f"Status: Stream ended - {msg}")
                        
                        # Print a summary of the message
                        if 'error' in data:
                            print(f"Error: {data['error']}")
                        elif 'status' in data and data['status'] == 'frame_processed':
                            num_faces = len(data['face_locations']) if 'face_locations' in data else 0
                            has_low_focus = data.get('has_low_focus', False)
                            phone_usage = data.get('phone_usage', False)
                            emotions = data.get('emotions', [])
                            
                            # Format emotions for display
                            emotion_str = ", ".join(emotions) if emotions else "none"
                            
                            print(f"Message {message_count}: Faces={num_faces}, Low Focus={has_low_focus}, Phone Usage={phone_usage}, Emotions={emotion_str}")
                            
                            # Save result
                            results.append({
                                "timestamp": data.get('timestamp', datetime.now().isoformat()),
                                "num_faces": num_faces,
                                "has_low_focus": has_low_focus,
                                "phone_usage": phone_usage,
                                "emotions": data.get('emotions', []),
                                "focus_scores": data.get('focus_scores', [])
                            })
                            
                            # Save results to file periodically
                            if message_count % 10 == 0:
                                with open(results_file, 'w') as f:
                                    json.dump(results, f, indent=2)
                    except json.JSONDecodeError:
                        print(f"Warning: Received invalid JSON data: {message['data']}")
                    except Exception as e:
                        print(f"Error processing message: {str(e)}")
            except redis.exceptions.ConnectionError as e:
                print(f"Redis connection error: {str(e)}. Attempting to reconnect...")
                time.sleep(1)
                try:
                    # Try to reconnect
                    r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=5.0)
                    r.ping()
                    p = r.pubsub()
                    p.subscribe(channel)
                    print("Reconnected to Redis successfully")
                except Exception as reconnect_error:
                    print(f"Failed to reconnect to Redis: {str(reconnect_error)}")
                    break
            
            # If we're halfway through and haven't received any frame data, warn the user
            elapsed = time.time() - start_time
            if elapsed > duration / 2 and message_count <= 5 and not rtsp_connected:
                print("\nWarning: Not receiving enough frame data. The RTSP stream might be unavailable or inaccessible.")
                print("Check if the RTSP URL is correct and that the camera is accessible from the EC2 instance.")
                print("The EC2 instance may not be able to reach the RTSP server due to networking restrictions.")
                
            # Display a progress indicator
            progress = int((elapsed / duration) * 100)
            sys.stdout.write(f"\rProgress: {progress}% complete ({int(elapsed)}s/{duration}s)")
            sys.stdout.flush()
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nReceiving events interrupted by user.")
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReceived {message_count} messages in {int(time.time() - start_time)} seconds")
    
    if message_count == 0:
        print("\nNo data was received from the stream. Possible issues:")
        print("1. The RTSP stream URL might be incorrect or inaccessible from the EC2 instance")
        print("2. The camera might be offline or not streaming")
        print("3. There might be network connectivity issues between the EC2 instance and the camera")
        print("4. The camera might require authentication that was not provided or is incorrect")
        print("5. The video stream format might not be compatible with OpenCV")
        print("6. The EC2 instance may have outbound network restrictions preventing RTSP access")
        print("\nTry checking the EC2 server logs for more information.")
    
    print(f"Results saved to {os.path.abspath(results_file)}")

def main():
    args = parse_arguments()
    
    if not args.source:
        print("Error: No source provided. Please specify a video source with --source.")
        return
    
    api_url = f"http://{args.host}:{args.port}"
    
    print(f"Testing Mood Analyzer API at {api_url}")
    print(f"Using Redis at {args.redis_host}:{args.redis_port}")
    print(f"Video source: {args.source}")
    print(f"Test duration: {args.duration} seconds")
    
    # Check if using RTSP
    is_rtsp = args.source.startswith('rtsp://')
    if is_rtsp:
        print(f"Using RTSP stream: {args.source}")
        print("Note: RTSP streams might take longer to connect, especially on remote servers.")
        print("Attempting to connect with extended timeout...")
    
    # Start a stream with retry logic
    max_retries = 3
    retry_count = 0
    client_id = None
    
    while retry_count < max_retries and client_id is None:
        client_id = start_stream(api_url, args.source)
        if client_id is None:
            retry_count += 1
            if retry_count < max_retries:
                print(f"Connection attempt {retry_count} failed. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to connect after {max_retries} attempts.")
                return
    
    try:
        # Wait longer for RTSP to initialize
        wait_time = 10 if is_rtsp else 3
        print(f"Waiting {wait_time} seconds for stream to initialize...")
        time.sleep(wait_time)
        
        # Check if the stream is active
        active_streams = get_stream_status(api_url)
        print(f"Active streams: {active_streams}")
        
        if client_id not in active_streams:
            print(f"Warning: Stream {client_id} not found in active streams list.")
            print("This may indicate an issue with the mood analyzer service.")
            
            # If it's an RTSP stream and not found in active streams, provide more diagnostics
            if is_rtsp:
                print("\nRTSP stream connection might have failed on the server side.")
                print("Possible issues with the RTSP stream:")
                print("1. The RTSP server may be unreachable from the EC2 instance")
                print("2. The authentication credentials may be incorrect")
                print("3. The camera might be offline or using a different stream format")
                print("4. The network between the EC2 instance and the RTSP server might have firewall restrictions")
                print("\nTry checking the server logs for more information.")
            
            print("Proceeding anyway...")
        
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