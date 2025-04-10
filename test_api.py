import requests
import json
import redis
import time
import sys
import uuid
import argparse

# Configuration
API_URL = "http://localhost:5001"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

def test_start_stream(source):
    """Test starting a stream"""
    print(f"Starting stream for {source}...")
    response = requests.post(
        f"{API_URL}/api/stream/start",
        json={"source": source}
    )
    print(f"Response: {response.json()}")
    return response.json().get("client_id")

def test_stop_stream(client_id):
    """Test stopping a stream"""
    print(f"Stopping stream for client {client_id}...")
    response = requests.post(
        f"{API_URL}/api/stream/stop",
        json={"client_id": client_id}
    )
    print(f"Response: {response.json()}")

def test_stream_status():
    """Test getting stream status"""
    print("Getting stream status...")
    response = requests.get(f"{API_URL}/api/stream/status")
    print(f"Response: {response.json()}")

def subscribe_to_stream(client_id):
    """Subscribe to a stream and print results"""
    print(f"Subscribing to stream for client {client_id}...")
    
    # Get channel name
    response = requests.get(f"{API_URL}/api/stream/subscribe?client_id={client_id}")
    channel = response.json().get("channel")
    print(f"Channel: {channel}")
    
    # Connect to Redis
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel)
    
    print(f"Listening for messages on channel {channel}...")
    print("Press Ctrl+C to stop")
    
    try:
        for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                print(f"Received: {data}")
    except KeyboardInterrupt:
        print("Stopping subscription...")
        pubsub.unsubscribe()
        pubsub.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the Cheating Detection API')
    parser.add_argument('--source', type=str, help='Video source (RTSP URL, local file path, or public URL)')
    parser.add_argument('--local', action='store_true', help='Use local video file')
    args = parser.parse_args()
    
    # Determine the source
    if args.source:
        # Use the provided source
        source = args.source
    elif args.local:
        # Use the local video file
        source = "/videos/Cheating_detection (1).mp4"
    else:
        # Default to local video file
        source = "/videos/Cheating_detection (1).mp4"
    
    print(f"Using video source: {source}")
    
    # Test stream status
    test_stream_status()
    
    # Start stream
    client_id = test_start_stream(source)
    
    if client_id:
        try:
            # Subscribe to stream
            subscribe_to_stream(client_id)
        except KeyboardInterrupt:
            pass
        finally:
            # Stop stream
            test_stop_stream(client_id)
    
    # Test stream status again
    test_stream_status()

if __name__ == "__main__":
    main() 