import requests
import redis
import json
import time
import argparse
import sys

def test_ec2_api(video_source):
    base_url = "http://13.48.5.29:5001"
    
    print(f"Using video source: {video_source}")
    
    # Check stream status
    print("Getting stream status...")
    response = requests.get(f"{base_url}/api/stream/status")
    print(f"Response: {response.json()}")
    
    # Start stream
    print(f"Starting stream for {video_source}...")
    response = requests.post(
        f"{base_url}/api/stream/start",
        json={"source": video_source}
    )
    print(f"Response: {response.json()}")
    
    if not response.ok:
        print("Failed to start stream")
        return
    
    client_id = response.json()["client_id"]
    
    # Subscribe to stream
    print(f"Subscribing to stream for client {client_id}...")
    response = requests.get(f"{base_url}/api/stream/subscribe", params={"client_id": client_id})
    print(f"Response: {response.json()}")
    
    channel = response.json()["channel"]
    print(f"Channel: {channel}")
    
    # Connect to Redis
    redis_client = redis.Redis(host='13.48.5.29', port=6379)
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
        print("\nStopping stream...")
        response = requests.post(
            f"{base_url}/api/stream/stop",
            json={"client_id": client_id}
        )
        print(f"Stop response: {response.json()}")
        pubsub.unsubscribe()
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test EC2 API with video source')
    parser.add_argument('--source', type=str, required=True, help='Video source URL or path')
    args = parser.parse_args()
    
    test_ec2_api(args.source) 