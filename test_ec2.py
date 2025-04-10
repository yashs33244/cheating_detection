import requests
import redis
import json
import time
import argparse
import sys
import socket

def test_ec2_api(video_source):
    base_url = "http://13.48.5.29:5001"
    redis_host = '13.48.5.29'
    redis_port = 6379
    
    print(f"Using video source: {video_source}")
    
    # Check stream status
    print("Getting stream status...")
    try:
        response = requests.get(f"{base_url}/api/stream/status")
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        return
    
    # Start stream
    print(f"Starting stream for {video_source}...")
    try:
        response = requests.post(
            f"{base_url}/api/stream/start",
            json={"source": video_source}
        )
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error starting stream: {e}")
        return
    
    if not response.ok:
        print("Failed to start stream")
        return
    
    client_id = response.json()["client_id"]
    
    # Subscribe to stream
    print(f"Subscribing to stream for client {client_id}...")
    try:
        response = requests.get(f"{base_url}/api/stream/subscribe", params={"client_id": client_id})
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error subscribing to stream: {e}")
        return
    
    channel = response.json()["channel"]
    print(f"Channel: {channel}")
    
    # Test Redis connection
    print(f"Testing Redis connection to {redis_host}:{redis_port}...")
    try:
        # Test TCP connection first
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((redis_host, redis_port))
        sock.close()
        
        if result != 0:
            print(f"Error: Cannot connect to Redis server at {redis_host}:{redis_port}")
            print("Please check if Redis is running and the port is open in the EC2 security group")
            return
            
        # Connect to Redis
        redis_client = redis.Redis(host=redis_host, port=redis_port, socket_timeout=5)
        redis_client.ping()  # Test if Redis is responding
        print("Redis connection successful!")
        
        pubsub = redis_client.pubsub()
        pubsub.subscribe(channel)
        
        print(f"Listening for messages on channel {channel}...")
        print("Press Ctrl+C to stop")
        
        # Set a timeout for the listen operation
        pubsub.subscribe(channel)
        
        # Add a counter for timeouts
        timeout_counter = 0
        max_timeouts = 3
        
        while True:
            try:
                message = pubsub.get_message(timeout=10)  # 10 second timeout
                if message is None:
                    timeout_counter += 1
                    print(f"No messages received for 10 seconds (timeout {timeout_counter}/{max_timeouts})")
                    if timeout_counter >= max_timeouts:
                        print("Too many timeouts. The stream might not be processing correctly.")
                        print("Please check the EC2 server logs for any errors.")
                        break
                    continue
                
                timeout_counter = 0  # Reset counter on successful message
                
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    print(f"Received: {data}")
            except redis.TimeoutError:
                print("Redis timeout error. The connection might be unstable.")
                break
            except Exception as e:
                print(f"Error receiving message: {e}")
                break
                
    except redis.ConnectionError as e:
        print(f"Redis connection error: {e}")
        print("Please check if Redis is running and accessible")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("\nStopping stream...")
        try:
            response = requests.post(
                f"{base_url}/api/stream/stop",
                json={"client_id": client_id}
            )
            print(f"Stop response: {response.json()}")
        except:
            print("Failed to stop stream")
        
        try:
            pubsub.unsubscribe()
        except:
            pass
        
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test EC2 API with video source')
    parser.add_argument('--source', type=str, required=True, help='Video source URL or path')
    args = parser.parse_args()
    
    test_ec2_api(args.source) 