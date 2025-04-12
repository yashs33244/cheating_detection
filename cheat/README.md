# Cheating Detection Service

This service provides real-time cheating detection for classroom surveillance using Redis pub/sub for event streaming.

## Features

- Real-time detection of students looking at each other's papers
- Detection of phone usage during exams
- Detection of students passing notes or whispering
- Mood and focus analysis of students
- Redis pub/sub for event streaming
- RESTful API for starting/stopping streams and subscribing to events

## API Endpoints

- `POST /api/cheat/start`: Start a cheating detection stream
  - Request body: `{"source": "rtsp://camera_url_or_video_path", "client_id": "optional_client_id"}`
  - Response: `{"success": true, "client_id": "client_id", "message": "Stream started for client client_id"}`

- `POST /api/cheat/stop`: Stop a cheating detection stream
  - Request body: `{"client_id": "client_id"}`
  - Response: `{"success": true, "message": "Stream stopped for client client_id"}`

- `GET /api/cheat/status`: Get active streams
  - Response: `{"active_streams": ["client_id1", "client_id2", ...]}`

- `GET /api/cheat/subscribe?client_id=client_id`: Get Redis channel for a client
  - Response: `{"channel": "cheating_detection:client_id", "message": "Subscribe to Redis channel 'cheating_detection:client_id' to receive real-time updates"}`

## Docker Setup

### Build and run using Docker Compose

```bash
docker-compose up -d
```

### Build and run using Docker

```bash
# Build Docker image
docker build -t cheat_detection .

# Run container
docker run -p 5003:5003 --name cheat_detection cheat_detection
```

## Usage

1. Start the service using Docker Compose
2. Send a POST request to `/api/cheat/start` with the camera or video source
3. Subscribe to the Redis channel returned from `/api/cheat/subscribe`
4. Process real-time cheating detection events from the Redis channel
5. Stop the stream when done by sending a POST request to `/api/cheat/stop`

## Example

```python
# Start a stream
import requests

response = requests.post("http://localhost:5003/api/cheat/start", json={
    "source": "rtsp://admin123:admin123@192.168.79.98:554/11"
})
client_id = response.json()["client_id"]

# Subscribe to events
import redis
r = redis.Redis(host='localhost', port=6379)
p = r.pubsub()
p.subscribe(f"cheating_detection:{client_id}")

# Process events
for message in p.listen():
    if message['type'] == 'message':
        print(message['data'])

# Stop the stream when done
requests.post("http://localhost:5003/api/cheat/stop", json={
    "client_id": client_id
})
``` 