# Mood Analyzer Service

This service provides real-time mood and focus analysis for classroom surveillance using Redis pub/sub for event streaming.

## Features

- Real-time face detection and emotion analysis
- Focus assessment based on detected emotions
- Phone usage detection
- Redis pub/sub for event streaming
- RESTful API for starting/stopping streams and subscribing to events

## API Endpoints

- `POST /api/mood/start`: Start a mood analysis stream
  - Request body: `{"source": "rtsp://camera_url_or_video_path", "client_id": "optional_client_id"}`
  - Response: `{"success": true, "client_id": "client_id", "message": "Stream started for client client_id"}`

- `POST /api/mood/stop`: Stop a mood analysis stream
  - Request body: `{"client_id": "client_id"}`
  - Response: `{"success": true, "message": "Stream stopped for client client_id"}`

- `GET /api/mood/status`: Get active streams
  - Response: `{"active_streams": ["client_id1", "client_id2", ...]}`

- `GET /api/mood/subscribe?client_id=client_id`: Get Redis channel for a client
  - Response: `{"channel": "mood_analysis:client_id", "message": "Subscribe to Redis channel 'mood_analysis:client_id' to receive real-time updates"}`

## Docker Setup

### Build and run using Docker Compose

```bash
docker-compose up -d
```

### Build and run using Docker

```bash
# Build Docker image
docker build -t mood_analyzer .

# Run container
docker run -p 5002:5002 --name mood_analyzer mood_analyzer
```

## Usage

1. Start the service using Docker Compose
2. Send a POST request to `/api/mood/start` with the camera or video source
3. Subscribe to the Redis channel returned from `/api/mood/subscribe`
4. Process real-time mood and focus analysis events from the Redis channel
5. Stop the stream when done by sending a POST request to `/api/mood/stop`

## Example

```python
# Start a stream
import requests

response = requests.post("http://localhost:5002/api/mood/start", json={
    "source": "rtsp://admin123:admin123@192.168.79.98:554/11"
})
client_id = response.json()["client_id"]

# Subscribe to events
import redis
r = redis.Redis(host='localhost', port=6379)
p = r.pubsub()
p.subscribe(f"mood_analysis:{client_id}")

# Process events
for message in p.listen():
    if message['type'] == 'message':
        print(message['data'])

# Stop the stream when done
requests.post("http://localhost:5002/api/mood/stop", json={
    "client_id": client_id
})
``` 