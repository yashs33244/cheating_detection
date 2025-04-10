# Cheating Detection API

This API provides real-time cheating detection from video streams using YOLOv8 and CLIP models. It uses a pub/sub architecture with Redis to deliver real-time results to clients.

## Features

- Real-time video stream processing
- Support for RTSP camera feeds, local video files, and public video URLs (including S3)
- Detection of suspicious activities during exams
- Pub/sub architecture for real-time updates
- RESTful API for stream management
- Docker support for easy deployment

## API Endpoints

### Start a Stream

```
POST /api/stream/start
```

Request body:
```json
{
  "source": "rtsp://example.com/stream",  // RTSP URL, local file path, or public URL
  "client_id": "optional-client-id"  // Optional, will generate UUID if not provided
}
```

Response:
```json
{
  "success": true,
  "client_id": "generated-client-id",
  "message": "Stream started for client generated-client-id"
}
```

### Stop a Stream

```
POST /api/stream/stop
```

Request body:
```json
{
  "client_id": "client-id"
}
```

Response:
```json
{
  "success": true,
  "message": "Stream stopped for client client-id"
}
```

### Get Stream Status

```
GET /api/stream/status
```

Response:
```json
{
  "active_streams": ["client-id-1", "client-id-2"]
}
```

### Subscribe to Stream

```
GET /api/stream/subscribe?client_id=client-id
```

Response:
```json
{
  "channel": "cheating_detection:client-id",
  "message": "Subscribe to Redis channel 'cheating_detection:client-id' to receive real-time updates"
}
```

## Running with Docker

1. Build and start the containers:
   ```
   docker-compose up -d
   ```

2. The API will be available at http://localhost:5001

## Testing the API

Use the provided test script to test the API:

```
# Test with a local video file
python test_api.py --local

# Test with an RTSP URL
python test_api.py --source rtsp://example.com/stream

# Test with a public video URL (including S3)
python test_api.py --source https://example.com/video.mp4
```

## Supported Video Sources

The API supports the following types of video sources:

1. **RTSP Camera Feeds**: For real-time camera streams
   ```
   rtsp://username:password@camera-ip:554/stream
   ```

2. **Local Video Files**: For processing local video files
   ```
   /videos/example.mp4
   ```

3. **Public Video URLs**: For processing videos hosted on public servers (including S3)
   ```
   https://example.com/video.mp4
   https://s3.amazonaws.com/bucket/video.mp4
   ```

## Redis Pub/Sub

To receive real-time updates, clients need to subscribe to the Redis channel:

1. Get the channel name from the `/api/stream/subscribe` endpoint
2. Subscribe to the Redis channel using a Redis client
3. Process messages as they arrive

Example message format:
```json
{
  "timestamp": "2023-04-09T12:34:56.789Z",
  "is_suspicious": true,
  "description": "Suspicious activity detected: students looking at each other's papers during exam (Confidence: 85.5%, People: 2, Phones: 0)"
}
```

## Architecture

The system uses a pub/sub architecture with Redis:

1. The API receives requests to start/stop streams
2. Each stream is processed in a separate thread
3. Results are published to Redis channels
4. Clients subscribe to Redis channels to receive real-time updates

This architecture allows for:
- Scalability (multiple clients can subscribe to the same stream)
- Real-time updates
- Decoupling of producers and consumers 