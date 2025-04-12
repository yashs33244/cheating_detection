# Classroom Monitoring System

This repository contains a comprehensive classroom monitoring system with two main services:

1. **Cheating Detection Service**: Detects suspicious activities like looking at others' papers, phone usage, and passing notes
2. **Mood Analysis Service**: Analyzes students' emotions and focus levels

Both services use Redis pub/sub for event streaming and provide RESTful APIs for integration.

## Quick Start

The easiest way to run all services together is to use the provided `docker-compose-all.yml` file:

```bash
docker compose -f docker-compose-all.yml up -d
```

This will start:
- The original cheating detection service on port 5001
- The new cheating detection service on port 5003
- The mood analyzer service on port 5002
- A shared Redis instance on port 6379

## Individual Services

Alternatively, you can run each service separately:

### Cheating Detection Service

```bash
cd cheat
docker compose up -d
```

### Mood Analysis Service

```bash
cd doff
docker compose up -d
```

## Testing the Services

Each service has a dedicated test script that can connect to the API, start a stream, and process events.

### Testing Cheating Detection

```bash
cd cheat
python test_cheat_api.py --source "path/to/video.mp4"
```

For RTSP streams:
```bash
python test_cheat_api.py --source "rtsp://username:password@camera_ip:port/path"
```

### Testing Mood Analysis

```bash
cd doff
python test_mood_api.py --source "path/to/video.mp4"
```

For RTSP streams:
```bash
python test_mood_api.py --source "rtsp://username:password@camera_ip:port/path"
```

## API Endpoints

### Cheating Detection API (port 5003)

- `POST /api/cheat/start`: Start a cheating detection stream
- `POST /api/cheat/stop`: Stop a cheating detection stream
- `GET /api/cheat/status`: Get active streams
- `GET /api/cheat/subscribe`: Get Redis channel for a client

### Mood Analysis API (port 5002)

- `POST /api/mood/start`: Start a mood analysis stream
- `POST /api/mood/stop`: Stop a mood analysis stream
- `GET /api/mood/status`: Get active streams
- `GET /api/mood/subscribe`: Get Redis channel for a client

## Detailed Documentation

For more information on each service, refer to the respective README files:

- [Cheating Detection Service](cheat/README.md)
- [Mood Analysis Service](doff/README.md)

## Troubleshooting

If you encounter issues with the tests:

1. Make sure the services are running
2. Check that the video source is accessible
3. Verify Redis is running and accessible
4. For Docker users, ensure proper network connectivity 