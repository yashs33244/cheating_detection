# Testing the Mood Analyzer API

This document provides instructions for testing the Mood Analyzer API using the provided `test_mood_api.py` script.

## Prerequisites

Make sure you have the following prerequisites installed:
- Python 3.x
- Required Python packages: `requests`, `redis`

You can install the required packages using:
```bash
pip install requests redis
```

## Running the Test Script

The test script connects to the Mood Analyzer API, starts a stream, subscribes to Redis events, and processes the events for a specified duration.

### Basic Usage

```bash
python test_mood_api.py --source "your_video_source"
```

### Command Line Options

- `--host`: API host (default: localhost)
- `--port`: API port (default: 5002)
- `--redis-host`: Redis host (default: localhost)
- `--redis-port`: Redis port (default: 6379)
- `--source`: Video source (RTSP URL or file path)
- `--duration`: Test duration in seconds (default: 60)

### Examples

1. **Testing with a local video file:**

```bash
python test_mood_api.py --source "Cheating_detection (1).mp4" --duration 120
```

2. **Testing with an RTSP stream:**

```bash
python test_mood_api.py --source "rtsp://admin123:admin123@192.168.79.98:554/11" --duration 300
```

3. **Testing with Docker environment:**

If you're running the services using Docker Compose, use the following command:

```bash
python test_mood_api.py --host mood_analyzer --redis-host redis --source "rtsp://admin123:admin123@192.168.79.98:554/11"
```

## Output

The script will:
1. Connect to the API and start a stream with your specified source
2. Subscribe to Redis events from the mood analyzer service
3. Process events for the specified duration
4. Save the results in a directory named `test_results_[timestamp]`
5. Create a file named `mood_results.json` containing all mood analysis events
6. Stop the stream when complete

## Interpreting Results

The mood analyzer detects:
- Face locations in each frame
- Emotional states (happy, sad, angry, neutral, etc.)
- Focus scores (0-1 scale where higher values indicate better focus)
- Phone usage detection

## Troubleshooting

If you encounter issues:

1. **API Connection Issues**: 
   - Ensure the mood analyzer service is running
   - Check the host and port settings

2. **Redis Connection Issues**:
   - Ensure Redis is running
   - Check the Redis host and port settings

3. **Video Source Issues**:
   - For RTSP: Ensure the URL is correct and the camera is accessible
   - For video files: Ensure the path is correct and the file exists

4. **Permission Issues**:
   - Ensure you have write permissions in the current directory for saving results 