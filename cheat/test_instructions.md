# Testing the Cheating Detection API

This document provides instructions for testing the Cheating Detection API using the provided `test_cheat_api.py` script.

## Prerequisites

Make sure you have the following prerequisites installed:
- Python 3.x
- Required Python packages: `requests`, `redis`

You can install the required packages using:
```bash
pip install requests redis
```

## Running the Test Script

The test script connects to the Cheating Detection API, starts a stream, subscribes to Redis events, and processes the events for a specified duration.

### Basic Usage

```bash
python test_cheat_api.py --source "your_video_source"
```

### Command Line Options

- `--host`: API host (default: localhost)
- `--port`: API port (default: 5003)
- `--redis-host`: Redis host (default: localhost)
- `--redis-port`: Redis port (default: 6379)
- `--source`: Video source (RTSP URL or file path)
- `--duration`: Test duration in seconds (default: 60)

### Examples

1. **Testing with a local video file:**

```bash
python test_cheat_api.py --source "Cheating_detection (1).mp4" --duration 120
```

2. **Testing with an RTSP stream:**

```bash
python test_cheat_api.py --source "rtsp://admin123:admin123@192.168.79.98:554/11" --duration 300
```

3. **Testing with Docker environment:**

If you're running the services using Docker Compose, use the following command:

```bash
python test_cheat_api.py --host cheat_detection --redis-host redis --source "rtsp://admin123:admin123@192.168.79.98:554/11"
```

## Output

The script will:
1. Connect to the API and start a stream with your specified source
2. Subscribe to Redis events from the cheating detection service
3. Process events for the specified duration
4. Save the results in a directory named `test_results_[timestamp]`
5. Create two files:
   - `cheat_results.json`: Contains all detection events
   - `cheat_incidents.json`: Contains categorized incidents (phones, looking, passing notes)
6. Stop the stream when complete

## Troubleshooting

If you encounter issues:

1. **API Connection Issues**: 
   - Ensure the cheating detection service is running
   - Check the host and port settings

2. **Redis Connection Issues**:
   - Ensure Redis is running
   - Check the Redis host and port settings

3. **Video Source Issues**:
   - For RTSP: Ensure the URL is correct and the camera is accessible
   - For video files: Ensure the path is correct and the file exists

4. **Permission Issues**:
   - Ensure you have write permissions in the current directory for saving results 