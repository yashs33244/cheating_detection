# Testing Both Services Together

This document provides instructions for testing both the Cheating Detection and Mood Analysis services together using the provided `combined_test.py` script.

## Prerequisites

Make sure you have the following prerequisites installed:
- Python 3.x
- Required Python packages: `requests`, `redis`

You can install the required packages using:
```bash
pip install requests redis
```

## Running the Combined Test

The combined test script connects to both APIs, starts streams, subscribes to Redis events, and processes the events simultaneously.

### Basic Usage

```bash
python combined_test.py --source "your_video_source"
```

### Command Line Options

- `--source`: Video source (RTSP URL or file path) - **Required**
- `--duration`: Test duration in seconds (default: 60)

#### Cheating Detection Service Options
- `--cheat-host`: Cheating detection host (default: localhost)
- `--cheat-port`: Cheating detection port (default: 5003)

#### Mood Analysis Service Options
- `--mood-host`: Mood analysis host (default: localhost)
- `--mood-port`: Mood analysis port (default: 5002)

#### Redis Options
- `--redis-host`: Redis host (default: localhost)
- `--redis-port`: Redis port (default: 6379)

### Examples

1. **Testing with a local video file:**

```bash
python combined_test.py --source "Cheating_detection (1).mp4" --duration 120
```

2. **Testing with an RTSP stream:**

```bash
python combined_test.py --source "rtsp://admin123:admin123@192.168.79.98:554/11" --duration 300
```

3. **Testing with Docker environment:**

If you're running the services using Docker Compose, use the following command:

```bash
python combined_test.py --cheat-host cheat_detection --mood-host mood_analyzer --redis-host redis --source "rtsp://admin123:admin123@192.168.79.98:554/11"
```

## Output

The script will:
1. Connect to both APIs and start streams with your specified source
2. Subscribe to Redis events from both services
3. Process events from both services simultaneously
4. Save the results in a directory named `combined_results_[timestamp]`
5. Create two files:
   - `cheat_results.json`: Contains all cheating detection events
   - `mood_results.json`: Contains all mood analysis events
6. Stop the streams when complete

## Troubleshooting

If you encounter issues:

1. **API Connection Issues**: 
   - Ensure both services are running
   - Check the host and port settings for each service

2. **Redis Connection Issues**:
   - Ensure Redis is running
   - Check the Redis host and port settings

3. **Video Source Issues**:
   - For RTSP: Ensure the URL is correct and the camera is accessible
   - For video files: Ensure the path is correct and the file exists

4. **Stopping the Test**:
   - You can press Ctrl+C at any time to stop the test
   - The script will attempt to gracefully stop all streams 