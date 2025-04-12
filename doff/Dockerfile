FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless OpenCV
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="foo=bar"
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_DEBUG=1
ENV OPENCV_LOG_LEVEL=DEBUG

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create output directory
RUN mkdir -p mood_frames

# Expose port
EXPOSE 5002

# Run the application
CMD ["python", "mood_analyzer.py", "--redis-host", "redis", "--redis-port", "6379", "--headless", "true"] 