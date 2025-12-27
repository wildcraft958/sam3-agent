# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements
# Assuming you have a requirements.txt, or we specify main deps here
# We'll create a requirements.txt next, so copying it is valid
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# --no-cache-dir to keep image small
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application code
WORKDIR /app
COPY sam3 /app/sam3
COPY sam3_app /app/sam3_app
COPY assets /app/assets

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Create a non-root user for security (optional but recommended)
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "sam3_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
