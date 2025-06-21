# Use official Streamlit image
FROM python:3.9-slim

# Set environment
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy code
COPY . .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Streamlit config (optional, disables telemetry)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
