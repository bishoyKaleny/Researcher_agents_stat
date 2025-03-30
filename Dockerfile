FROM python:3.10-slim

# Prevent .pyc files & ensure logs are printed straight to terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY req.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Expose Gradio's default port
EXPOSE 7860

# Launch your app (update if you run a different file)
CMD ["python3", "gradio_app.py"]
