FROM python:3.10-slim

# Prevent python from buffering stdout/stderr (critical for real-time Render logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (ffmpeg for audio, pango/cairo for PDF generation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz-0b \
    libjpeg-dev \
    libopenjp2-7-dev \
    libffi-dev \
    shared-mime-info \
    libcairo2 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Render sets this dynamically, but Gradio uses it)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
