FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all game files
COPY . .

# Cloud Run sets PORT environment variable
ENV PORT=8080

CMD gunicorn oh_hell_server:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --worker-class sync \
    --threads 1 \
    --timeout 300 \
    --log-level info \
    --capture-output \
    --enable-stdio-inheritance \
    --preload
