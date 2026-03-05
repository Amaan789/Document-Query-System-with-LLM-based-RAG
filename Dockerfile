# Base Python image
FROM public.ecr.aws/docker/library/python:3.12.0-slim-bullseye

# Add Lambda Web Adapter extension
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.3 \
/lambda-adapter /opt/extensions/lambda-adapter

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/opt/nltk_data
ENV PYPPETEER_HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    gobject-introspection \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create non-root user
RUN useradd -ms /bin/bash admin

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install nltk resources at build time
RUN mkdir -p /opt/nltk_data && \
    python -m nltk.downloader punkt -d /opt/nltk_data && \
    python -m nltk.downloader punkt_tab -d /opt/nltk_data && \
    python -m nltk.downloader stopwords -d /opt/nltk_data

# Copy application code
COPY . .

# Fix permissions
RUN chown -R admin:admin /app
RUN chmod -R 755 /app

# Switch to non-root user
USER admin

# Start FastAPI
CMD ["python", "main.py"]