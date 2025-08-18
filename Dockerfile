# Use slim image
FROM python:3.9-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install only necessary system packages
RUN apt-get update && apt-get install -y \
    libpoppler-cpp-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Download NLTK data
RUN python -m nltk.downloader -d ./nltk_data stopwords punkt

EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]