# Stage 1: Build
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Install some nice things
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    vim \
    nano \
    curl \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "🏠_Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
