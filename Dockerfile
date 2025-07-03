# Stage 1: Build the application
FROM node:18-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsndfile1-dev \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Node.js dependency files
COPY package*.json ./
RUN npm ci

# Copy Python requirements
COPY samfair/requirements.txt ./samfair/
COPY samfair/ ./samfair/

# Create and activate Python virtual environment, install dependencies
RUN apt-get update && apt-get install -y python3-pip python3-venv nano && \
    python3 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --no-cache-dir -r samfair/requirements.txt

# Copy the entire application
COPY . .

# Build TypeScript code
RUN npm run build

# Stage 2: Create the production image
FROM node:18-slim

# Install Python runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create uploads directory and set permissions
RUN mkdir -p /app/uploads && chmod -R 777 /app/uploads

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv ./venv

# Copy built artifacts and required directories
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/samfair ./samfair
RUN chown -R 999:999 /app/samfair && chmod -R 775 /app/samfair

# Set environment variables for Python
ENV PATH="/app/venv/bin:$PATH"

# Install only production Node.js dependencies
RUN npm ci --omit=dev --no-audit --no-fund

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser

# Expose the application port
EXPOSE 5007

# Health check configuration
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5007/health || exit 1

# Start the application
CMD ["node", "dist/index.js"]