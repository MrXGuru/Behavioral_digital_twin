FROM python:3.11-slim

# Install Node.js for building frontend
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire project
COPY . .

# Build Frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Install Backend dependencies
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# The application runs on port 8000
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
