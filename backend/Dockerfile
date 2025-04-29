FROM python:3.11-slim AS base
WORKDIR /app

# Install OS‑level deps (if needed)
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends libpq-dev \
#     && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for production
RUN useradd --create-home appuser

# Development stage with hot-reload
FROM base AS development
# Keep root for development (simplifies volume mounting permissions)
EXPOSE 8000
# Command with reload enabled for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production
# Copy application code for production
COPY . .
# Use non-root user in production
USER appuser
EXPOSE 8000
# Command for production (no reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
