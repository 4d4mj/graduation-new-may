# Python base image
FROM python:3.11-slim AS python-base

# Set working directory
WORKDIR /app

# Prevent Python from writing bytecode files and ensure logs go to stdout immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Node.js base - use the official Node image to extract the necessary files
FROM node:20-slim AS node-base

# Final base image combining Python with Node.js binaries
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Prevent Python from writing bytecode files and ensure logs go to stdout immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy Node.js binaries from the node image
COPY --from=node-base /usr/local/bin/node /usr/local/bin/
COPY --from=node-base /usr/local/lib/node_modules /usr/local/lib/node_modules
# Create npm symlink
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version \
    && npm --version

# Create non-root user for production
RUN useradd --create-home appuser

# Development stage
FROM base AS development

# Install Python dependencies for development (including development tools)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep root for development (simplifies volume mounting permissions)
EXPOSE 8000

# Command with reload enabled for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production

# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code for production
COPY . .

# Ensure correct ownership of files
RUN chown -R appuser:appuser /app

# Switch to non-root user for production
USER appuser

# Expose the port
EXPOSE 8000

# Health check for the container
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command for production (no reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
