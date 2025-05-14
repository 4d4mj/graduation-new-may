# STAGE 1: Node.js base - for extracting Node.js binaries if needed
FROM node:20-slim AS node-base
# No operations needed here, just a source for COPY --from=node-base

# STAGE 2: Python base with common tools and OS dependencies
FROM python:3.11-slim AS common-base
# This 'common-base' will be used by both 'development' and 'production' targets

# Set working directory
WORKDIR /app

# Prevent Python from writing bytecode files and ensure logs go to stdout immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS-level dependencies including curl and ca-certificates
# ca-certificates is crucial for HTTPS requests (like Google API calls)
# curl is useful for health checks and debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional: Copy Node.js binaries if your backend Python code or build process needs Node/NPM.
# If not, you can remove these COPY lines and the 'node-base' stage.
COPY --from=node-base /usr/local/bin/node /usr/local/bin/
COPY --from=node-base /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version \
    && npm --version

# STAGE 3: Development environment
FROM common-base AS development
# No 'AS development' needed here if it's the default target when build context specifies target=development
# The target selection in docker-compose.yml handles which final stage is used.

# Install Python dependencies for development (including development tools)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep root for development (simplifies volume mounting permissions for local dev)
# The WORKDIR /app is inherited from common-base
EXPOSE 8000

# Command with reload enabled for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]

# STAGE 4: Production environment
FROM common-base AS production

# Create a non-root user for security in production
RUN useradd --create-home appuser

# Install only production Python dependencies
# This assumes requirements.txt contains ALL dependencies (dev and prod).
# For stricter prod builds, you might have a separate requirements.prod.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.txt # --no-deps if all deps are truly in requirements.txt

# Copy application code for production
# This should be one of the last steps for better layer caching
COPY . .

# Ensure correct ownership of files for the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user for production
USER appuser

# WORKDIR /app is inherited
EXPOSE 8000

# Health check for the container
# curl is now available from common-base
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command for production (no reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]