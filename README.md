# Graduation Project Setup and Execution

This document outlines the steps to set up, run, and manage the Graduation Project application.

## Overview

The project consists of a backend service (Python/FastAPI), a frontend service (Next.js), and a PostgreSQL database. Docker and Docker Compose are used for containerization and orchestration, and Make is used for simplifying common commands. Alembic is used for database migrations.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Docker and Docker Compose**:
    *   Install Docker Desktop (which includes Docker Compose) from [Docker's official website](https://www.docker.com/products/docker-desktop/).
2.  **Make**:
    *   **Windows**: You can install Make via Chocolatey (`choco install make`) or by installing Windows Subsystem for Linux (WSL) and installing Make within the Linux distribution.
    *   **macOS**: Make is typically pre-installed. If not, you can install it using Homebrew (`brew install make`).
    *   **Linux**: Install Make using your distribution's package manager (e.g., `sudo apt-get install make` for Debian/Ubuntu).

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Graduation-new
```

### 2. Environment Configuration

The application uses environment variables for configuration, especially for database connection details.

*   The default database credentials are set in `compose.base.yml`:
    *   `POSTGRES_USER=postgres`
    *   `POSTGRES_PASSWORD=2326`
    *   `POSTGRES_DB=postgres`
*   Your backend application should be configured to read its database connection string (e.g., `DATABASE_URL`) from environment variables. These can be passed to the backend service via the `compose.dev.yml` file or a `.env` file used by Docker Compose.

    Example for backend service in `compose.dev.yml`:
    ```yaml
    services:
      backend:
        # ... other configurations ...
        environment:
          - DATABASE_URL=postgresql://postgres:2326@db:5432/postgres
          - GOOGLE_API_KEY=your_google_api_key_here
          - COHERE_API_KEY=your_cohere_api_key_here
          # Add other necessary environment variables
        # ...
    ```
    **Important**: Create a `.env` file in the project root or configure your `compose.dev.yml` or `compose.prod.yml` to supply necessary API keys and other sensitive or environment-specific configurations. **Do not commit sensitive keys directly into `compose.*.yml` files if the repository is public.**

### 3. Running the Application

You can use the provided `Makefile` to simplify common tasks.

*   **Build and start all services (development mode):**
    ```bash
    make dev
    ```
    This command typically uses `docker-compose -f compose.yml -f compose.dev.yml up --build`.

*   **Stop all services:**
    ```bash
    make down
    ```
    This command typically uses `docker-compose down`.

*   **To run in production mode (ensure `compose.prod.yml` is configured):**
    ```bash
    make prod # Or the specific command defined in your Makefile for production
    ```

If not using `make`, you can use `docker-compose` directly:
```bash
# For development
docker-compose -f compose.yml -f compose.dev.yml up --build -d

# For production (ensure compose.prod.yml is correctly set up)
# docker-compose -f compose.yml -f compose.prod.yml up --build -d

# To stop
docker-compose -f compose.yml -f compose.dev.yml down # Or add compose.prod.yml
```

### 4. Database Migrations (Alembic)

After the database container is running, you need to apply database migrations. Alembic is used for this. Migrations are typically run from within the backend container or a container that has access to the backend codebase and database.

*   **Ensure your `alembic.ini` in the `backend` directory is configured to connect to the database.** The `sqlalchemy.url` should point to the Dockerized PostgreSQL instance (e.g., `postgresql://postgres:2326@db:5432/postgres`). This URL might also be configurable via an environment variable that Alembic reads.

*   **To run migrations (execute this command in a terminal that can run commands inside the backend container or has the backend environment set up):**

    If you have a make command for migrations:
    ```bash
    make migrate # Or similar command
    ```

    Otherwise, you'll need to execute the alembic command within the running backend service or a temporary one:
    ```bash
    # Find your backend service name (e.g., graduation-new-backend-1)
    docker-compose ps

    # Execute alembic upgrade head
    docker-compose exec backend alembic upgrade head
    ```
    (Replace `backend` with the actual service name of your backend application as defined in your `compose.*.yml` files).

### 5. Seeding Data

The project includes scripts to seed the database with initial data. These scripts are located in the `backend/scripts/` directory.

*   **Examples of seeding scripts:**
    *   `seed_doctors.py`
    *   `ingest_data.py`

*   **To run a seeding script (execute from within the backend container or an environment with access to it):**
    ```bash
    # Example for seed_doctors.py
    docker-compose exec backend python scripts/seed_doctors.py

    # Example for ingest_data.py
    docker-compose exec backend python scripts/ingest_data.py
    ```
    Adjust the command based on the specific script and its requirements. You might also have `make` targets for these.

## Backend and Frontend

*   **Backend**: Accessible at `http://localhost:8000` (or the port configured for the backend service).
*   **Frontend**: Accessible at `http://localhost:3000` (or the port configured for the frontend service).

## Stopping the Application

*   Using Make:
    ```bash
    make down
    ```
*   Using Docker Compose directly:
    ```bash
    docker-compose -f compose.yml -f compose.dev.yml down # (or compose.prod.yml)
    ```

## Further Configuration

*   **API Keys**: Ensure `GOOGLE_API_KEY` and `COHERE_API_KEY` (and any other external service keys) are properly set as environment variables for the backend service.

This README provides a general guide. You may need to adjust commands and configurations based on the exact content of your `Makefile` and Docker Compose files.
