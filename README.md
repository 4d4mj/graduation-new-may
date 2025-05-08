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
git clone https://github.com/4d4mj/graduation-new-may.git
cd Graduation-new
```

### 2. Environment Configuration

The application uses environment variables for configuration. Docker Compose will automatically load variables from a `.env` file located in the project root (`d:\\Projects\\Graduation-new\\.env`).

*   **Create a `.env` file in the project root directory.**
    Copy the contents of `.env.example` (if one exists) or create it from scratch.
    Example `.env` file content:
    ```env
    # PostgreSQL Settings (Docker Compose will use these for the db service if not overridden)
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=2326 # Consider changing this for production
    POSTGRES_DB=postgres

    # Backend Application Settings
    DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
    GOOGLE_API_KEY=your_google_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    # Add other necessary environment variables for the backend

    # Frontend Application Settings (if any, typically prefixed with NEXT_PUBLIC_)
    # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
    ```
*   The `compose.base.yml` file sets default PostgreSQL credentials. If you define `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` in your `.env` file, Docker Compose will use these values when creating the database service.
*   The backend service (and potentially frontend) in your `compose.dev.yml` or `compose.prod.yml` should be configured to use these environment variables. Often, no explicit `environment:` block is needed in the compose file for variables defined in the `.env` file, as Docker Compose injects them automatically. However, you might still have an `env_file:` directive pointing to the `.env` file for clarity or specific override needs.

    ```yaml
    # Example snippet for backend service in compose.dev.yml
    services:
      backend:
        # ... other configurations ...
        # env_file:
        #   - .env # Docker Compose usually picks this up automatically from the root
        # environment: # Only if you need to override or set vars not in .env
        #   - SOME_OTHER_VARIABLE=some_value
        # ...
    ```
    **Important**: Ensure your `.env` file is added to your `.gitignore` file to prevent committing secrets to your repository.

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

If not using `make`, you can use `docker-compose` directly:
```bash
# For development
docker-compose -f compose.yml -f compose.dev.yml up --build -d

# To stop
docker-compose -f compose.yml -f compose.dev.yml down
```

### 4. Database Migrations (Alembic)

After the database container is running (e.g., after `make dev`), you need to apply database migrations.

*   **Ensure your `alembic.ini` in the `backend` directory is configured correctly.** The `sqlalchemy.url` should be set to use the `DATABASE_URL` environment variable, which will be available inside the container (e.g., `sqlalchemy.url = %(DATABASE_URL)s`).

*   **To run migrations:**
    1.  Enter the backend container's interactive shell:
        ```bash
        make backend
        ```
    2.  Once inside the container's shell (you'll see a prompt like `appuser@<container_id>:/app$`), run the Alembic upgrade command:
        ```bash
        alembic upgrade head
        ```
    3.  Type `exit` to leave the container's shell.

    Alternatively, if you have a direct make command for migrations (e.g., `make migrate`), you can use that.

### 5. Seeding Data

The project includes scripts to seed the database with initial data, located in `backend/scripts/`.

*   **To run a seeding script:**
    1.  Enter the backend container's interactive shell:
        ```bash
        make backend
        ```
    2.  Once inside the container's shell, execute the desired Python script. For example:
        ```bash
        # Example for seed_doctors.py
        python scripts/seed_doctors.py

        # Example for ingest_data.py
        python scripts/ingest_data.py
        ```
    3.  Type `exit` to leave the container's shell.

    Adjust the command based on the specific script and its requirements. You might also have dedicated `make` targets for these.

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
