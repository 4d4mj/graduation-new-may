# ──────────────────────────────────────────────────────────────────────────────
#  Makefile (root)
# ──────────────────────────────────────────────────────────────────────────────

# Compose files
COMPOSE_DEV = docker compose -f compose.base.yml -f compose.dev.yml
COMPOSE_PROD = docker compose -f compose.base.yml -f ompose.prod.yml

.PHONY: dev lint test gen-frontend gen-backend gen-all prod  psql backend frontend down-dev migrate ingest ingest-clear
# ─── DATA MANAGEMENT ───────────────────────────────────────────────────────────

ingest:           ## Ingest documents into the RAG vector store (appends data)
	@echo "→ Starting RAG data ingestion (appending)..."
	$(COMPOSE_DEV) run --rm backend python scripts/ingest_data.py
	@echo "→ Ingestion finished."

ingest-clear:     ## Clear RAG vector store THEN ingest documents
	@echo "→ Clearing RAG vector store and starting fresh ingestion..."
	$(COMPOSE_DEV) run --rm backend python scripts/ingest_data.py --clear
	@echo "→ Ingestion finished."

# ─── DEVELOPMENT ───────────────────────────────────────────────────────────────

dev:           ## spin up the full stack in dev mode (hot-reload)
	$(COMPOSE_DEV) up --build

down:
	$(COMPOSE_DEV) down

lint:          ## lint code in both frontend & backend
	@echo "→ Linting backend…"
	$(COMPOSE_DEV) run --rm backend   ruff check src
	@echo "→ Linting frontend…"
	$(COMPOSE_DEV) run --rm frontend-dev npm run lint

test:          ## run backend tests
	$(COMPOSE_DEV) run --rm backend pytest

migrate:		## run backend migrations
	$(COMPOSE_DEV) run --rm backend alembic upgrade head

# ─── PRODUCTION ────────────────────────────────────────────────────────────────

prod:          ## spin up prod images (uses standalone Next.js & built backend)
	$(COMPOSE_PROD) up -d --build

# ─── DATABASE MANAGEMENT ───────────────────────────────────────────────────────

.PHONY: migrate

migrate:       ## Apply database migrations using Alembic
	@echo "→ Applying database migrations..."
	$(COMPOSE_DEV) run --rm backend alembic upgrade head
	@echo "→ Migrations applied."

backend: 	  ## Open the backend shell
	docker compose -f compose.base.yml exec backend bash

frontend:
	docker compose -f compose.base.yml exec frontend-dev bash


psql:          ## Open a psql shell to the database
	docker compose -f compose.base.yml exec db psql -U postgres
