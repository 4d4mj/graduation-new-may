# ──────────────────────────────────────────────────────────────────────────────
#  Makefile (root)
# ──────────────────────────────────────────────────────────────────────────────

# Compose files
COMPOSE_DEV = docker compose -f compose.base.yml -f compose.dev.yml
COMPOSE_PROD = docker compose -f compose.base.yml -f ompose.prod.yml

.PHONY: dev lint test gen-frontend gen-backend gen-all prod  psql backend frontend down-dev

# ─── DEVELOPMENT ───────────────────────────────────────────────────────────────

dev:           ## spin up the full stack in dev mode (hot-reload)
	$(COMPOSE_DEV) up --build

down-dev:
	$(COMPOSE_DEV) down -v

lint:          ## lint code in both frontend & backend
	@echo "→ Linting backend…"
	$(COMPOSE_DEV) run --rm backend   ruff check src
	@echo "→ Linting frontend…"
	$(COMPOSE_DEV) run --rm frontend-dev npm run lint

test:          ## run backend tests
	$(COMPOSE_DEV) run --rm backend pytest

# ─── PRODUCTION ────────────────────────────────────────────────────────────────

prod:          ## spin up prod images (uses standalone Next.js & built backend)
	$(COMPOSE_PROD) up -d --build

# ─── DATABASE MANAGEMENT ───────────────────────────────────────────────────────

backend: 	  ## Open the backend shell
	docker compose -f compose.base.yml exec backend bash

frontend:
	docker compose -f compose.base.yml exec frontend-dev bash


psql:          ## Open a psql shell to the database
	docker compose -f compose.base.yml exec db psql -U postgres


# Generate Zod schemas inside the frontend container
# gen-frontend:
# 	@echo "→ Generating Zod schemas…"
# 	$(COMPOSE_DEV) run --rm frontend bash -c '\
# 		set -eu ;\
# 		npm ci --prefer-offline --no-audit --progress=false ;\
# 		mkdir -p src/schemas ;\
# 		for f in /shared/schemas/*.json ; do \
# 			echo "→ processing $$f" ;\
# 			npx --yes json-schema-to-zod -i "$$f" -o "src/schemas/$$(basename "$$f" .json).ts" ;\
# 		done \
# 	'

# # Generate Pydantic models inside the backend container
# gen-backend:
# 	@echo "→ Generating Pydantic models…"
# 	$(COMPOSE_DEV) run --rm backend bash -c '\
# 		set -euo pipefail ;\
# 		mkdir -p app/schemas ;\
# 		for f in /shared/schemas/*.json ; do \
# 			echo "→ processing $$f" ;\
# 			datamodel-codegen \
# 				--encoding utf-8 \
# 				--input "$$f" --input-file-type jsonschema \
# 				--output "app/schemas/$$(basename "$$f" .json).py" \
# 				--output-model-type pydantic_v2.BaseModel \
# 				--snake-case-field ;\
# 		done \
# 	'

# # Run both generators back-to-back
# gen-all: gen-frontend gen-backend
