SEASONS ?= 2009-2024

.PHONY: install format lint typecheck test ingest etl warehouse build audit index train \
	eval-routing up down kind-up kind-forward kind-down web-install web web-test clean

install:
	python -m pip install -e ".[dev]"

format:
	ruff format .

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

test:
	python -m pytest

ingest:
	fourthdown ingest --seasons $(SEASONS)

etl:
	fourthdown etl --seasons $(SEASONS)

warehouse:
	fourthdown warehouse

build: ingest etl warehouse

audit:
	fourthdown audit --output docs/data_audit.md

train:
	fourthdown train --output docs/model_eval.md

index:
	docker compose up -d
	fourthdown index --seasons $(SEASONS) --playoff-drives

eval-routing:
	fourthdown eval-routing --output docs/routing_eval.md

# One service in the foreground, e.g. `make serve-warehouse`; see fourthdown.services.
serve-%:
	fourthdown serve $*

# Every service in Docker; the dashboard is on http://localhost:8000.
up:
	docker compose --profile app up -d --build

down:
	docker compose --profile app down

# The Helm chart on a local single-node kind cluster, with the images built here and
# loaded into it. POSTGRES_PASSWORD comes from .env. Then `make kind-forward`.
-include .env
SERVICES = gateway warehouse retrieval models

kind-up:
	kind get clusters | grep -qx fourthdown || kind create cluster --config deploy/kind.yaml
	for service in $(SERVICES); do \
		docker build --target $$service -t fourthdown-$$service . && \
		kind load docker-image fourthdown-$$service --name fourthdown || exit 1; \
	done
	helm upgrade --install fourthdown deploy/helm/fourthdown \
		--set postgres.password="$(POSTGRES_PASSWORD)"

kind-forward:
	kubectl port-forward svc/fourthdown-gateway 8000:8000

kind-down:
	kind delete cluster --name fourthdown

web-install:
	cd web && npm install

web:
	cd web && npm run dev

web-test:
	cd web && npm run build && npm run test

clean:
	rm -rf data/processed data/warehouse
