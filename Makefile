SEASONS ?= 2009-2024

PORT ?= 8000

.PHONY: install format lint typecheck test ingest etl warehouse build audit index train \
	eval-routing serve web-install web web-test clean

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

serve:
	fourthdown serve --port $(PORT)

web-install:
	cd web && npm install

web:
	cd web && npm run dev

web-test:
	cd web && npm run build && npm run test

clean:
	rm -rf data/processed data/warehouse
