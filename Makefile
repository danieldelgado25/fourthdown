SEASONS ?= 2009-2024

.PHONY: install format lint typecheck test ingest etl warehouse build audit index clean

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

index:
	docker compose up -d
	fourthdown index --seasons $(SEASONS) --playoff-drives

clean:
	rm -rf data/processed data/warehouse
