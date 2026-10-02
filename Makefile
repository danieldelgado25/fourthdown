SEASONS ?= 2009-2024

PORT ?= 8000

CI_SUITES ?= guard,routing_keyword,references,models

.PHONY: install format lint typecheck test ingest etl warehouse build audit index train \
	eval-routing scorecard scorecard-ci serve web-install web web-test clean

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

scorecard:
	fourthdown scorecard --output docs/scorecard.md

# What the CI eval job runs: the offline suites, all required, on the documented split.
scorecard-ci:
	fourthdown scorecard --suites $(CI_SUITES) --require $(CI_SUITES) \
		--output eval/scorecard.md --json eval/scorecard.json

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
