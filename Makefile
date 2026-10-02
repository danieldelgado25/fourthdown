SEASONS ?= 2009-2024

PORT ?= 8000
WP_PORT ?= 8080
WP_IMAGE ?= fourthdown-winprob
BASE_IMAGE ?= python:3.11-slim

CI_SUITES ?= guard,routing_keyword,references,models

.PHONY: install format lint typecheck test ingest etl warehouse build audit index train \
	eval-routing scorecard scorecard-ci serve lineage verify-data mlflow-ui wp-image wp-run \
	web-install web web-test clean

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

lineage:
	fourthdown lineage --output docs/lineage.md

verify-data:
	fourthdown lineage --verify --output docs/lineage.md

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db

# One image per trained model, tagged with the data version it was trained on.
wp-image:
	$(eval CARD := data/models/model_card.json)
	$(eval DATA_VERSION := $(shell python -c "import json;print(json.load(open('$(CARD)'))['data_version'])"))
	$(eval RUN_ID := $(shell python -c "import json;print(json.load(open('$(CARD)'))['mlflow_run_id'] or 'untracked')"))
	docker build -f docker/winprob.Dockerfile \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg DATA_VERSION=$(DATA_VERSION) --build-arg MLFLOW_RUN_ID=$(RUN_ID) \
		--build-arg GIT_COMMIT=$(shell git rev-parse HEAD) \
		-t $(WP_IMAGE):$(DATA_VERSION) -t $(WP_IMAGE):latest .

wp-run:
	docker run --rm -p $(WP_PORT):8080 $(WP_IMAGE):latest

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
