SEASONS ?= 2009-2025

PORT ?= 8000
WP_PORT ?= 8080
WP_IMAGE ?= fourthdown-winprob
BASE_IMAGE ?= python:3.11-slim

KIND_CLUSTER ?= fourthdown
RELEASE ?= fourthdown
CHART := deploy/helm/fourthdown
HELM_ARGS ?=

CI_SUITES ?= guard,routing_keyword,references,grounding,models

# Images built from the repo Dockerfile, one per target: fourthdown-<target>.
SERVICES = gateway warehouse retrieval models
TARGETS = $(SERVICES) pipeline

# POSTGRES_PASSWORD (and the LLM endpoint) come from .env; see .env.example.
-include .env
# Credentials from .env reach helm as shell variables, so make echoes their names, not their values.
export POSTGRES_PASSWORD FOURTHDOWN_LLM_BASE_URL FOURTHDOWN_LLM_MODEL FOURTHDOWN_LLM_API_KEY

.PHONY: install format lint typecheck test ingest etl warehouse build audit index train 	dagster dagster-pipeline dagster-up 	eval-routing scorecard scorecard-ci lineage verify-data dbt-docs mlflow-ui wp-image wp-run 	images constraints up down pipeline helm-lint kind-up kind-load k8s-deploy k8s-test 	kind-down web-install web web-test clean

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

# The dbt lineage graph and model docs (dbt_project/), served on http://localhost:8081.
dbt-docs:
	fourthdown dbt docs generate
	fourthdown dbt docs serve --port 8081 --no-browser

verify-data:
	fourthdown lineage --verify --output docs/lineage.md

# The Dagster asset graph (fourthdown.orchestration) on http://localhost:3000, with its
# run history in data/dagster: the same instance `make dagster-pipeline` records into.
dagster:
	DAGSTER_HOME="$(CURDIR)/data/dagster" dagster dev -m fourthdown.orchestration -p 3000

# Run the asset graph once on the host: every completed week of SEASONS, then publish.
dagster-pipeline:
	fourthdown pipeline --seasons $(SEASONS)

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

# Every service image plus the pipeline image; none holds data.
images:
	for target in $(TARGETS); do \
		docker build --target $$target -t fourthdown-$$target:latest . || exit 1; \
	done

# Re-pin docker/constraints.txt to the current environment's runtime packages.
constraints:
	{ echo "# Runtime pins for the Dockerfile: the versions the test suite ran against."; \
	  echo "# torch is installed separately from the CPU wheel index; regenerate with \`make constraints\`."; \
	  pip freeze --exclude-editable | grep -Eiv '^(fourthdown|torch|pytest|ruff|mypy|mypy[-_]extensions|types-|pluggy|iniconfig|pathspec|librt|ast[-_]serialize)\b'; \
	} > docker/constraints.txt

# Every service in Docker; the dashboard is on http://localhost:8000.
up:
	docker compose --profile app up -d --build

down:
	docker compose --profile app --profile pipeline --profile dagster down

# Build, train, and index into ./data inside containers instead of on the host.
pipeline:
	docker compose --profile pipeline up --build pipeline

# The Dagster UI and daemon in Docker (weekly schedule + new-data sensor), on :3000.
dagster-up:
	docker compose --profile dagster up -d --build

helm-lint:
	helm lint $(CHART) --set postgres.password=lint
	helm lint $(CHART) -f $(CHART)/values-kind.yaml --set ollama.enabled=false \
		--set postgres.password=lint

# The chart on a local single-node kind cluster; the dashboard is on http://localhost:8088.
kind-up:
	kind get clusters | grep -qx $(KIND_CLUSTER) || kind create cluster --config deploy/kind/cluster.yaml

kind-load: images
	kind load docker-image --name $(KIND_CLUSTER) $(foreach t,$(TARGETS),fourthdown-$(t):latest)
	-kind load docker-image --name $(KIND_CLUSTER) $(WP_IMAGE):latest

k8s-deploy:
	helm upgrade --install $(RELEASE) $(CHART) -f $(CHART)/values-kind.yaml \
		--set postgres.password="$$POSTGRES_PASSWORD" \
		$${FOURTHDOWN_LLM_BASE_URL:+--set llm.baseUrl="$$FOURTHDOWN_LLM_BASE_URL" \
		--set llm.model="$$FOURTHDOWN_LLM_MODEL" --set llm.apiKey="$$FOURTHDOWN_LLM_API_KEY"} \
		$(HELM_ARGS)

k8s-test:
	helm test $(RELEASE) --logs

kind-down:
	kind delete cluster --name $(KIND_CLUSTER)

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

# One service in the foreground, e.g. `make serve-warehouse`; see fourthdown.services.
serve-%:
	fourthdown serve $*

web-install:
	cd web && npm install

web:
	cd web && npm run dev

web-test:
	cd web && npm run build && npm run test

clean:
	rm -rf data/processed data/warehouse
