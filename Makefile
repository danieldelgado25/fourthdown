SEASONS ?= 2009-2024

PORT ?= 8000
WP_PORT ?= 8080
WP_IMAGE ?= fourthdown-winprob
APP_IMAGE ?= fourthdown-app
WEB_IMAGE ?= fourthdown-web
BASE_IMAGE ?= python:3.11-slim
NODE_IMAGE ?= node:20-alpine
NGINX_IMAGE ?= nginx:1.27-alpine

KIND_CLUSTER ?= fourthdown
RELEASE ?= fourthdown
CHART := deploy/helm/fourthdown
HELM_ARGS ?=

CI_SUITES ?= guard,routing_keyword,references,models

.PHONY: install format lint typecheck test ingest etl warehouse build audit index train \
	eval-routing scorecard scorecard-ci serve lineage verify-data mlflow-ui wp-image wp-run \
	app-image web-image images constraints stack-up stack-down helm-lint kind-up kind-load \
	k8s-deploy k8s-test kind-down web-install web web-test clean

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

# The API/pipeline image and the dashboard image; neither holds data.
app-image:
	docker build -f docker/app.Dockerfile --build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg GIT_COMMIT=$(shell git rev-parse HEAD) -t $(APP_IMAGE):latest .

web-image:
	docker build -f docker/web.Dockerfile --build-arg NODE_IMAGE=$(NODE_IMAGE) \
		--build-arg NGINX_IMAGE=$(NGINX_IMAGE) -t $(WEB_IMAGE):latest .

images: app-image web-image wp-image

# Re-pin docker/constraints.txt to the current environment's runtime packages.
constraints:
	{ echo "# Runtime pins for docker/app.Dockerfile: the versions the test suite ran against."; \
	  echo "# torch is installed separately from the CPU wheel index; regenerate with \`make constraints\`."; \
	  pip freeze --exclude-editable | grep -Eiv '^(fourthdown|torch|pytest|ruff|mypy|mypy[-_]extensions|types-|pluggy|iniconfig|pathspec|librt|ast[-_]serialize)\b'; \
	} > docker/constraints.txt

stack-up:
	docker compose --profile stack up -d

stack-down:
	docker compose --profile stack down

helm-lint:
	helm lint $(CHART)
	helm lint $(CHART) -f $(CHART)/values-kind.yaml --set ollama.enabled=false

kind-up:
	kind create cluster --config deploy/kind/cluster.yaml

kind-load:
	kind load docker-image --name $(KIND_CLUSTER) $(APP_IMAGE):latest $(WEB_IMAGE):latest $(WP_IMAGE):latest

k8s-deploy:
	helm upgrade --install $(RELEASE) $(CHART) -f $(CHART)/values-kind.yaml $(HELM_ARGS)

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
