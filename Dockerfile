# One image per service, each with only its own pip extra (see fourthdown.services):
#   docker build --target gateway   -t fourthdown-gateway .
#   docker build --target warehouse -t fourthdown-warehouse .
#   docker build --target retrieval -t fourthdown-retrieval .
#   docker build --target models    -t fourthdown-models .
#   docker build --target pipeline  -t fourthdown-pipeline .   # Dagster: pipeline, UI, daemon
# `make images` builds all five. No image holds data: the warehouse, Parquet, and models
# live on a volume at /data, built in place by the pipeline image (`fourthdown pipeline`,
# the Dagster asset graph) or on the host with `make build train`. Versions are pinned by
# docker/constraints.txt.

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim AS python
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FOURTHDOWN_DATA_DIR=/data
# A numeric uid, so Kubernetes' runAsNonRoot can verify it.
RUN useradd --system --uid 10001 --create-home app && mkdir /data && chown app /data
COPY docker/constraints.txt /tmp/constraints.txt
COPY pyproject.toml README.md ./
COPY src ./src
# The dbt project that builds the warehouse; read by the pipeline image (and by
# `fourthdown lineage --verify`), a few kilobytes in the others.
COPY dbt_project ./dbt_project
ENV FOURTHDOWN_DBT_PROJECT=/app/dbt_project

FROM python AS torch
# CPU torch first; the default wheel drags in CUDA.
RUN pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu

FROM python AS gateway
RUN pip install -c /tmp/constraints.txt .
COPY --from=web /web/dist ./web/dist
ENV FOURTHDOWN_WEB_DIR=/app/web/dist
USER 10001
EXPOSE 8000
CMD ["python", "-m", "fourthdown.services.gateway"]

FROM python AS warehouse
RUN pip install -c /tmp/constraints.txt ".[warehouse]"
USER 10001
EXPOSE 8001
CMD ["python", "-m", "fourthdown.services.warehouse"]

FROM python AS retrieval
RUN pip install -c /tmp/constraints.txt ".[retrieval]"
USER 10001
EXPOSE 8002
CMD ["python", "-m", "fourthdown.services.retrieval"]

FROM torch AS models
RUN pip install -c /tmp/constraints.txt ".[models]"
USER 10001
EXPOSE 8003
CMD ["python", "-m", "fourthdown.services.models"]

FROM torch AS pipeline
RUN pip install -c /tmp/constraints.txt ".[pipeline]"
# The Dagster instance: config in the image, storage on the data volume (/data/dagster).
ENV DAGSTER_HOME=/opt/dagster
COPY docker/dagster.yaml /opt/dagster/dagster.yaml
# Parse the dbt project once, here, as root: the orchestrator loads its dbt assets from
# the manifest, and the project directory is read-only to the app user at runtime.
RUN python -c "from fourthdown.data import modeling; modeling.parse_manifest()"     && rm -rf /data/* /app/dbt_project/target/logs
USER 10001
# The asset graph, once: every completed week of $FOURTHDOWN_SEASONS, then publish.
# The compose dagster services override this to serve the UI and run the daemon.
CMD ["sh", "-c", "fourthdown pipeline --seasons \"${FOURTHDOWN_SEASONS:-2009-}\" ${FOURTHDOWN_TRAIN_ARGS:-}"]
