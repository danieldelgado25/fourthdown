# The FourthDown application: the Flask API, plus the CLI that the pipeline and index
# jobs run. Holds no data -- the warehouse, models, and MLflow store live on a volume
# mounted at /data, built in place by docker/pipeline.sh.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu

# Dependencies first, against a stub package, so source edits do not reinstall them.
WORKDIR /app
COPY docker/constraints.txt /tmp/constraints.txt
COPY pyproject.toml README.md ./
RUN mkdir -p src/fourthdown && touch src/fourthdown/__init__.py \
    && pip install -c /tmp/constraints.txt ".[serve]" \
    && rm -rf src build
COPY src ./src
RUN pip install --no-deps .
COPY docker/pipeline.sh /usr/local/bin/fourthdown-pipeline

ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.title="fourthdown-app" \
      org.opencontainers.image.source="https://github.com/danieldelgado25/fourthdown" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

RUN useradd --system --uid 10001 --create-home app && mkdir /data && chown app /data
USER 10001
ENV FOURTHDOWN_DATA_DIR=/data \
    WEB_CONCURRENCY=2
VOLUME /data

# Each gunicorn worker opens its own read-only DuckDB connection and Postgres session;
# the timeout covers a CPU-bound LLM answer.
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "300", "--access-logfile", "-", \
     "fourthdown.api.app:create_app()"]
