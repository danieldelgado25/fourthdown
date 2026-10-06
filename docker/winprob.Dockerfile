# The win-probability model behind an HTTP API.
#
# The trained artifact and its model card are baked in, so one image is one model:
# its data version and MLflow run id are on the image labels and in every response.
# Build with `make wp-image` after `fourthdown train`.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
COPY docker/winprob-requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY data/models/winprob.pt data/models/model_card.json /models/

ARG DATA_VERSION=unknown
ARG MLFLOW_RUN_ID=unknown
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.title="fourthdown-winprob" \
      org.opencontainers.image.source="https://github.com/danieldelgado25/fourthdown" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      fourthdown.data_version="${DATA_VERSION}" \
      fourthdown.mlflow_run_id="${MLFLOW_RUN_ID}"

ENV FOURTHDOWN_MODEL_DIR=/models
RUN useradd --system --uid 10001 --create-home app
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--preload", \
     "fourthdown.serving.winprob:create_app()"]
