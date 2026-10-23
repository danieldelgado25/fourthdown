# One image per service, each with only its own pip extra (see fourthdown.services):
#   docker build --target gateway   -t fourthdown-gateway .
#   docker build --target warehouse -t fourthdown-warehouse .
#   docker build --target retrieval -t fourthdown-retrieval .
#   docker build --target models    -t fourthdown-models .
# Data (warehouse, Parquet, model artifacts) is not baked in: build and train on the host,
# then mount data/ at /data in the warehouse and models containers.

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim AS python
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FOURTHDOWN_DATA_DIR=/data
RUN useradd --system app
COPY pyproject.toml README.md ./
COPY src ./src

FROM python AS gateway
RUN pip install .
COPY --from=web /web/dist ./web/dist
ENV FOURTHDOWN_WEB_DIR=/app/web/dist
USER app
EXPOSE 8000
CMD ["python", "-m", "fourthdown.services.gateway"]

FROM python AS warehouse
RUN pip install ".[warehouse]"
USER app
EXPOSE 8001
CMD ["python", "-m", "fourthdown.services.warehouse"]

FROM python AS retrieval
RUN pip install ".[retrieval]"
USER app
EXPOSE 8002
CMD ["python", "-m", "fourthdown.services.retrieval"]

FROM python AS models
# CPU torch first; the default wheel drags in CUDA.
RUN pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[models]"
USER app
EXPOSE 8003
CMD ["python", "-m", "fourthdown.services.models"]
