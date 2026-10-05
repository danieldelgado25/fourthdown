# The React dashboard, built once and served by nginx, which also proxies /api to the
# Flask service so the browser sees one origin (as the Vite dev server does).
ARG NODE_IMAGE=node:20-alpine
ARG NGINX_IMAGE=nginx:1.27-alpine

FROM ${NODE_IMAGE} AS build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM ${NGINX_IMAGE}
COPY docker/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /web/dist /usr/share/nginx/html
ENV FOURTHDOWN_API_UPSTREAM=http://api:8000
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD wget -qO /dev/null http://127.0.0.1/healthz || exit 1
