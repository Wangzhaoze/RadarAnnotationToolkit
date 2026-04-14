FROM node:20-alpine

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY
ARG no_proxy

WORKDIR /workspace/apps/web

COPY apps/web/package*.json ./
RUN npm install --no-audit --no-fund

COPY apps/web /workspace/apps/web

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
