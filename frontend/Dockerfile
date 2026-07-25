# VisionEdge Lab frontend — build the SPA, serve it as static files.
#
# VITE_API_BASE is baked in at BUILD time (Vite inlines import.meta.env), so it is
# a build arg, not a runtime env var. Leave it empty for a same-origin deployment
# behind a reverse proxy: the app then falls back to window.location.origin.
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# --- serve -----------------------------------------------------------------
FROM nginx:1.27-alpine

# SPA routing: unknown paths fall back to index.html so client-side routes work.
RUN printf 'server {\n\
  listen 80;\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
  location / {\n\
    try_files $uri $uri/ /index.html;\n\
  }\n\
}\n' > /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1
