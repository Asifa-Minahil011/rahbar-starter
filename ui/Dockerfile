# Current nginx stable on Alpine 3.24 (the first image had dozens of fixable HIGH findings).
FROM nginx/nginx-unprivileged:1.30.6-alpine-slim
USER root
RUN apk upgrade --no-cache
USER 101
COPY nginx.conf /etc/nginx/nginx.conf
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
