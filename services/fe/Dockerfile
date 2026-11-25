# Giao diện chat (HTML tĩnh) + reverse proxy sang be. Build context: services/fe
FROM nginx:1.27.5-alpine
ENV BE_URL=http://be:8002
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY static/ /usr/share/nginx/html/
