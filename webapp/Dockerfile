# Static SparseTSF demo served by nginx
FROM nginx:1.27-alpine

# Remove the default nginx page and copy the site
RUN rm -rf /usr/share/nginx/html/*
COPY index.html /usr/share/nginx/html/
COPY assets/ /usr/share/nginx/html/assets/
COPY data/ /usr/share/nginx/html/data/

EXPOSE 80
HEALTHCHECK CMD wget -qO- http://localhost/ >/dev/null || exit 1
