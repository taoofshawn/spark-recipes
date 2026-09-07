FROM openresty/openresty:1.27.1.2-0-alpine

# gettext for envsubst in the entrypoint (template rendering).
RUN apk add --no-cache gettext bash

COPY nginx.conf.template /usr/local/openresty/nginx/conf/nginx.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.sh
COPY lua/ /usr/local/openresty/nginx/lua/
RUN chmod +x /usr/local/openresty/nginx/conf/nginx.conf.template /docker-entrypoint.sh && \
    mkdir -p /var/run/openresty

EXPOSE 4000

ENTRYPOINT ["/docker-entrypoint.sh"]
