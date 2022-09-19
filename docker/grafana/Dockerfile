FROM grafana/grafana-oss

COPY docker/grafana/config.ini /etc/grafana/grafana.ini
COPY docker/grafana/datasource.yml /etc/grafana/provisioning/datasources/default.yaml
COPY docker/grafana/dashboard.yml /etc/grafana/provisioning/dashboards/default.yaml
COPY docker/grafana/dashboards /var/lib/grafana/dashboards
