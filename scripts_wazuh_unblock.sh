#!/usr/bin/env bash
# One-time Wazuh indexer unblock for --live (needs sudo). Safe to re-run.
# After this, `python main.py --live` can use REAL Wazuh detection.
set -euo pipefail
SN="$(cd "$(dirname "$0")/wazuh-docker/single-node" && pwd)"
cd "$SN"

echo "[1/4] raise vm.max_map_count (OpenSearch needs >=262144)"
sudo sysctl -w vm.max_map_count=262144
# persist across reboots
grep -q '^vm.max_map_count' /etc/sysctl.conf 2>/dev/null \
  && sudo sed -i 's/^vm.max_map_count.*/vm.max_map_count=262144/' /etc/sysctl.conf \
  || echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf >/dev/null

echo "[2/4] remove bogus cert dirs (docker auto-created *.pem as empty root dirs)"
sudo rm -rf "$SN/config/wazuh_indexer_ssl_certs"

echo "[3/4] generate indexer/manager SSL certs"
docker compose -f generate-indexer-certs.yml run --rm generator

echo "[4/4] start the full Wazuh stack (manager + indexer + dashboard)"
docker compose up -d
sleep 20
curl -sk -o /dev/null -w "indexer https://localhost:9200 -> http_code=%{http_code}\n" \
  -u admin:SecretPassword https://localhost:9200 || true
echo "done. verify agent enrolled:"
docker exec single-node-wazuh.manager-1 /var/ossec/bin/agent_control -l 2>/dev/null || true
