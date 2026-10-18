#!/bin/sh
# Start the Wazuh agent (enroll + connect to the manager), then run the Flask app.
# The agent is what makes REAL Wazuh detection possible on this target: it ships this
# host's logs to the manager, which evaluates rules and writes alerts to the indexer.
set -e

MANAGER="${WAZUH_MANAGER:-host.docker.internal}"

if [ -f /var/ossec/etc/ossec.conf ]; then
    # point the agent at the manager (covers the case where the build-time value didn't stick)
    sed -i "s#<address>[^<]*</address>#<address>${MANAGER}</address>#" /var/ossec/etc/ossec.conf || true
    # enroll + start the agent; never let an agent hiccup take down the web app
    /var/ossec/bin/wazuh-control start || echo "[entrypoint] wazuh-agent start failed (continuing)"
fi

exec python app.py
