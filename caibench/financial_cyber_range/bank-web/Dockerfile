FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Wazuh agent: enables REAL SIEM detection on this target ---
# Installs the agent, pointed at the manager, so this host's logs reach Wazuh.
ARG WAZUH_VERSION=4.14.8-1
ARG WAZUH_MANAGER=host.docker.internal
ARG WAZUH_AGENT_NAME=financial-bank-web
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg procps ca-certificates && \
    curl -s https://packages.wazuh.com/key/GPG-KEY-WAZUH | \
        gpg --no-default-keyring --keyring gnupg-ring:/usr/share/keyrings/wazuh.gpg --import && \
    chmod 644 /usr/share/keyrings/wazuh.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/wazuh.gpg] https://packages.wazuh.com/4.x/apt/ stable main" \
        > /etc/apt/sources.list.d/wazuh.list && \
    apt-get update && \
    WAZUH_MANAGER="${WAZUH_MANAGER}" WAZUH_AGENT_NAME="${WAZUH_AGENT_NAME}" \
        apt-get install -y wazuh-agent="${WAZUH_VERSION}" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Tell the agent to tail the web access log (Apache format) so HTTP attacks are detected.
RUN mkdir -p /var/log/bank-web && \
    printf '\n<ossec_config>\n  <localfile>\n    <log_format>apache</log_format>\n    <location>/var/log/bank-web/access.log</location>\n  </localfile>\n</ossec_config>\n' \
        >> /var/ossec/etc/ossec.conf

COPY app.py .
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
EXPOSE 5000
CMD ["/entrypoint.sh"]
