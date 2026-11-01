FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir flask==3.0.3

# --- Wazuh agent (host-level SIEM detection) ---
ARG WAZUH_VERSION=4.14.8-1
ARG WAZUH_MANAGER=host.docker.internal
ARG WAZUH_AGENT_NAME=mayhem-host
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg procps ca-certificates iputils-ping sudo coreutils findutils \
        openssh-server iptables && \
    curl -s https://packages.wazuh.com/key/GPG-KEY-WAZUH | \
        gpg --no-default-keyring --keyring gnupg-ring:/usr/share/keyrings/wazuh.gpg --import && \
    chmod 644 /usr/share/keyrings/wazuh.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/wazuh.gpg] https://packages.wazuh.com/4.x/apt/ stable main" \
        > /etc/apt/sources.list.d/wazuh.list && \
    apt-get update && \
    WAZUH_MANAGER="${WAZUH_MANAGER}" WAZUH_AGENT_NAME="${WAZUH_AGENT_NAME}" \
        apt-get install -y wazuh-agent="${WAZUH_VERSION}" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# low-priv user that owns the RCE foothold + holds the user flag
RUN useradd -m -u 1000 -s /bin/bash ctf && mkdir -p /var/run/sshd /var/lib/shell-target

# tail the host command-audit log (RCE/privesc/flag access detection)
RUN mkdir -p /var/log/shell-target && chown ctf:ctf /var/log/shell-target && \
    printf '\n<ossec_config>\n  <localfile>\n    <log_format>syslog</log_format>\n    <location>/var/log/shell-target/cmd.log</location>\n  </localfile>\n</ossec_config>\n' \
        >> /var/ossec/etc/ossec.conf

COPY vuln.py /app/vuln.py
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
EXPOSE 8080 22
CMD ["/entrypoint.sh"]
