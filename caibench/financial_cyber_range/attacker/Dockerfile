# Attacker box for the financial cyber-range. Built LOCALLY from a Docker Hub base
# image -- no GitLab / pentestperf registry needed. Red runs its tools FROM here
# against the bank-web / bank-db services on the range networks.
FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        nmap \
        curl \
        wget \
        netcat-openbsd \
        iputils-ping \
        dnsutils \
        openssh-client \
        postgresql-client \
        python3 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# stay alive so the orchestrator can `docker exec` attack commands into it
CMD ["sleep", "infinity"]
