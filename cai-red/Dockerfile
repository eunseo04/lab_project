FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y \
        git \
        curl \
        wget \
        nmap \
        net-tools \
        iputils-ping \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir cai-framework==0.5.10

CMD ["/bin/bash"]
