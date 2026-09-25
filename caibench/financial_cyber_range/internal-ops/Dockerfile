FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server python3 && rm -rf /var/lib/apt/lists/*
RUN useradd -m analyst && echo 'analyst:analyst123' | chpasswd && mkdir -p /run/sshd
COPY internal_service.py /opt/internal_service.py
EXPOSE 22 8080
CMD ["/bin/bash", "-c", "python3 /opt/internal_service.py & /usr/sbin/sshd -D"]
