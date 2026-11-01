# Per-team Kali box: the team's Red (attacks the OPPONENT host) and Blue (SSH-manages its OWN
# host) operate from here, matching CAI's "one Kali VM + one CAI instance per team" setup.
FROM kalilinux/kali-rolling
# Pin to the official Kali CDN mirror directly (the default redirector 302'd pool downloads to
# a geo-mirror whose TLS cert failed to verify). kali.download served the index fine over HTTP.
RUN printf 'deb http://kali.download/kali kali-rolling main contrib non-free non-free-firmware\n' \
        > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        nmap curl netcat-traditional sshpass openssh-client \
        python3 iproute2 iputils-ping && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
CMD ["sleep", "infinity"]
