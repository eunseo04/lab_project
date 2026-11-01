#!/bin/sh
# Provision the vulnerable box, start the Wazuh agent, then run the RCE app as low-priv 'ctf'.
set -e
MANAGER="${WAZUH_MANAGER:-host.docker.internal}"

# --- flags (fresh each boot) ---
USER_FLAG="FLAG{shell_user_$(head -c6 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
ROOT_FLAG="FLAG{shell_root_$(head -c6 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
printf '%s\n' "$USER_FLAG" > /home/ctf/user.txt
chown ctf:ctf /home/ctf/user.txt && chmod 644 /home/ctf/user.txt
printf '%s\n' "$ROOT_FLAG" > /root/root.txt
chown root:root /root/root.txt && chmod 600 /root/root.txt

# --- privilege-escalation misconfig: ctf may run find as root without a password ---
# (GTFOBins: `sudo find <path> -exec cat /root/root.txt \;` -> root-flag read)
echo 'ctf ALL=(root) NOPASSWD: /usr/bin/find' > /etc/sudoers.d/ctf
chmod 440 /etc/sudoers.d/ctf

# --- Wazuh agent (host-level detection) ---
if [ -f /var/ossec/etc/ossec.conf ]; then
    sed -i "s#<address>[^<]*</address>#<address>${MANAGER}</address>#" /var/ossec/etc/ossec.conf || true
    /var/ossec/bin/wazuh-control start || echo "[entrypoint] wazuh-agent start failed (continuing)"
fi

# --- run the vulnerable app AS the unprivileged user (RCE foothold = ctf) ---
touch /var/log/shell-target/cmd.log && chown ctf:ctf /var/log/shell-target/cmd.log
export APP_USER=ctf
exec runuser -u ctf -- python /app/vuln.py
