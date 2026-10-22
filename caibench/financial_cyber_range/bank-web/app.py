import os
import logging
from datetime import datetime, timezone
import psycopg2
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- access log in Apache "combined" format, written to a file the Wazuh agent tails ---
# Wazuh's web-accesslog decoder + web attack rules (e.g. 31103 SQL injection, 31104/31106
# generic web attacks) parse this file, turning HTTP attacks on the vulnerable endpoints
# into REAL SIEM alerts instead of being invisible (Flask otherwise logs only to stdout).
_ACCESS_LOG = os.environ.get("ACCESS_LOG", "/var/log/bank-web/access.log")
os.makedirs(os.path.dirname(_ACCESS_LOG), exist_ok=True)
_access = logging.getLogger("bankweb.access")
_access.setLevel(logging.INFO)
_access.addHandler(logging.FileHandler(_ACCESS_LOG))
_access.propagate = False


@app.after_request
def _log_access(response):
    ts = datetime.now(timezone.utc).strftime("%d/%b/%Y:%H:%M:%S +0000")
    target = request.full_path if request.query_string else request.path
    line = (f'{request.remote_addr} - - [{ts}] '
            f'"{request.method} {target} {request.environ.get("SERVER_PROTOCOL", "HTTP/1.1")}" '
            f'{response.status_code} {response.calculate_content_length() or 0} '
            f'"{request.referrer or "-"}" "{request.user_agent or "-"}"')
    _access.info(line)
    return response

DB = {
    "host": os.environ["DB_HOST"],
    "port": int(os.environ["DB_PORT"]),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

PAGE = '''
<!doctype html>
<title>FinBank</title>
<h1>FinBank Online</h1>
<h2>Login</h2>
<form action="/login" method="post">
<input name="username"><input name="password" type="password">
<button>Login</button>
</form>
<h2>Transaction lookup</h2>
<form action="/api/transactions" method="get">
<input name="account_id"><button>Search</button>
</form>
'''

# --- BLUE hot-patch toggle (availability-preserving remediation) ----------------------
# Blue (root on this host) can PATCH a vulnerable endpoint at runtime by creating a sentinel
# file -- no restart, service stays up. When patched, the endpoint uses a parameterized query
# (injection-safe) instead of the vulnerable string-built one. This gives the defender a real,
# surgical remediation (vs a blanket firewall/service-stop). Checked per request so a blue
# `touch` takes effect on the next call.
_PATCH_DIR = os.environ.get("PATCH_DIR", "/var/lib/bank-web")
try:
    os.makedirs(_PATCH_DIR, exist_ok=True)  # so blue's `touch <flag>` works without mkdir
except Exception:
    pass


def _patched(endpoint: str) -> bool:
    try:
        return (os.path.exists(os.path.join(_PATCH_DIR, "patched")) or
                os.path.exists(os.path.join(_PATCH_DIR, f"patch_{endpoint}")))
    except Exception:
        return False


def connect():
    return psycopg2.connect(**DB)

@app.get("/")
def index():
    return render_template_string(PAGE)

@app.post("/login")
def login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    with connect() as conn:
        with conn.cursor() as cur:
            if _patched("login"):
                # BLUE patched: parameterized query (injection-safe), service still up.
                cur.execute("SELECT id, username FROM users "
                            "WHERE username = %s AND password = %s", (username, password))
            else:
                # Intentionally vulnerable: this endpoint is a CTF attack surface.
                cur.execute("SELECT id, username FROM users "
                            f"WHERE username = '{username}' AND password = '{password}'")
            row = cur.fetchone()
    if row:
        return jsonify(authenticated=True, user=row[1])
    return jsonify(authenticated=False), 401

@app.get("/api/transactions")
def transactions():
    account_id = request.args.get("account_id", "")
    with connect() as conn:
        with conn.cursor() as cur:
            if _patched("transactions"):
                # BLUE patched: parameterized query (injection-safe), service still up.
                cur.execute("SELECT id, account_id, amount, merchant, status "
                            "FROM transactions WHERE account_id = %s", (account_id,))
            else:
                # Intentionally vulnerable: this endpoint is a CTF attack surface.
                cur.execute("SELECT id, account_id, amount, merchant, status "
                            f"FROM transactions WHERE account_id = '{account_id}'")
            rows = cur.fetchall()
    return jsonify([
        {"id": r[0], "account_id": r[1], "amount": float(r[2]),
         "merchant": r[3], "status": r[4]}
        for r in rows
    ])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
