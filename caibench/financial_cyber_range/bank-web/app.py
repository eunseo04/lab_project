import os
import psycopg2
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

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

def connect():
    return psycopg2.connect(**DB)

@app.get("/")
def index():
    return render_template_string(PAGE)

@app.post("/login")
def login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    # Intentionally vulnerable: this endpoint is a CTF attack surface.
    query = (
        "SELECT id, username FROM users "
        f"WHERE username = '{username}' AND password = '{password}'"
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
    if row:
        return jsonify(authenticated=True, user=row[1])
    return jsonify(authenticated=False), 401

@app.get("/api/transactions")
def transactions():
    account_id = request.args.get("account_id", "")
    # Intentionally vulnerable: this endpoint is a CTF attack surface.
    query = (
        "SELECT id, account_id, amount, merchant, status "
        f"FROM transactions WHERE account_id = '{account_id}'"
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
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
