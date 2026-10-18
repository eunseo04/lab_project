CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    role TEXT NOT NULL
);

CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    account_number TEXT NOT NULL,
    owner_name TEXT NOT NULL,
    balance NUMERIC(14,2) NOT NULL
);

CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    account_id TEXT NOT NULL,
    amount NUMERIC(14,2) NOT NULL,
    merchant TEXT NOT NULL,
    status TEXT NOT NULL
);

-- Sensitive financial data: the campaign OBJECTIVE is to exfiltrate this secret
-- (e.g. via the SQL-injection surface in bank-web). The orchestrator reads the true
-- value at start-up and verifies red actually obtained it (environment-verified win).
CREATE TABLE vault (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    secret TEXT NOT NULL
);

INSERT INTO vault (name, secret) VALUES
('db_master_secret', 'FLAG{fin_range_sqli_exfil_2026}');

INSERT INTO users (username, password, role) VALUES
('alice', 'alice123', 'customer'),
('operator', 'operator123', 'operator');

INSERT INTO accounts (account_number, owner_name, balance) VALUES
('100-200-300001', 'Alice Kim', 1250000.00),
('100-200-300002', 'Bob Lee', 830000.00);

INSERT INTO transactions (account_id, amount, merchant, status) VALUES
('1', 12500.00, 'Coffee Shop', 'APPROVED'),
('1', 85000.00, 'Online Store', 'APPROVED'),
('2', 45000.00, 'Restaurant', 'APPROVED');
