"""Read real, host-scoped Wazuh Indexer alerts for evaluation."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

try:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # pragma: no cover
    requests = None


class WazuhConfig:
    def __init__(self):
        self.url = os.environ.get("WAZUH_INDEXER_URL", "https://localhost:9200").rstrip("/")
        self.user = os.environ.get("WAZUH_USER", "admin")
        self.password = os.environ.get("WAZUH_PASSWORD", "")
        self.index = os.environ.get("WAZUH_ALERTS_INDEX", "wazuh-alerts-*")
        self.verify_ssl = os.environ.get("WAZUH_VERIFY_SSL", "false").lower() == "true"
        self.agent_name = os.environ.get("WAZUH_AGENT_NAME", "") or None
        self.timeout = float(os.environ.get("WAZUH_TIMEOUT", "15"))


class WazuhClient:
    def __init__(self, config: Optional[WazuhConfig] = None):
        self.config = config or WazuhConfig()
        # last_error distinguishes "Wazuh unreachable" from "reachable but 0 alerts".
        # Without it a connection failure returns [] and is silently scored as a MISS.
        self.last_error: Optional[Exception] = None

    @property
    def available(self) -> bool:
        return bool(requests and self.config.password)

    def ping(self) -> bool:
        if not self.available:
            return False
        try:
            r = requests.get(self.config.url, auth=(self.config.user, self.config.password),
                             verify=self.config.verify_ssl, timeout=self.config.timeout)
            return r.ok
        except Exception as e:  # pragma: no cover
            print(f"[wazuh] ping failed: {e}")
            return False

    def query_alerts(self, since: Optional[datetime] = None, agent_name: Optional[str] = None,
                     size: int = 200) -> list[dict]:
        """Return alert _source docs since `since` (UTC), optionally filtered by agent."""
        if not self.available:
            self.last_error = RuntimeError("Wazuh is not configured")
            return []
        must = []
        if since is not None:
            # accept either a datetime or an epoch-seconds number (time.time()); callers such as
            # mayhem._detect pass a float, so coerce it rather than crash on `.astimezone`.
            if isinstance(since, (int, float)):
                since = datetime.fromtimestamp(since, tz=timezone.utc)
            iso = since.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            must.append({"range": {"timestamp": {"gte": iso}}})
        agent = agent_name or self.config.agent_name
        if agent:
            must.append({"term": {"agent.name": agent}})
        body = {"size": size, "sort": [{"timestamp": {"order": "desc"}}],
                "query": {"bool": {"must": must or [{"match_all": {}}]}}}
        try:
            r = requests.post(f"{self.config.url}/{self.config.index}/_search",
                              json=body, auth=(self.config.user, self.config.password),
                              verify=self.config.verify_ssl, timeout=self.config.timeout)
            r.raise_for_status()
            hits = r.json().get("hits", {}).get("hits", [])
            self.last_error = None
            return [h.get("_source", {}) for h in hits]
        except Exception as e:  # pragma: no cover
            print(f"[wazuh] query failed: {e}")
            self.last_error = e
            return []

    @staticmethod
    def technique_ids(alerts: list[dict]) -> set[str]:
        """Extract MITRE technique ids from alerts' rule.mitre.id."""
        out: set[str] = set()
        for a in alerts:
            mitre = (a.get("rule", {}) or {}).get("mitre", {}) or {}
            ids = mitre.get("id") or []
            if isinstance(ids, str):
                ids = [ids]
            for t in ids:
                out.add(t)
                if "." in t:              # also record the parent technique
                    out.add(t.split(".")[0])
        return out


@lru_cache(maxsize=1)
def get_wazuh_client() -> WazuhClient:
    c = WazuhClient()
    if c.available:
        ok = c.ping()
        print(f"[wazuh] indexer={c.config.url} reachable={ok}")
    else:
        print("[wazuh] WAZUH_PASSWORD not set -> live Wazuh detection disabled.")
    return c
