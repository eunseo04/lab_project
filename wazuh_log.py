def extract_wazuh_context(alert):
    """
    Wazuh Alert JSON에서 공격 상태에 필요한 정보만 추출한다.
    """

    rule = alert.get("rule", {})
    agent = alert.get("agent", {})
    data = alert.get("data", {})

    context = {
        "timestamp": alert.get("timestamp"),

        "host": {
            "name": agent.get("name"),
            "ip": agent.get("ip")
        },

        "source": {
            "ip": data.get("srcip"),
            "user": data.get("srcuser")
        },

        "event": {
            "rule_id": rule.get("id"),
            "description": rule.get("description"),
            "level": rule.get("level"),
            "groups": rule.get("groups", []),
            "decoder": alert.get(
                "decoder", {}
            ).get("name")
        }
    }

    return context