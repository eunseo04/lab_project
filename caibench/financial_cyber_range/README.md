# Financial Cyber Range

CAI archive의 `easy_techcorp2` / `CobaltGroupRansomware`에서 사용한
Docker Compose + network segmentation 구조를 금융 서비스용으로 단순화한 예제.

## Topology

CAI/Red Agent -> `financial_dmz` -> `bank-web`
                                  |
                                  v
                           `financial_backend`
                             |            |
                          `bank-db`   `internal-ops`

- `bank-web`: 외부 금융 웹 서비스
- `bank-db`: 내부 DB
- `internal-ops`: 내부 운영 서버(HTTP + SSH)
- `attacker`: CAI 공격자 컨테이너 예시

## Start

```bash
docker compose up -d --build
docker compose ps
```

CAI의 attacker container를 사용한다면:

```bash
docker network connect financial_dmz <CAI_ATTACKER_CONTAINER_ID>
```

## Reset

```bash
docker compose down -v
docker compose up -d --build
```

`-v`는 DB 데이터를 초기화한다.

## Research mapping

이 환경은 SQL Injection, 내부망 접근, SSH pivoting 같은 공격이 가능한
금융형 Target 환경을 제공한다. 공격 판단 자체는 Red Agent가 담당하고,
이 디렉터리는 Target의 서비스/네트워크/상태를 제공한다.
