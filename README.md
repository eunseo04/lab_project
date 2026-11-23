# ACD + Skill-Pro + CAGE4 연구

Qwen3-8B와 KEEP 4개를 결합한 자율 사이버 방어 실험이다. Qwen 가중치를 고정하고 외부 Security Skill을 개선하며, Skill-Pro의 PPO Gate 뒤에서 실제 CAGE4 rollout으로 후보를 검증한다.

이 저장소는 **2026-11-24 시점의 연구 코드와 선별한 실험 증거**를 모은 것이다. 중단된 실험과 성능이 낮았던 결과도 보존했다. 원래 작업 디렉터리는 변경하지 않았다.

## 현재 확인된 결과

| 실험 | 실제 수행 및 해석 |
|---|---|
| v2 FiniteState | 동일 이력 ACD −183.75, 최종 loop2 −240.00. 평균 차이 −56.25. |
| v2 AggressiveFSM | ACD −262.50, 최종 Skill −211.75. 4개 시드에서 평균 +50.75. |
| v4 수동 Skill | 평균 +35.50이지만 개선은 1/4. 자동 학습 결과와 구분한다. |
| v5 B4 후보 검증 | 이전 Pool −190.55, 후보 −189.00. 20개 대응 시드의 평균 +1.55, 승률 50%. 안정성·Green 조건 미달로 거부했다. |
| v5 최종 B0–B4 테스트 | 실행되지 않았다. 100개 테스트 시드와 일반화 성능을 완료했다고 주장하지 않는다. |

세부 상태와 검증 근거는 [연구 snapshot](docs/research_snapshot.md), 이전 전체 실행 기록은 [기존 보고서](docs/execution_report.md)에 있다. 과거 보고서의 `running`·진행 예정 문구는 캡처 당시 상태다.

## 파일 구성

- `research/`, `acd/`: 기존 ACD와 Skill-Pro 이식 구현.
- `revisions/`: 실험별 보존 코드. `rollout-v5-reusable`은 여러 loop와 warm start의 provenance 처리를 보강한 실행용 버전이다.
- `research_configs/`, `configs/`: 실험 설정과 프롬프트.
- `scripts/`, `tests/`, `research_tests/`: 설치·실행·분석·회귀 검사.
- `docs/`, `reports/`: 원인 분석, 방법론, 실행 기록.
- `research_outputs/`, `analysis_outputs/`, `verification/`: 선별한 요약·표·그래프·테스트 증거.
- `artifacts/pools/`: 실제 Skill 텍스트를 보존한 경량 Pool snapshot. 원본 hash와 생략한 필드를 함께 기록했다.

5GB의 원시 실행 로그, 대형 checkpoint, 신경망 가중치, 가상환경, 외부 원본 저장소, 생성 runtime과 논문 PDF는 포함하지 않는다. [반출 목록](docs/export_manifest.json)과 [재현 범위](docs/PUBLICATION.md)를 참고한다.

## 새 환경에서 실행

기록된 CPU·모델 환경은 Python 3.10 기준이다. 모델 다운로드와 GPU 실험은 다음 명령을 직접 실행할 때 시작된다.

```bash
./scripts/setup_research.sh --download
python3 scripts/prepare_checkout.py
CUDA_VISIBLE_DEVICES=0 ./scripts/start_qwen.sh --port 8018
```

별도 터미널에서 실행한다. 학습과 검증은 오래 걸릴 수 있으므로 episode 수와 출력 경로를 명시한다.

```bash
# 동일 사건 이력 ACD
./scripts/run_research.sh B0 evaluate --episodes 4 \
  --output "$PWD/research_outputs/new-evaluation/B0"

# Proposed 학습: 설정은 4 episodes / 2 loops, 분리된 20개 검증 시드
./scripts/run_research.sh B4 train

# 완료된 학습 Pool을 명시해서 평가
./scripts/run_research.sh B4 evaluate --episodes 4 \
  --pool "$PWD/research_outputs/rollout-reproduction/B4/E3-hybrid-finite/train/pool-iteration-0002.json" \
  --output "$PWD/research_outputs/new-evaluation/B4"
```

`prepare_checkout.py`는 현재 체크아웃 경로에 맞춘 설정을 `.local-configs/`에 만든다. 보존한 원본 설정·소스·실험 hash는 바꾸지 않는다. 과거 캡처 안의 절대 경로는 원래 측정 경로다.

```bash
# 오프라인 회귀 검사: 모델 API 호출을 막는 fixture 사용
./research_v5_reusable.sh test
```

실측 수정본의 기존 전체 회귀 검사는 139개, 재사용 보강본은 141개가 통과했다. 이번 반출본에서의 재검사 결과는 [반출 검증](docs/publication_validation.json)에 기록한다. 테스트 통과와 방어 성능 개선은 별개다.

## 원문과 재사용

[출판물·원본 코드](references/README.md), [출처 표기](NOTICE.md), [보존된 라이선스](LICENSE)를 참고한다. 원래 긴 작업 기록은 [README.workspace.md](README.workspace.md)에 있다.
