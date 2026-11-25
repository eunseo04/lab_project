# 연구 원문과 원본 코드

원문 PDF와 전체 텍스트는 이 저장소에 포함하지 않는다. 고정한 코드 commit과 모델 revision은 루트의 `sources.lock.json`, `research_sources.lock.json`, `research_models.lock.json`에 기록했다.

- ACD: [논문 v2](https://arxiv.org/abs/2505.04843v2), [원본 코드](https://github.com/r4wd3r/llms-are-acd/tree/81ff29a0d8f7a7184cafecc62c39945d03acdda6).
- Skill-Pro: [출판 논문](https://proceedings.mlr.press/v306/mi26d.html), [원본 코드](https://github.com/Miracle1207/Skill-Pro).
- KEEP: [CAGE4 제출 코드](https://github.com/cybermonic/cage-4-submission).
- 시뮬레이터: [CAGE Challenge 4](https://github.com/cage-challenge/cage-challenge-4).

설치 스크립트가 원본 저장소를 지정 commit으로 내려받는다. 원본 저장소의 라이선스·저작권 표기를 그대로 유지한다.

Skill-Pro 코드는 3be7a9be2d4c024d132efe394d537404eba7e4c8에 고정한다. 2026-11-26 확인한 [PMLR 페이지](https://proceedings.mlr.press/v306/mi26d.html)의 제목은 “Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents”다. 기존 공식 PDF 감사 hash는 475d6ce43b9ed6ab152afe7bd190a9fb4120962d22f4b6b45aca6cde55bd984b이며 로컬 과거 보고서 paper_gate_crosscheck.json에 URL과 범위를 보관했다. 페이지 제목만으로 과거 PDF 버전의 동일성을 가정하지 않는다.

paper_rtg/paper_batch/paper_algorithm1과 공개 코드의 official_code 모드는 구분한다. 현재 CAGE4 공통조건 B0는 원논문의 모델·통신 재학습 KEEP·전체 궤적을 그대로 재현한 절대 점수가 아니다. 공식 evaluator 보상 산술 parity와 전체 actor/observation hook 재현도 서로 다른 검증이다.
