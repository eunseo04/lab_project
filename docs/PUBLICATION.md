# GitHub 커밋 준비

이 체크아웃은 `https://github.com/eunseo04/lab_project.git`을 clone한 독립 저장소다. 원래 연구 작업 디렉터리와 홈 디렉터리의 Git index는 수정하지 않았다.

## 포함한 자료

연구 코드·설정·테스트·설치 도구·원본 revision lock·보고서·표·그래프와 선별한 측정 증거를 포함했다. 실패 실험의 요약도 남겼다. 보존한 코드·결과의 원래 hash와 경로는 원본 provenance이므로 과거 캡처 안에서 유지한다. 파일별 출처는 [반출 목록](export_manifest.json)에 있다.

`artifacts/pools/`는 대형 checkpoint에서 Skill 텍스트와 모델·학습 시드 정보만 옮긴 별도 자료다. 실제 policy 값은 바꾸지 않았으며 원본 파일 hash와 생략한 필드를 각 JSON에 표시했다. 경험 버퍼·SG 전체 이력이 없으므로 이 경량 snapshot으로 원래 훈련을 완전 재개한 것으로 간주하지 않는다.

## 로컬에 남긴 자료

모델 가중치, `.venv`/`.model-venv`, 다운로드한 `third_party`, 생성한 `runtime`, raw JSONL, pending checkpoint, 프로세스 로그/PID와 논문 전문은 제외했다. 홈 디렉터리의 shell 이력·IDE 설정·Codex 대화·인증 파일은 반출하지 않았다. 원본 자료를 삭제하지 않았다.

외부 소스와 모델은 고정한 commit/revision으로 setup에서 복원한다. 원시 궤적을 요구하는 과거 replay·report 감사는 원래 로컬 archive가 추가로 필요하다. 이 저장소만으로 대용량 원시 궤적의 모든 hash를 검증할 수 있다고 주장하지 않는다.

revision의 공통 자원 symlink는 저장소 내부의 상대 경로로 바꿨다. clone 직후 `.venv`, `runtime`, `third_party`, `models` 대상이 없는 것은 정상이며 setup 후 사용할 수 있다. 새 설정은 `scripts/prepare_checkout.py`가 `.local-configs/`에 생성한다.

## 커밋

선별한 결과는 최초 준비 시 명시적으로 스테이징했다. `.gitignore`는 앞으로 생성되는 원시 결과와 대용량 자원이 무심코 추가되지 않게 한다. 이미 추적하는 선별 결과의 수정은 일반 Git 변경으로 표시된다.

```bash
cd /home/jeonges/lab_project
python3 scripts/check_publication.py
git diff --cached --stat
git commit -m "Add ACD and Skill-Pro CAGE4 research snapshot"
git push -u origin main
```

준비 과정에서는 commit과 push를 실행하지 않았다. 반출 검사·회귀 검사의 결과는 [publication_validation.json](publication_validation.json)에 있다.
