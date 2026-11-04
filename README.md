# AMR 고장예지 AI

배송로봇(AMR) 구동부에서 **어느 바퀴의 무엇이 고장 나는지**와 **정지 위험까지 남은 시간**을 미리 알려 주는 파이프라인입니다.
2026 DSC 공유대학 × KT AI 활용 지역문제 해결 캠프 참가작입니다.

실측 데이터만으로 학습하고, **바퀴 4개의 진동만** 씁니다 (전류·배터리·속도·온도 없음, 없는 값을 만들어 넣지 않음).

```
바퀴 4개 진동 → 특징 48개 → 구간별 z-score
                              ├→ 오토인코더(건강 점수) → 칼만 필터(정지 위험까지 남은 분)
                              └→ 랜덤포레스트 + SHAP(원인·고장 바퀴·판단 근거)
```

## 처음 받은 팀원용: 바로 시작하기 (Windows + VS Code)

미리 설치할 것: [Python 3.9 이상](https://www.python.org/downloads/) (설치할 때 "Add python.exe to PATH" 체크), [VS Code](https://code.visualstudio.com/), [Git](https://git-scm.com/download/win)

1. **코드 받기.** 초대 메일에서 수락한 뒤, 이 페이지의 초록색 `Code` 버튼 → 주소 복사 → 아래 명령 실행 (처음 한 번은 GitHub 로그인 창이 뜸)
   ```
   git clone https://github.com/dongi3644/amr-pdm-ai.git
   ```
   Git이 어렵다면 `Code` → `Download ZIP`으로 받아 압축을 풀어도 됩니다 (이 경우 나중에 바뀐 코드는 다시 받아야 함).
2. **VS Code로 열기.** `파일 → 파일에서 작업 영역 열기` → `AMR_고장예지AI.code-workspace`. 오른쪽 아래에 확장 설치 안내가 뜨면 `설치` (Python, Python Debugger).
3. **실행.** 왼쪽 `실행 및 디버그`(Ctrl+Shift+D) 목록에서 아래 항목을 순서대로 고르고 ▶:
   | 순서 | 항목 | 언제 |
   |---|---|---|
   | 1 | 환경 설치 | 처음 한 번 (몇 분) |
   | — | `amr_pdm_real/data/`에 CSV 넣기 | 데이터를 받을 때 |
   | 2 | 데이터 검사 | 형식 확인 |
   | 3 | 학습 + 로봇 단위 교차검증 | 모델 만들기 (데이터가 바뀌면 다시) |
   | 4 | 진단 | 로봇별 결과 → `amr_pdm_real/reports/diagnose/` |
4. **바뀐 코드 받기.** 다른 사람이 코드를 고친 뒤에는 `git pull` (VS Code 왼쪽 아래 동기화 버튼도 같음).

데이터(CSV)와 학습된 모델은 GitHub에 올라가지 않으므로 팀원끼리 따로 주고받아야 합니다. 형식은 [`amr_pdm_real/README.md`](amr_pdm_real/README.md)를 보세요.

## 폴더

| 경로 | 내용 |
|---|---|
| `amr_pdm_real/` | 실측 데이터 전용 파이프라인 (바퀴 4개 진동). 사용법·모델 구성·데이터 형식은 [`amr_pdm_real/README.md`](amr_pdm_real/README.md) |
| `AMR_고장예지AI.code-workspace` | VS Code 작업 영역 (위 2번에서 여는 파일) |
