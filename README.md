# AMR 고장예지 AI

배송로봇(AMR) 구동부에서 **어느 바퀴의 무엇이 고장 나는지**와 **정지 위험까지 남은 시간**을 미리 알려 주는 파이프라인입니다.
2026 DSC 공유대학 × KT AI 활용 지역문제 해결 캠프 참가작입니다.

실측 데이터용 `amr_pdm_real/`은 **바퀴 4개의 진동만** 씁니다 (전류·배터리·속도·온도 없음, 없는 값을 만들어 넣지 않음).

```
바퀴 4개 진동 → 특징 48개 → 구간별 z-score
                              ├→ 오토인코더(건강 점수) → 칼만 필터(정지 위험까지 남은 분)
                              └→ 랜덤포레스트 + SHAP(원인·고장 바퀴·판단 근거)
```

## 폴더

| 경로 | 내용 |
|---|---|
| `amr_pdm_real/` | **실측 데이터 전용 버전 (진동만)**: 가상 데이터 없이 학습, 로봇 단위 교차검증. 사용법은 `amr_pdm_real/README.md` |
| `amr_pdm/` | 시뮬레이션 버전 (2륜·진동·전류·온도, 회차 단위): 시뮬레이터, AI 파이프라인, 대시보드 |
| `amr_pdm/실행방법.md` | VS Code·Visual Studio에서 실행하는 법 |
| `amr_pdm/DATA_FORMAT.md` | 내 데이터(CSV) 형식 |
| `amr_fault_sim/`, `lstm_rul.py`, `web_simulator.html` | 초기 자료 (수정하지 않음) |
| `AMR_고장예지AI_Colab.ipynb` | Colab용 노트북 (`amr_pdm/build_colab.py`로 생성) |

## 빠른 시작

```bash
cd amr_pdm
python 실행/1_환경설치.py
python 실행/3_대시보드.py
```

`amr_pdm/reports/`의 수치는 **시뮬레이션 데이터** 결과입니다. 실측 검증 전이므로 발표 수치로 그대로 쓰면 안 됩니다.
