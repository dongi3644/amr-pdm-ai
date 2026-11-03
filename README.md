# AMR 고장예지 AI

배송로봇(AMR) 구동부의 진동·전류·온도로 **어느 바퀴의 무엇이 고장 나는지**와 **정지 위험까지 남은 회차**를 미리 알려 주는 파이프라인입니다.
2026 DSC 공유대학 × KT AI 활용 지역문제 해결 캠프 참가작입니다.

```
특징 추출 → Ridge(전류 잔차) → 구간별 z-score
                                 ├→ 오토인코더(건강 점수) → 칼만 필터(남은 회차)
                                 └→ 랜덤포레스트 + SHAP(원인·판단 근거)
```

## 폴더

| 경로 | 내용 |
|---|---|
| `amr_pdm/` | 본 프로젝트: 시뮬레이터, AI 파이프라인, 대시보드, 내 데이터 입력 |
| `amr_pdm_real/` | 실측 데이터 전용 버전: 가상 데이터 없이 학습, 로봇 단위 교차검증, Paderborn 공개 데이터 변환기 |
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
