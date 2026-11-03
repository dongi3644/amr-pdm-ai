"""2. 시뮬레이션 데이터 생성 + 학습 + 평가 → artifacts/model_bundle.joblib, reports/ (처음에는 10분 안팎)"""
from _common import run

REGEN = False      # True면 저장된 시뮬레이션 데이터를 지우고 다시 만듦

run(["-u", "run_all.py"] + (["--regen"] if REGEN else []))
