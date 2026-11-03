"""3. 로봇(베어링) 단위 교차검증 평가 + 최종 모델 학습 → reports/ (평가 수치·그래프), artifacts/ (모델)"""
from _common import run

PATH = "data"
BASELINE = "auto"
EVALUATE = True      # False면 교차검증 없이 최종 모델만 학습 (빠름)

run(["train", "--path", PATH, "--baseline", BASELINE] + ([] if EVALUATE else ["--no-eval"]))
