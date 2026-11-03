"""6. 내 데이터 진단 → reports/user/ 에 로봇별 회차 결과 CSV와 그래프 저장"""
from _common import run

PATH = "user_data"
MODEL = "auto"       # auto: 내 모델이 있으면 내 모델 / user: 내 모델 / sim: 시뮬레이션 모델
LAP_S = None
FS = None

args = ["-u", "run_custom.py", "diagnose", "--path", PATH, "--model", MODEL]
if LAP_S:
    args += ["--lap-s", LAP_S]
if FS:
    args += ["--fs", FS]
run(args)
