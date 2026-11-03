"""5. (선택) 내 데이터로 학습 → artifacts/user_model.joblib (시뮬레이션 모델은 그대로 둠)"""
from _common import run

PATH = "user_data"
HOLDOUT = ""         # 학습에서 빼고 성능 확인에 쓸 로봇 이름, 쉼표로 구분 (예: "R3,R4")
LAP_S = None
FS = None

args = ["-u", "run_custom.py", "train", "--path", PATH, "--holdout", HOLDOUT]
if LAP_S:
    args += ["--lap-s", LAP_S]
if FS:
    args += ["--fs", FS]
run(args)
