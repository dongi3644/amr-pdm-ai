"""0. 한 번에 실행: (처음이면 환경 설치) → 데이터 검사 → 학습·평가 → 진단

[역할] 1~4번 실행 파일을 차례로 부르는 묶음. 한 단계라도 실패하면 그 자리에서 멈추고 이유를 보여 준다.
       따로따로 돌리고 싶으면 1~4번 파일을 하나씩 실행하면 된다 (하는 일은 같음).
[결과] reports/ (평가 수치·그래프), artifacts/ (모델), reports/diagnose/ (로봇별 진단)
"""
import subprocess
import sys
from pathlib import Path

from _common import VENV_PY, run

PATH = "data"
BASELINE = "auto"    # auto / per_unit / reference (README.md)
EVALUATE = True      # False면 교차검증 없이 최종 모델만 학습 (빠름)

if not VENV_PY.exists():                                   # 처음 실행: 1_환경설치.py를 먼저 돌림
    print("[0/3] 가상환경이 없어 환경 설치부터 합니다", flush=True)
    code = subprocess.call([sys.executable, str(Path(__file__).with_name("1_환경설치.py"))])
    if code:
        sys.exit(code)

print("\n[1/3] 데이터 검사", flush=True)
run(["check", "--path", PATH, "--baseline", BASELINE], last=False)
print("\n[2/3] 학습 + 로봇 단위 교차검증", flush=True)
run(["train", "--path", PATH, "--baseline", BASELINE] + ([] if EVALUATE else ["--no-eval"]), last=False)
print("\n[3/3] 진단", flush=True)
run(["diagnose", "--path", PATH])
