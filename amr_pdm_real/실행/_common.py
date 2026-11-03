"""실행 스크립트 공통: 프로젝트 가상환경(.venv)의 Python으로 run.py를 실행한다.

[역할] 실행/ 폴더의 번호 붙은 스크립트들이 공유하는 도우미. IDE가 어떤 Python을 골라 두었든
       항상 .venv(패키지가 설치된 환경)로 run.py를 실행하게 해서, 버튼 하나로 같은 결과가 나오게 한다.
[구조] 실행/2_데이터_검사.py → _common.run(["check", ...]) → .venv\\Scripts\\python.exe run.py check ...
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(args, last=True):
    """last=False면 성공했을 때 끝내지 않고 돌아와 다음 단계를 이어서 실행할 수 있다"""
    if not VENV_PY.exists():
        print("가상환경이 없습니다. 먼저 실행/1_환경설치.py 를 실행하세요.")
        sys.exit(1)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        code = subprocess.call([str(VENV_PY), "-u", "run.py", *map(str, args)], cwd=ROOT, env=env)
    except KeyboardInterrupt:
        code = 0
    if code:
        print(f"\n종료 코드 {code}")
    if code or last:
        sys.exit(code)
