"""실행 스크립트 공통: 프로젝트 가상환경(.venv)의 Python으로 명령을 실행한다.
VS Code나 Visual Studio가 어떤 Python을 고르고 있든 항상 같은 환경에서 돌도록 하기 위함."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(args):
    if not VENV_PY.exists():
        print("가상환경이 없습니다. 먼저 실행/1_환경설치.py 를 실행하세요.")
        sys.exit(1)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        code = subprocess.call([str(VENV_PY), *map(str, args)], cwd=ROOT, env=env)
    except KeyboardInterrupt:
        code = 0
    if code:
        print(f"\n종료 코드 {code}")
    sys.exit(code)
