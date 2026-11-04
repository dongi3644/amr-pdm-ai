"""1. 처음 한 번: 프로젝트 가상환경(.venv)을 만들고 필요한 패키지를 설치한다 (몇 분 걸림)."""
import subprocess
import sys

from _common import ROOT, VENV, VENV_PY

if sys.version_info < (3, 9):
    sys.exit(f"Python 3.9 이상이 필요합니다. 지금은 {sys.version.split()[0]}입니다.")
if not VENV_PY.exists():
    print("가상환경 만드는 중:", VENV)
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
print("패키지 설치 중 (requirements.txt)...")
subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"])
subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
subprocess.check_call([str(VENV_PY), "-c", "import numpy, pandas, scipy, sklearn, shap; print('설치 확인 완료')"])
print("\n다음 단계: data 폴더에 CSV를 넣고 실행/2_데이터_검사.py")
