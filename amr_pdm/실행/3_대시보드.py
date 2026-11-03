"""3. 대시보드 실행 → 준비되면 브라우저에서 http://localhost:8501 을 연다. 끝낼 때는 IDE의 중지 버튼(또는 Ctrl+C)."""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

from _common import ROOT, VENV_PY

PORT = 8501
OPEN_BROWSER = True

if not VENV_PY.exists():
    sys.exit("가상환경이 없습니다. 먼저 실행/1_환경설치.py 를 실행하세요.")
url = f"http://localhost:{PORT}"
# 화면 없는 모드로 켜야 Streamlit이 첫 실행 때 터미널에서 이메일 입력을 기다리지 않음
proc = subprocess.Popen([str(VENV_PY), "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(PORT), "--server.headless", "true"],
                        cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
try:
    for _ in range(90):
        if proc.poll() is not None:
            sys.exit(f"대시보드가 시작되지 않았습니다 (종료 코드 {proc.returncode}). 위 메시지를 확인하세요. 포트 {PORT}가 사용 중이면 PORT를 바꾸세요.")
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=1)
            break
        except OSError:
            time.sleep(1)
    print(f"\n대시보드: {url}  (끝내려면 중지 버튼 또는 Ctrl+C)")
    if OPEN_BROWSER:
        webbrowser.open(url)
    proc.wait()
except KeyboardInterrupt:
    pass
finally:
    if proc.poll() is None:
        proc.terminate()
