"""0. 아두이노 기록: USB로 연결한 아두이노의 센서 값(MPU6050 + INA219)을 받아 data 폴더에 CSV로 저장한다.

[먼저] 1_환경설치.py를 한 번 실행하고, 아두이노에 arduino/amr_logger/amr_logger.ino를 올려 둔다 (아두이노 IDE).
[멈추기] MINUTES가 지나면 저절로 멈춘다. 0이면 터미널에서 Ctrl+C를 누를 때까지 기록한다.
[저장] data/<ROBOT>_<날짜_시각>.csv. 같은 ROBOT 이름의 파일들은 학습·진단 때 한 로봇의 이어진 기록으로 합쳐진다.
       정상 기록과 고장 실험 기록을 다른 로봇처럼 따로 평가하고 싶으면 이름을 다르게 (예: R1_정상, R1_풀림).
"""
import os
import subprocess
import sys
from pathlib import Path

from _common import ROOT, VENV_PY

PORT = "COM3"          # 아두이노 포트. 모르면 아두이노 IDE의 '도구 → 포트'에서 확인 (틀리면 연결된 포트 목록을 보여 줌)
BAUD = 500000          # 스케치의 BAUD와 같아야 함
ROBOT = "R1"           # 로봇 이름 (파일 이름과 robot 열에 들어감)
MINUTES = 10           # 기록 시간 [분]. 정상 기준에 처음 5분을 쓰므로 6분 이상 기록할 것. 0 = Ctrl+C까지

# 고장 실험 기록일 때만 적는다 (정상 운행이면 모두 "" 그대로). 학습 때 원인 분류의 정답으로 쓰인다
SEGMENT = ""           # 장소·노면 이름 (예: "복도A"). 비우면 구간 구분 없음
FAULT_KIND = ""        # 예: "looseness"(체결부 풀림), "bearing"(베어링 결함), "unbalance"(불평형). 정상 기록이면 "normal" 또는 ""
FAULT_WHEEL = ""       # 예: "fl"(앞 왼쪽), "fr", "rl", "rr"
FAULT_SEVERITY = ""    # 고장 정도 0~1 (예: "0.5"). 모르면 비움 (고장 종류만 적으면 1로 봄)

if __name__ == "__main__":
    if not VENV_PY.exists():
        sys.exit("가상환경이 없습니다. 먼저 실행/1_환경설치.py 를 실행하세요.")
    if VENV_PY.resolve() != Path(sys.executable).resolve():
        # IDE가 다른 Python으로 실행했으면 .venv(pyserial이 설치된 환경)의 Python으로 다시 실행
        sys.exit(subprocess.call([str(VENV_PY), __file__], cwd=ROOT))
    sys.path.insert(0, str(ROOT))                          # pipeline 폴더를 불러올 수 있게
    os.chdir(ROOT)
    from pipeline.arduino import record
    print(f"기록 시작: 포트 {PORT}, 로봇 {ROBOT}, {MINUTES or '무제한'}분 (멈추려면 Ctrl+C)")
    record(PORT, ROBOT, MINUTES, BAUD, SEGMENT, FAULT_KIND, FAULT_WHEEL, FAULT_SEVERITY)
    print("다음 단계: 실행/2_데이터_검사.py")
