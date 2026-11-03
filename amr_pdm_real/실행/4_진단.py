"""4. 로봇별 진단 (분 단위) → reports/diagnose/ 에 CSV와 그래프"""
from _common import run

PATH = "data"

run(["diagnose", "--path", PATH])
