"""4. 로봇별 진단 (분 단위) → reports/diagnose/ 에 CSV와 그래프"""
from _common import run

PATH = "data"        # 진단할 CSV 폴더 (학습에 안 쓴 새 로봇 기록을 넣으면 '처음 보는 로봇'으로 진단)

run(["diagnose", "--path", PATH])
